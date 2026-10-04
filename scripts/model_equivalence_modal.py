#!/usr/bin/env python3
"""Cross-environment SantaCoder conversion equivalence check.

The native checkpoint is evaluated with its contemporary Transformers release;
TransformerLens is evaluated in the thesis's frozen environment.  Comparing
both in Transformers 4.57 is invalid because SantaCoder's legacy remote class no
longer matches the modern GPT-2 block API.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("mechinterp-santacoder-equivalence")
volume = modal.Volume.from_name("mechinterp-hf-cache", create_if_missing=True)
native_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.6.0", "transformers==4.28.1", "huggingface_hub==0.36.2",
)
hooked_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.6", "transformer-lens==2.17.0",
        "typeguard==4.5.1", "einops==0.8.2", "numpy==1.26.4",
        "huggingface_hub==0.36.2",
    )
    .add_local_python_source("scripts")
)

REVISION = "bb3be599767d93ce716293e9193c027e855a9524"
PROMPTS = (
    "def add(a, b):\n    return a + b",
    "result = matrix[index][column",
    "function f(x) {\n  return call(x",
    "alpha beta gamma alpha beta",
)


@app.function(image=native_image, gpu="A10G", timeout=60 * 15, volumes={"/cache": volume})
def native_reference() -> dict:
    import os
    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "bigcode/santacoder"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=REVISION, trust_remote_code=True
    ).eval().cuda()
    rows = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            logits = model(tokens).logits[0, -1].float().cpu()
            rows.append({"prompt": prompt, "tokens": tokens[0].cpu().tolist(), "logits": logits.tolist()})
    return {"transformers": "4.28.1", "rows": rows}


@app.function(image=hooked_image, gpu="A10G", timeout=60 * 15, volumes={"/cache": volume})
def hooked_reference() -> dict:
    import os
    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    import torch
    from scripts.benchmark import load_model

    model = load_model(device="cuda")
    rows = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            tokens = model.to_tokens(prompt, prepend_bos=False)
            logits = model(tokens)[0, -1].float().cpu()
            rows.append({"prompt": prompt, "tokens": tokens[0].cpu().tolist(), "logits": logits.tolist()})
    return {"transformers": "4.57.6", "transformer_lens": "2.17.0", "rows": rows}


@app.local_entrypoint()
def main(save_path: str = "outputs/reproducibility/model_equivalence.json"):
    import numpy as np

    native = native_reference.remote()
    hooked = hooked_reference.remote()
    rows = []
    for native_row, hooked_row in zip(native["rows"], hooked["rows"]):
        if native_row["prompt"] != hooked_row["prompt"]:
            raise RuntimeError("Prompt order mismatch")
        if native_row["tokens"] != hooked_row["tokens"]:
            raise RuntimeError(f"Tokenizer mismatch for {native_row['prompt']!r}")
        left = np.asarray(native_row["logits"], dtype=np.float64)
        right = np.asarray(hooked_row["logits"], dtype=np.float64)
        difference = left - right
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        centered_difference = left_centered - right_centered
        cosine = float(
            np.dot(left_centered, right_centered)
            / (np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        )
        rows.append({
            "prompt": native_row["prompt"],
            "n_tokens": len(native_row["tokens"]),
            "max_absolute_logit_difference": float(np.abs(difference).max()),
            "mean_absolute_logit_difference": float(np.abs(difference).mean()),
            "mean_logit_offset": float(difference.mean()),
            "max_absolute_centered_logit_difference": float(np.abs(centered_difference).max()),
            "centered_logit_cosine_similarity": cosine,
            "top_token_matches": int(left.argmax()) == int(right.argmax()),
        })
    tolerance = 2e-3
    payload = {
        "model": "bigcode/santacoder", "revision": REVISION,
        "native_environment": {key: value for key, value in native.items() if key != "rows"},
        "hooked_environment": {key: value for key, value in hooked.items() if key != "rows"},
        "rows": rows,
        "overall_max_absolute_centered_logit_difference": max(
            row["max_absolute_centered_logit_difference"] for row in rows
        ),
        "tolerance": tolerance,
    }
    payload["passed"] = payload["overall_max_absolute_centered_logit_difference"] < tolerance
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit("TransformerLens/native logit equivalence failed")
