#!/usr/bin/env python3
"""Run the SantaCoder induction benchmark on Modal.

Usage:
    uv run modal run scripts/modal_run.py
    uv run modal run scripts/modal_run.py --n-prompts 10 --seq-len 10 --seed 1

This script keeps the research logic in `scripts.benchmark`
and only moves execution to a GPU-backed Modal function.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("mechinterp-santacoder-induction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "transformer-lens==2.17.0",
        "typeguard==4.5.1",
        "einops==0.8.2",
        "numpy==1.26.4",
        "pandas==2.3.3",
        "matplotlib==3.10.8",
        "tqdm==4.67.3",
        "huggingface_hub",
    )
    .add_local_python_source("scripts")
)

hf_cache = modal.Volume.from_name("mechinterp-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/cache": hf_cache},
    secrets=[],
)
def run_remote(
    prompt_family: str = "synthetic",
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
    candidate_k: int = 10,
) -> dict:
    import os

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface")

    from scripts.benchmark import run_benchmark

    result, details = run_benchmark(
        model_name="santacoder",
        device="cuda",
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
        candidate_k=candidate_k,
    )

    payload = {
        "result": details["result"],
        "top_heads": details["top_heads"],
        "examples": details["examples"],
        "modal_config": {
            "gpu": "A10G",
            "prompt_family": prompt_family,
            "n_prompts": n_prompts,
            "seq_len": seq_len,
            "seed": seed,
            "candidate_k": candidate_k,
        },
    }
    return payload


@app.local_entrypoint()
def main(
    prompt_family: str = "synthetic",
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
    candidate_k: int = 10,
    save_path: str = "outputs/induction_heads/modal_latest.json",
): 
    payload = run_remote.remote(
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
        candidate_k=candidate_k,
    )
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    result = payload["result"]
    print("Modal induction benchmark complete")
    print(f"  prompt family:  {result['prompt_family']}")
    print(f"  detection head: L{result['detection_top_head'][0]}H{result['detection_top_head'][1]}")
    print(f"  causal head:    L{result['top_head'][0]}H{result['top_head'][1]}")
    print(f"  detect score:   {result['detection_top_head_score']:.4f}")
    print(f"  causal score:   {result['causal_selective_drop']:.4f}")
    print(f"  behavior gain:  {result['behavioral_gain']:.4f}")
    print(f"  composite:      {result['composite_score']:.4f}")
    print(f"  saved:          {out}")
