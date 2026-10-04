#!/usr/bin/env python3
"""Run the matched delimiter experiment on a cached Modal A10G."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("mechinterp-santacoder-brackets")

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
)
def run_remote(
    mode: str = "behavior",
    split: str = "discovery",
    causal_limit: int | None = None,
    layers: list[int] | None = None,
    causal_kind: str = "both",
) -> dict:
    import os

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    from scripts.bracket_experiment import run

    return run(
        device="cuda",
        mode=mode,
        split=split,
        causal_limit=causal_limit,
        layers=layers,
        causal_kind=causal_kind,
    )


@app.local_entrypoint()
def main(
    mode: str = "behavior",
    split: str = "discovery",
    causal_limit: int | None = None,
    layers: str = "",
    causal_kind: str = "both",
    save_path: str = "outputs/bracket_matching/matched_discovery.json",
):
    layer_list = [int(value) for value in layers.split(",") if value.strip()] or None
    payload = run_remote.remote(
        mode=mode,
        split=split,
        causal_limit=causal_limit,
        layers=layer_list,
        causal_kind=causal_kind,
    )
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["behavioral_summary"], indent=2))
    if payload.get("head_patching"):
        print("Top bidirectional patching heads:")
        for row in payload["head_patching"][:10]:
            print(f"  L{row['layer']}H{row['head']}: {row['bidirectional_score']:.4f}")
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}")
    print(f"saved={path}")
