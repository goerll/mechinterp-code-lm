#!/usr/bin/env python3
"""Evaluate single-head and multi-head ablations for recurring induction candidates on Modal."""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("mechinterp-santacoder-head-sets")

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


@app.function(image=image, gpu="A10G", timeout=60 * 30, volumes={"/cache": hf_cache})
def run_remote(
    prompt_family: str = "synthetic",
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
) -> dict:
    import os

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface")

    from scripts.benchmark import (
        build_prompt_dataset,
        evaluate_head_set,
        load_model,
        run_benchmark,
    )

    model = load_model(model_name="santacoder", device="cuda")
    dataset = build_prompt_dataset(
        model,
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
    )
    baseline, _ = run_benchmark(
        model_name="santacoder",
        device="cuda",
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
    )

    head_sets = [
        [(1, 11)],
        [(8, 6)],
        [(4, 13)],
        [(1, 11), (8, 6)],
        [(1, 11), (4, 13)],
        [(8, 6), (4, 13)],
        [(1, 11), (8, 6), (4, 13)],
    ]

    rows = []
    for head_set in head_sets:
        row = evaluate_head_set(model, dataset, head_set)
        rows.append(row)

    rows.sort(key=lambda row: (row["causal_selective_drop"], row["repeated_ablation_drop"]), reverse=True)
    return {
        "baseline": baseline.__dict__,
        "rows": rows,
        "config": {
            "prompt_family": prompt_family,
            "n_prompts": n_prompts,
            "seq_len": seq_len,
            "seed": seed,
        },
    }


@app.local_entrypoint()
def main(
    prompt_family: str = "synthetic",
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
    save_path: str = "outputs/induction_heads/head_sets.json",
):
    payload = run_remote.remote(
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
    )
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print("Modal head-set analysis complete")
    print(f"  prompt family: {payload['config']['prompt_family']}")
    print(f"  baseline detection head: L{payload['baseline']['detection_top_head'][0]}H{payload['baseline']['detection_top_head'][1]}")
    print(f"  baseline causal head:    L{payload['baseline']['top_head'][0]}H{payload['baseline']['top_head'][1]}")
    print("  top ablation sets:")
    for row in payload["rows"][:3]:
        print(
            f"    {row['head_set']}: selective={row['causal_selective_drop']:.4f}, "
            f"repeated_drop={row['repeated_ablation_drop']:.4f}"
        )
    print(f"  saved: {out}")
