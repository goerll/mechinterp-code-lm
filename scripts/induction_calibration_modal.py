#!/usr/bin/env python3
"""Held-out induction calibration with intervention-baseline sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
import time

import modal


app = modal.App("mechinterp-santacoder-induction-calibration")
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


@app.function(image=image, gpu="A10G", timeout=60 * 30, volumes={"/cache": hf_cache})
def run_remote() -> dict:
    import os

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    import numpy as np

    from scripts.benchmark import (
        build_prompt_dataset,
        intervened_target_logprob,
        load_model,
        mean_target_logprob,
        score_heads,
    )
    from scripts.patch_santacoder import REVISION

    started = time.time()
    model = load_model(device="cuda")
    # Candidates were preregistered from the first-semester exploratory sweep.
    candidates = [(8, 5), (14, 11), (8, 1), (9, 15), (12, 4), (5, 4)]
    conditions = [
        {"family": "synthetic", "n": 8, "length": 8, "seed": seed}
        for seed in (100, 101, 102)
    ] + [
        {"family": "synthetic", "n": 8, "length": length, "seed": 103}
        for length in (6, 12)
    ] + [
        {"family": "code", "n": 4, "length": 8, "seed": 100}
    ]
    rows = []
    for condition in conditions:
        dataset = build_prompt_dataset(
            model,
            prompt_family=condition["family"],
            n_prompts=condition["n"],
            seq_len=condition["length"],
            seed=condition["seed"],
        )
        repeated = mean_target_logprob(model, dataset, "repeated")
        control = mean_target_logprob(model, dataset, "control")
        detection = score_heads(model, dataset)
        top_flat = int(detection.flatten().argmax().item())
        row = {
            **condition,
            "behavioral_gain": repeated - control,
            "detection_top_head": [top_flat // model.cfg.n_heads, top_flat % model.cfg.n_heads],
            "detection_top_score": float(detection.flatten()[top_flat].item()),
            "candidates": [],
        }
        for head in candidates:
            candidate = {"head": list(head), "detection_score": float(detection[head].item())}
            for baseline in ("zero", "mean", "resample"):
                repeated_ablated = intervened_target_logprob(
                    model, dataset, "repeated", head=head, baseline=baseline
                )
                control_ablated = intervened_target_logprob(
                    model, dataset, "control", head=head, baseline=baseline
                )
                candidate[baseline] = {
                    "repeated_drop": repeated - repeated_ablated,
                    "control_drop": control - control_ablated,
                    "selective_drop": (repeated - repeated_ablated)
                    - (control - control_ablated),
                }
            row["candidates"].append(candidate)
        rows.append(row)

    aggregate = []
    for head in candidates:
        relevant_rows = [row for row in rows if row["family"] == "synthetic"]
        candidate_rows = [
            next(candidate for candidate in row["candidates"] if candidate["head"] == list(head))
            for row in relevant_rows
        ]
        aggregate.append(
            {
                "head": list(head),
                "mean_detection_score": float(np.mean([row["detection_score"] for row in candidate_rows])),
                **{
                    f"{baseline}_selective_drop_mean": float(
                        np.mean([row[baseline]["selective_drop"] for row in candidate_rows])
                    )
                    for baseline in ("zero", "mean", "resample")
                },
            }
        )
    aggregate.sort(key=lambda row: row["resample_selective_drop_mean"], reverse=True)
    return {
        "schema_version": 1,
        "model_revision": REVISION,
        "candidate_selection": "preregistered from first-semester exploratory sweep",
        "conditions": rows,
        "synthetic_aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }


@app.local_entrypoint()
def main(save_path: str = "outputs/induction_heads/calibration_confirmation.json"):
    payload = run_remote.remote()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print("Induction confirmation complete")
    for row in payload["synthetic_aggregate"]:
        print(
            f"  L{row['head'][0]}H{row['head'][1]}: "
            f"detect={row['mean_detection_score']:.3f}, "
            f"zero={row['zero_selective_drop_mean']:.4f}, "
            f"mean={row['mean_selective_drop_mean']:.4f}, "
            f"resample={row['resample_selective_drop_mean']:.4f}"
        )
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}")
    print(f"saved={path}")
