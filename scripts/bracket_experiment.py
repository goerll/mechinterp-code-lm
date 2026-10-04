#!/usr/bin/env python3
"""Behavioral and causal evaluation for matched delimiter counterfactuals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from transformer_lens import utils as tl_utils

from scripts.benchmark import load_model
from scripts.bracket_dataset import BracketPair, dataset_fingerprint, generate_bracket_pairs
from scripts.patch_santacoder import REVISION

torch.set_grad_enabled(False)

CLOSERS = (")", "]", "}")


def _target_ids(model) -> dict[str, int]:
    result = {}
    for closer in CLOSERS:
        ids = model.to_tokens(closer, prepend_bos=False)[0].tolist()
        if len(ids) != 1:
            raise RuntimeError(f"Closer {closer!r} is not one token: {ids}")
        result[closer] = ids[0]
    return result


def _tokens(model, prompt: str) -> torch.Tensor:
    return model.to_tokens(prompt, prepend_bos=True)


def validate_pair_alignment(model, pair: BracketPair) -> dict:
    clean = _tokens(model, pair.clean_prompt)
    corrupt = _tokens(model, pair.corrupt_prompt)
    negative = _tokens(model, pair.negative_control_prompt)
    aligned = clean.shape == corrupt.shape
    differing_positions = []
    if aligned:
        differing_positions = torch.where(clean[0] != corrupt[0])[0].tolist()
    return {
        "aligned": bool(aligned),
        "clean_length": int(clean.shape[1]),
        "corrupt_length": int(corrupt.shape[1]),
        "negative_length": int(negative.shape[1]),
        "differing_token_positions": differing_positions,
    }


def _candidate_logits(model, tokens: torch.Tensor, target_ids: dict[str, int]) -> dict[str, float]:
    logits = model(tokens)[0, -1]
    return {closer: float(logits[token_id].item()) for closer, token_id in target_ids.items()}


def _margin(logits: dict[str, float], positive: str, negative: str) -> float:
    return logits[positive] - logits[negative]


def behavioral_rows(model, pairs: list[BracketPair]) -> list[dict]:
    target_ids = _target_ids(model)
    rows = []
    for pair in pairs:
        alignment = validate_pair_alignment(model, pair)
        clean_logits = _candidate_logits(model, _tokens(model, pair.clean_prompt), target_ids)
        corrupt_logits = _candidate_logits(model, _tokens(model, pair.corrupt_prompt), target_ids)
        negative_logits = _candidate_logits(
            model, _tokens(model, pair.negative_control_prompt), target_ids
        )
        clean_margin = _margin(clean_logits, pair.clean_target, pair.corrupt_target)
        corrupt_clean_margin = _margin(
            corrupt_logits, pair.clean_target, pair.corrupt_target
        )
        negative_margin = _margin(
            negative_logits, pair.clean_target, pair.corrupt_target
        )
        rows.append(
            {
                **asdict(pair),
                **alignment,
                "clean_logits": clean_logits,
                "corrupt_logits": corrupt_logits,
                "negative_control_logits": negative_logits,
                "clean_margin": clean_margin,
                "corrupt_clean_margin": corrupt_clean_margin,
                "negative_control_margin": negative_margin,
                "relevant_opener_effect": clean_margin - corrupt_clean_margin,
                "negative_control_effect": clean_margin - negative_margin,
                "clean_prediction": max(clean_logits, key=clean_logits.get),
                "corrupt_prediction": max(corrupt_logits, key=corrupt_logits.get),
            }
        )
    return rows


def _bootstrap_lcb(values: list[float], *, seed: int = 0, n_boot: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = [float(rng.choice(array, len(array), replace=True).mean()) for _ in range(n_boot)]
    return float(np.quantile(means, 0.05))


def summarize_behavior(rows: list[dict]) -> dict:
    clean_correct = [row["clean_prediction"] == row["clean_target"] for row in rows]
    corrupt_correct = [row["corrupt_prediction"] == row["corrupt_target"] for row in rows]
    pair_correct = [clean and corrupt for clean, corrupt in zip(clean_correct, corrupt_correct)]
    relevant = [row["relevant_opener_effect"] for row in rows]
    negative = [abs(row["negative_control_effect"]) for row in rows]
    relevant_abs = [abs(value) for value in relevant]

    by_condition: dict[str, list[bool]] = defaultdict(list)
    for row, correct in zip(rows, pair_correct):
        key = f"{row['clean_target']}|d{row['depth']}|{row['distance']}"
        by_condition[key].append(correct)

    return {
        "n_pairs": len(rows),
        "aligned_fraction": sum(row["aligned"] for row in rows) / len(rows),
        "single_difference_fraction": sum(
            len(row["differing_token_positions"]) == 1 for row in rows
        )
        / len(rows),
        "clean_accuracy": float(np.mean(clean_correct)),
        "corrupt_accuracy": float(np.mean(corrupt_correct)),
        "paired_accuracy": float(np.mean(pair_correct)),
        "paired_accuracy_bootstrap_95_lcb": _bootstrap_lcb(
            [float(value) for value in pair_correct]
        ),
        "relevant_opener_sensitivity": float(np.mean([value > 0 for value in relevant])),
        "mean_relevant_opener_effect": float(np.mean(relevant)),
        "mean_absolute_negative_control_effect": float(np.mean(negative)),
        "negative_to_relevant_effect_ratio": float(
            np.mean(negative) / max(np.mean(relevant_abs), 1e-12)
        ),
        "paired_accuracy_by_condition": {
            key: float(np.mean(values)) for key, values in sorted(by_condition.items())
        },
    }


def select_balanced(rows: list[dict], limit: int | None) -> list[dict]:
    rows = [
        row
        for row in rows
        if row["clean_prediction"] == row["clean_target"]
        and row["corrupt_prediction"] == row["corrupt_target"]
    ]
    if limit is None or limit >= len(rows):
        return rows
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["clean_target"], row["depth"], row["distance"])].append(row)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


def _row_to_pair(row: dict) -> BracketPair:
    fields = BracketPair.__dataclass_fields__
    return BracketPair(**{key: row[key] for key in fields})


def head_patch_discovery(model, rows: list[dict], layers: list[int] | None = None) -> list[dict]:
    """Vectorized clean->corrupt patching of every query head in selected layers."""
    target_ids = _target_ids(model)
    layers = layers if layers is not None else list(range(model.cfg.n_layers))
    totals = {(layer, head): [] for layer in layers for head in range(model.cfg.n_heads)}
    reverse_totals = {(layer, head): [] for layer in layers for head in range(model.cfg.n_heads)}

    for row in rows:
        pair = _row_to_pair(row)
        clean_tokens = _tokens(model, pair.clean_prompt)
        corrupt_tokens = _tokens(model, pair.corrupt_prompt)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue
        clean_margin = row["clean_margin"]
        corrupt_margin = row["corrupt_clean_margin"]
        denominator = clean_margin - corrupt_margin
        if abs(denominator) < 1e-6:
            continue

        for layer in layers:
            act_name = tl_utils.get_act_name("z", layer)
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=[act_name])
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=[act_name])
            clean_z = clean_cache[act_name]
            corrupt_z = corrupt_cache[act_name]
            n_heads = model.cfg.n_heads

            def patch_clean_into_corrupt(z, hook):
                del hook
                z = z.clone()
                indices = torch.arange(n_heads, device=z.device)
                z[indices, :, indices, :] = clean_z[0, :, indices, :].permute(1, 0, 2)
                return z

            patched_logits = model.run_with_hooks(
                corrupt_tokens.repeat(n_heads, 1),
                fwd_hooks=[(act_name, patch_clean_into_corrupt)],
            )[:, -1]

            def patch_corrupt_into_clean(z, hook):
                del hook
                z = z.clone()
                indices = torch.arange(n_heads, device=z.device)
                z[indices, :, indices, :] = corrupt_z[0, :, indices, :].permute(1, 0, 2)
                return z

            reverse_logits = model.run_with_hooks(
                clean_tokens.repeat(n_heads, 1),
                fwd_hooks=[(act_name, patch_corrupt_into_clean)],
            )[:, -1]

            clean_id = target_ids[pair.clean_target]
            corrupt_id = target_ids[pair.corrupt_target]
            patched_margins = patched_logits[:, clean_id] - patched_logits[:, corrupt_id]
            reverse_margins = reverse_logits[:, clean_id] - reverse_logits[:, corrupt_id]
            for head in range(n_heads):
                restoration = float((patched_margins[head].item() - corrupt_margin) / denominator)
                damage = float((clean_margin - reverse_margins[head].item()) / denominator)
                totals[(layer, head)].append(restoration)
                reverse_totals[(layer, head)].append(damage)

    output = []
    for key, values in totals.items():
        if not values:
            continue
        reverse = reverse_totals[key]
        output.append(
            {
                "layer": key[0],
                "head": key[1],
                "clean_to_corrupt_restoration_mean": float(np.mean(values)),
                "clean_to_corrupt_restoration_std": float(np.std(values, ddof=1)),
                "corrupt_to_clean_damage_mean": float(np.mean(reverse)),
                "corrupt_to_clean_damage_std": float(np.std(reverse, ddof=1)),
                "bidirectional_score": float((np.mean(values) + np.mean(reverse)) / 2),
                "n_pairs": len(values),
            }
        )
    return sorted(output, key=lambda item: item["bidirectional_score"], reverse=True)


def component_patch_discovery(model, rows: list[dict], layers: list[int] | None = None) -> list[dict]:
    """Patch attention, MLP, and residual outputs layer-by-layer in both directions."""
    target_ids = _target_ids(model)
    layers = layers if layers is not None else list(range(model.cfg.n_layers))
    components = ("attn_out", "mlp_out", "resid_post")
    forward_values = {(layer, component): [] for layer in layers for component in components}
    reverse_values = {(layer, component): [] for layer in layers for component in components}
    for row in rows:
        pair = _row_to_pair(row)
        clean_tokens = _tokens(model, pair.clean_prompt)
        corrupt_tokens = _tokens(model, pair.corrupt_prompt)
        if clean_tokens.shape != corrupt_tokens.shape:
            continue
        denominator = row["clean_margin"] - row["corrupt_clean_margin"]
        if abs(denominator) < 1e-6:
            continue
        for layer in layers:
            names = [tl_utils.get_act_name(component, layer) for component in components]
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=names)
            _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=names)
            clean_hooks = []
            corrupt_hooks = []
            for batch_index, name in enumerate(names):
                def clean_hook(value, hook, batch_index=batch_index, name=name):
                    del hook
                    value = value.clone()
                    value[batch_index, -1] = clean_cache[name][0, -1]
                    return value

                def corrupt_hook(value, hook, batch_index=batch_index, name=name):
                    del hook
                    value = value.clone()
                    value[batch_index, -1] = corrupt_cache[name][0, -1]
                    return value

                clean_hooks.append((name, clean_hook))
                corrupt_hooks.append((name, corrupt_hook))
            forward_logits = model.run_with_hooks(
                corrupt_tokens.repeat(len(components), 1), fwd_hooks=clean_hooks
            )[:, -1]
            reverse_logits = model.run_with_hooks(
                clean_tokens.repeat(len(components), 1), fwd_hooks=corrupt_hooks
            )[:, -1]
            clean_id = target_ids[pair.clean_target]
            corrupt_id = target_ids[pair.corrupt_target]
            forward_margins = forward_logits[:, clean_id] - forward_logits[:, corrupt_id]
            reverse_margins = reverse_logits[:, clean_id] - reverse_logits[:, corrupt_id]
            for index, component in enumerate(components):
                forward_values[(layer, component)].append(
                    float(
                        (forward_margins[index].item() - row["corrupt_clean_margin"])
                        / denominator
                    )
                )
                reverse_values[(layer, component)].append(
                    float((row["clean_margin"] - reverse_margins[index].item()) / denominator)
                )
    result = []
    for key, values in forward_values.items():
        if not values:
            continue
        reverse = reverse_values[key]
        result.append(
            {
                "layer": key[0],
                "component": key[1],
                "clean_to_corrupt_restoration_mean": float(np.mean(values)),
                "corrupt_to_clean_damage_mean": float(np.mean(reverse)),
                "bidirectional_score": float((np.mean(values) + np.mean(reverse)) / 2),
                "n_pairs": len(values),
            }
        )
    return sorted(result, key=lambda item: item["bidirectional_score"], reverse=True)


def runtime_metadata(model) -> dict:
    return {
        "model": "bigcode/santacoder",
        "model_revision": REVISION,
        "transformer_lens_model": "santacoder",
        "device": str(model.cfg.device),
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def run(
    *,
    device: str,
    mode: str,
    split: str,
    causal_limit: int | None,
    layers: list[int] | None,
    causal_kind: str = "both",
) -> dict:
    started = time.time()
    model = load_model(device=device)
    pairs = [pair for pair in generate_bracket_pairs() if pair.split == split]
    rows = behavioral_rows(model, pairs)
    payload = {
        "schema_version": 1,
        "metadata": runtime_metadata(model),
        "dataset_fingerprint_sha256": dataset_fingerprint(generate_bracket_pairs()),
        "split": split,
        "behavioral_summary": summarize_behavior(rows),
        "behavioral_rows": rows,
    }
    if mode == "causal":
        causal_rows = select_balanced(rows, causal_limit)
        payload["causal_sample_labels"] = [row["label"] for row in causal_rows]
        if causal_kind in {"both", "components"}:
            payload["component_patching"] = component_patch_discovery(
                model, causal_rows, layers=layers
            )
        if causal_kind in {"both", "heads"}:
            payload["head_patching"] = head_patch_discovery(model, causal_rows, layers=layers)
    payload["elapsed_seconds"] = time.time() - started
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("behavior", "causal"), default="behavior")
    parser.add_argument("--split", choices=("discovery", "confirmation"), default="discovery")
    parser.add_argument("--causal-limit", type=int)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--causal-kind", choices=("both", "components", "heads"), default="both")
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        device=args.device,
        mode=args.mode,
        split=args.split,
        causal_limit=args.causal_limit,
        layers=args.layers,
        causal_kind=args.causal_kind,
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["behavioral_summary"], indent=2))
    if payload.get("head_patching"):
        print("Top bidirectional patching heads:")
        for row in payload["head_patching"][:10]:
            print(
                f"  L{row['layer']}H{row['head']}: "
                f"score={row['bidirectional_score']:.4f}"
            )
    print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
