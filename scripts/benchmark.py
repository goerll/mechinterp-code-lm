#!/usr/bin/env python3
"""Reproduce induction-head style evidence on SantaCoder.

This script is designed to be both:
1. a benchmark oracle for an optimization loop; and
2. an importable utility module for the project notebooks.

Method summary
--------------
We use a small synthetic repeated-token task inspired by prior induction-head work.
For a token sequence s = [t1, ..., tL], we evaluate three quantities:

- Detection: does a head attend from the second copy of ti to the first copy of ti?
- Behavioral gain: does repeating the prefix increase log-probability of the held-out target tL?
- Causal effect: does ablating the strongest detected head reduce that repetition benefit?

The composite score is logged for loop bookkeeping, but all components are also saved
separately so the scalar does not hide the underlying evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.patch_santacoder import REVISION, ensure_santacoder_transformers_compat

torch.set_grad_enabled(False)


SAFE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{1,11}$")
DEFAULT_OUTPUT = ROOT / "outputs" / "induction_heads" / "latest.json"


@dataclass
class PromptExample:
    repeated: list[int]
    control: list[int]
    target: int
    sequence: list[int]
    distractor: list[int]
    detection_sequence: list[int]
    align_pairs: list[tuple[int, int]]
    label: str
    repeated_text: str | None = None
    control_text: str | None = None
    detection_text: str | None = None
    target_text: str | None = None


@dataclass
class BenchmarkResult:
    model_name: str
    prompt_family: str
    n_prompts: int
    seq_len: int
    seed: int
    detection_top_head: list[int]
    detection_top_head_score: float
    top_head: list[int]
    top_head_score: float
    repeated_target_logprob: float
    control_target_logprob: float
    behavioral_gain: float
    ablated_repeated_target_logprob: float
    ablated_control_target_logprob: float
    repeated_ablation_drop: float
    control_ablation_drop: float
    causal_selective_drop: float
    composite_score: float
    elapsed_seconds: float


def load_model(model_name: str = "santacoder", device: str = "cpu") -> HookedTransformer:
    ensure_santacoder_transformers_compat()
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        revision=REVISION if model_name == "santacoder" else None,
    )
    model.eval()
    return model


def safe_token_pool(model: HookedTransformer, pool_size: int = 256) -> list[int]:
    seen: set[str] = set()
    token_ids: list[int] = []
    for token_id in range(model.cfg.d_vocab):
        try:
            token = model.to_single_str_token(token_id)
        except Exception:
            continue
        normalized = token.strip()
        if not SAFE_TOKEN_RE.fullmatch(normalized):
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        token_ids.append(token_id)
        if len(token_ids) >= pool_size:
            break
    if len(token_ids) < 64:
        raise RuntimeError(
            f"Only found {len(token_ids)} safe tokens; expected at least 64 for prompt generation"
        )
    return token_ids


def build_synthetic_prompt_dataset(
    model: HookedTransformer,
    *,
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
) -> list[PromptExample]:
    pool = safe_token_pool(model, pool_size=max(256, n_prompts * seq_len * 4))
    g = torch.Generator().manual_seed(seed)
    dataset: list[PromptExample] = []

    for prompt_idx in range(n_prompts):
        perm = torch.randperm(len(pool), generator=g).tolist()
        primary = [pool[i] for i in perm[:seq_len]]
        distractor = [pool[i] for i in perm[seq_len : 2 * seq_len]]
        repeated = primary + primary
        control = distractor + primary
        dataset.append(
            PromptExample(
                repeated=repeated[:-1],
                control=control[:-1],
                target=primary[-1],
                sequence=primary,
                distractor=distractor,
                detection_sequence=primary + primary,
                align_pairs=[(seq_len + i, i + 1) for i in range(seq_len - 1)],
                label=f"synthetic_{prompt_idx}",
            )
        )
    return dataset


def build_code_prompt_dataset(
    model: HookedTransformer,
    *,
    n_prompts: int = 6,
    seq_len: int = 0,
    seed: int = 0,
) -> list[PromptExample]:
    del seq_len, seed
    identifiers = [
        "result",
        "value",
        "items",
        "message",
        "buffer",
        "config",
        "token",
        "status",
    ]
    scaffolds = [
        {
            "label": "parse_print",
            "assign_rhs": "parse(data)",
            "use_lines": ["print({name})", "store({name})"],
            "query_prefix": "print(",
        },
        {
            "label": "load_process",
            "assign_rhs": "load_items()",
            "use_lines": ["process({name})", "cache({name})"],
            "query_prefix": "process(",
        },
        {
            "label": "compute_emit",
            "assign_rhs": "compute(x)",
            "use_lines": ["emit({name})", "save({name})"],
            "query_prefix": "emit(",
        },
        {
            "label": "fetch_log",
            "assign_rhs": "fetch_value()",
            "use_lines": ["log({name})", "reuse({name})"],
            "query_prefix": "log(",
        },
    ]

    single_token_identifiers = []
    for identifier in identifiers:
        token_ids = model.to_tokens(identifier, prepend_bos=False)[0].tolist()
        if len(token_ids) == 1:
            single_token_identifiers.append((identifier, token_ids[0]))

    if len(single_token_identifiers) < 4:
        raise RuntimeError(
            "Expected at least four code identifiers that tokenize to a single SantaCoder token"
        )

    templates = []
    ids_per_scaffold = max(2, math.ceil(n_prompts / len(scaffolds)))
    needed_identifiers = ids_per_scaffold * len(scaffolds)
    if len(single_token_identifiers) < needed_identifiers:
        raise RuntimeError(
            "Need more single-token identifiers to build matched code prompt controls"
        )

    for scaffold_idx, scaffold in enumerate(scaffolds):
        start = scaffold_idx * ids_per_scaffold
        stop = start + ids_per_scaffold
        for identifier, target_token in single_token_identifiers[start:stop]:
            block_lines = [f"{identifier} = {scaffold['assign_rhs']}"]
            block_lines.extend(line.format(name=identifier) for line in scaffold["use_lines"])
            block = "\n".join(block_lines) + "\n"
            templates.append(
                {
                    "label": f"{scaffold['label']}_{identifier}",
                    "block": block,
                    "query_prefix": scaffold["query_prefix"],
                    "target_text": identifier,
                    "target_token": target_token,
                }
            )

    templates = templates[:n_prompts]
    separator = "\n"
    dataset: list[PromptExample] = []

    tokenized_blocks = [model.to_tokens(item["block"], prepend_bos=False)[0].tolist() for item in templates]
    tokenized_separator = model.to_tokens(separator, prepend_bos=False)[0].tolist()

    for idx, item in enumerate(templates):
        group_start = (idx // ids_per_scaffold) * ids_per_scaffold
        group_stop = min(group_start + ids_per_scaffold, len(templates))
        group_size = group_stop - group_start
        if group_size <= 1:
            distractor_idx = idx
        else:
            group_offset = (idx - group_start + 1) % group_size
            distractor_idx = group_start + group_offset
        distractor = templates[distractor_idx]
        block_tokens = tokenized_blocks[idx]
        distractor_tokens = tokenized_blocks[distractor_idx]
        query_tokens = model.to_tokens(item["query_prefix"], prepend_bos=False)[0].tolist()

        detection_tokens = block_tokens + tokenized_separator + block_tokens
        second_offset = len(block_tokens) + len(tokenized_separator)
        align_pairs = [(second_offset + i, i + 1) for i in range(len(block_tokens) - 1)]

        dataset.append(
            PromptExample(
                repeated=block_tokens + tokenized_separator + query_tokens,
                control=distractor_tokens + tokenized_separator + query_tokens,
                target=item["target_token"],
                sequence=block_tokens,
                distractor=distractor_tokens,
                detection_sequence=detection_tokens,
                align_pairs=align_pairs,
                label=item["label"],
                repeated_text=item["block"] + separator + item["query_prefix"],
                control_text=distractor["block"] + separator + item["query_prefix"],
                detection_text=item["block"] + separator + item["block"],
                target_text=item["target_text"],
            )
        )
    return dataset


def build_prompt_dataset(
    model: HookedTransformer,
    *,
    prompt_family: str = "synthetic",
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
) -> list[PromptExample]:
    if prompt_family == "synthetic":
        return build_synthetic_prompt_dataset(model, n_prompts=n_prompts, seq_len=seq_len, seed=seed)
    if prompt_family == "code":
        return build_code_prompt_dataset(model, n_prompts=n_prompts, seq_len=seq_len, seed=seed)
    raise ValueError(f"Unknown prompt_family: {prompt_family}")


def score_heads(model: HookedTransformer, dataset: Iterable[PromptExample]) -> torch.Tensor:
    scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    count = 0
    for example in dataset:
        det_tokens = torch.tensor([example.detection_sequence], dtype=torch.long, device=model.cfg.device)
        _, cache = model.run_with_cache(det_tokens)
        pair_index = torch.tensor(example.align_pairs, dtype=torch.long, device=model.cfg.device)
        q_index = pair_index[:, 0]
        k_index = pair_index[:, 1]
        for layer in range(model.cfg.n_layers):
            pattern = cache["pattern", layer][0]
            scores[layer] += pattern[:, q_index, k_index].mean(dim=-1)
        count += 1
    if count == 0:
        raise RuntimeError("Dataset for head scoring is empty")
    return scores / count


def mean_target_logprob(
    model: HookedTransformer,
    dataset: Iterable[PromptExample],
    kind: str,
    *,
    ablate_head: tuple[int, int] | None = None,
    ablate_heads: list[tuple[int, int]] | None = None,
) -> float:
    head_groups: dict[int, list[int]] = {}
    requested_heads = []
    if ablate_heads is not None:
        requested_heads.extend(ablate_heads)
    if ablate_head is not None:
        requested_heads.append(ablate_head)
    for layer, head in requested_heads:
        head_groups.setdefault(layer, []).append(head)

    fwd_hooks = []
    for layer, heads in head_groups.items():
        unique_heads = sorted(set(heads))

        def zero_heads(z: torch.Tensor, hook, heads=tuple(unique_heads)) -> torch.Tensor:
            del hook
            z = z.clone()
            for head in heads:
                z[:, :, head, :] = 0.0
            return z

        fwd_hooks.append((tl_utils.get_act_name("z", layer), zero_heads))

    gathered_values: list[float] = []
    for example in dataset:
        tokens = torch.tensor([[*getattr(example, kind)]], dtype=torch.long, device=model.cfg.device)
        target = torch.tensor([[example.target]], dtype=torch.long, device=model.cfg.device)
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks) if fwd_hooks else model(tokens)
        logprobs = logits[:, -1].log_softmax(dim=-1)
        gathered = logprobs.gather(-1, target).squeeze(-1)
        gathered_values.append(float(gathered.item()))
    if not gathered_values:
        raise RuntimeError("Dataset for log-prob scoring is empty")
    return sum(gathered_values) / len(gathered_values)


def intervened_target_logprob(
    model: HookedTransformer,
    dataset: Iterable[PromptExample],
    kind: str,
    *,
    head: tuple[int, int],
    baseline: str,
) -> float:
    """Score a head under zero, position-mean, or matched resample ablation.

    ``resample`` uses the matched opposite-family prompt from the same example,
    preserving sequence length. ``mean`` broadcasts that donor activation's
    position mean. These are sensitivity analyses, not claims that either donor
    is perfectly in-distribution.
    """
    if baseline not in {"zero", "mean", "resample"}:
        raise ValueError(f"Unknown intervention baseline: {baseline}")
    layer, head_index = head
    act_name = tl_utils.get_act_name("z", layer)
    values = []
    for example in dataset:
        token_ids = getattr(example, kind)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=model.cfg.device)
        donor_kind = "control" if kind == "repeated" else "repeated"
        donor_tokens = torch.tensor(
            [getattr(example, donor_kind)], dtype=torch.long, device=model.cfg.device
        )
        if tokens.shape != donor_tokens.shape:
            raise RuntimeError("Matched intervention prompts must have equal token length")
        donor = None
        if baseline != "zero":
            _, donor_cache = model.run_with_cache(donor_tokens, names_filter=[act_name])
            donor = donor_cache[act_name][:, :, head_index, :]

        def intervene(z: torch.Tensor, hook) -> torch.Tensor:
            del hook
            z = z.clone()
            if baseline == "zero":
                z[:, :, head_index, :] = 0.0
            elif baseline == "resample":
                z[:, :, head_index, :] = donor
            else:
                z[:, :, head_index, :] = donor.mean(dim=1, keepdim=True)
            return z

        logits = model.run_with_hooks(tokens, fwd_hooks=[(act_name, intervene)])
        logprobs = logits[:, -1].log_softmax(dim=-1)
        values.append(float(logprobs[0, example.target].item()))
    if not values:
        raise RuntimeError("Dataset for intervention scoring is empty")
    return sum(values) / len(values)


def decode_sequence(model: HookedTransformer, token_ids: list[int]) -> list[str]:
    return [model.to_single_str_token(token_id) for token_id in token_ids]


def evaluate_head_set(
    model: HookedTransformer,
    dataset: Iterable[PromptExample],
    head_set: list[tuple[int, int]],
) -> dict:
    repeated_clean = mean_target_logprob(model, dataset, "repeated")
    control_clean = mean_target_logprob(model, dataset, "control")
    repeated_ablated = mean_target_logprob(model, dataset, "repeated", ablate_heads=head_set)
    control_ablated = mean_target_logprob(model, dataset, "control", ablate_heads=head_set)
    repeated_drop = repeated_clean - repeated_ablated
    control_drop = control_clean - control_ablated
    return {
        "head_set": [[int(layer), int(head)] for layer, head in head_set],
        "repeated_target_logprob": float(repeated_clean),
        "control_target_logprob": float(control_clean),
        "behavioral_gain": float(repeated_clean - control_clean),
        "ablated_repeated_target_logprob": float(repeated_ablated),
        "ablated_control_target_logprob": float(control_ablated),
        "repeated_ablation_drop": float(repeated_drop),
        "control_ablation_drop": float(control_drop),
        "causal_selective_drop": float(repeated_drop - control_drop),
    }


def run_benchmark(
    *,
    model_name: str = "santacoder",
    device: str = "cpu",
    prompt_family: str = "synthetic",
    n_prompts: int = 6,
    seq_len: int = 8,
    seed: int = 0,
    candidate_k: int = 10,
) -> tuple[BenchmarkResult, dict]:
    started = time.time()
    model = load_model(model_name=model_name, device=device)
    dataset = build_prompt_dataset(
        model,
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
    )

    head_scores = score_heads(model, dataset)
    flat_index = head_scores.view(-1).argmax().item()
    detection_top_layer = flat_index // model.cfg.n_heads
    detection_top_head = flat_index % model.cfg.n_heads
    detection_top_head_score = head_scores[detection_top_layer, detection_top_head].item()

    repeated_target_logprob = mean_target_logprob(model, dataset, "repeated")
    control_target_logprob = mean_target_logprob(model, dataset, "control")
    behavioral_gain = repeated_target_logprob - control_target_logprob

    candidate_rows = []
    ranked = torch.topk(head_scores.view(-1), k=min(candidate_k, head_scores.numel()))
    for value, idx in zip(ranked.values.tolist(), ranked.indices.tolist()):
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        ablated_repeated = mean_target_logprob(model, dataset, "repeated", ablate_head=(layer, head))
        ablated_control = mean_target_logprob(model, dataset, "control", ablate_head=(layer, head))
        repeated_drop = repeated_target_logprob - ablated_repeated
        control_drop = control_target_logprob - ablated_control
        selective = repeated_drop - control_drop
        candidate_rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "score": float(value),
                "ablated_repeated_target_logprob": float(ablated_repeated),
                "ablated_control_target_logprob": float(ablated_control),
                "repeated_ablation_drop": float(repeated_drop),
                "control_ablation_drop": float(control_drop),
                "causal_selective_drop": float(selective),
            }
        )

    positive_repeated = [row for row in candidate_rows if row["repeated_ablation_drop"] > 0]
    selection_pool = positive_repeated if positive_repeated else candidate_rows
    best = max(selection_pool, key=lambda row: row["causal_selective_drop"])

    top_layer = best["layer"]
    top_head = best["head"]
    top_head_score = best["score"]
    ablated_repeated_target_logprob = best["ablated_repeated_target_logprob"]
    ablated_control_target_logprob = best["ablated_control_target_logprob"]
    repeated_ablation_drop = best["repeated_ablation_drop"]
    control_ablation_drop = best["control_ablation_drop"]
    causal_selective_drop = best["causal_selective_drop"]

    composite_score = top_head_score + behavioral_gain + max(causal_selective_drop, 0.0)

    result = BenchmarkResult(
        model_name=model_name,
        prompt_family=prompt_family,
        n_prompts=n_prompts,
        seq_len=seq_len,
        seed=seed,
        detection_top_head=[int(detection_top_layer), int(detection_top_head)],
        detection_top_head_score=float(detection_top_head_score),
        top_head=[int(top_layer), int(top_head)],
        top_head_score=float(top_head_score),
        repeated_target_logprob=float(repeated_target_logprob),
        control_target_logprob=float(control_target_logprob),
        behavioral_gain=float(behavioral_gain),
        ablated_repeated_target_logprob=float(ablated_repeated_target_logprob),
        ablated_control_target_logprob=float(ablated_control_target_logprob),
        repeated_ablation_drop=float(repeated_ablation_drop),
        control_ablation_drop=float(control_ablation_drop),
        causal_selective_drop=float(causal_selective_drop),
        composite_score=float(composite_score),
        elapsed_seconds=float(time.time() - started),
    )

    details = {
        "result": asdict(result),
        "top_heads": candidate_rows,
        "examples": [
            {
                "label": ex.label,
                "sequence_token_ids": ex.sequence,
                "sequence_tokens": decode_sequence(model, ex.sequence),
                "distractor_token_ids": ex.distractor,
                "distractor_tokens": decode_sequence(model, ex.distractor),
                "target_token_id": ex.target,
                "target_token": model.to_single_str_token(ex.target),
                "target_text": ex.target_text,
                "repeated_text": ex.repeated_text,
                "control_text": ex.control_text,
                "detection_text": ex.detection_text,
                "align_pairs": ex.align_pairs,
            }
            for ex in dataset
        ],
    }
    return result, details


def print_human_summary(details: dict) -> None:
    result = details["result"]
    print("Induction reproduction benchmark")
    print(f"  model:                {result['model_name']}")
    print(f"  prompt family:        {result['prompt_family']}")
    print(f"  prompts x len:        {result['n_prompts']} x {result['seq_len']}")
    print(f"  best detection head:  L{result['detection_top_head'][0]}H{result['detection_top_head'][1]}")
    print(f"  selected causal head: L{result['top_head'][0]}H{result['top_head'][1]}")
    print(f"  selected head score:  {result['top_head_score']:.4f}")
    print(f"  behavioral gain:      {result['behavioral_gain']:.4f}")
    print(f"  causal selective:     {result['causal_selective_drop']:.4f}")
    print(f"  composite score:      {result['composite_score']:.4f}")
    print("\nCandidate heads:")
    for row in details["top_heads"][:5]:
        print(
            f"  L{row['layer']}H{row['head']}: detect={row['score']:.4f} "
            f"selective={row['causal_selective_drop']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="santacoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-family", default="synthetic", choices=["synthetic", "code"])
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric-only", action="store_true")
    args = parser.parse_args()

    result, details = run_benchmark(
        model_name=args.model_name,
        device=args.device,
        prompt_family=args.prompt_family,
        n_prompts=args.n_prompts,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(details, indent=2))

    if args.metric_only:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_human_summary(details)
        print(f"\nSaved detailed results to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
