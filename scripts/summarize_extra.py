"""Summarize the follow-up runs from scripts/extra_experiments.py (no GPU, no numpy).

    python -m scripts.summarize_extra data/extra/curve_bracket_seeds.json ...

Effects are normalized as everywhere else in the thesis: sum of effects over the
eligible pairs divided by the sum of their gaps.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def normalized(rows, key):
    return sum(r["curve"][key] for r in rows) / sum(r["gap"] for r in rows)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low, high = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def summarize_seeds(data):
    rows = data["curves"]
    keys = rows[0]["curve"].keys()
    out = {"n_pairs": len(rows), "ranked": {}, "random": {}}
    for key in keys:
        if key.startswith("ranked/"):
            out["ranked"][key.removeprefix("ranked/")] = normalized(rows, key)
    by_size = defaultdict(lambda: defaultdict(list))
    for key in keys:
        if key.startswith("random_s"):
            _, size, direction = key.split("/")
            by_size[direction][int(size)].append(normalized(rows, key))
    original = data["random_seeds"][0]
    for direction, sizes in by_size.items():
        out["random"][direction] = {
            size: {
                "median": statistics.median(v),
                "p5": percentile(v, 0.05),
                "p95": percentile(v, 0.95),
                "min": min(v),
                "max": max(v),
                "original_seed": normalized(rows, f"random_s{original}/{size}/{direction}"),
            }
            for size, v in sorted(sizes.items())
        }
    return out


def summarize_addback(data):
    rows = data["curves"]
    out = {"n_pairs": len(rows)}
    for direction in ("forward", "reverse"):
        base = normalized(rows, f"base128/{direction}")
        changes = sorted(
            (normalized(rows, k) - base, k.split("/")[1])
            for k in rows[0]["curve"]
            if k.startswith("add/") and k.endswith(direction)
        )
        out[direction] = {
            "base128": base,
            "n_lowering": sum(c < 0 for c, _ in changes),
            "sum_of_changes": sum(c for c, _ in changes),
            "most_lowering": changes[:10],
            "most_raising": changes[-5:],
        }
    return out


def main(paths):
    for path in map(Path, paths):
        data = json.loads(path.read_text())
        summary = (
            summarize_addback(data)
            if data["mode"].startswith("addback")
            else summarize_seeds(data)
        )
        target = path.with_name(path.stem + "_summary.json")
        target.write_text(json.dumps(summary, indent=2))
        print(path.name, "->", target.name)
        print(json.dumps(summary, indent=2)[:3000])


if __name__ == "__main__":
    main(sys.argv[1:])
