"""Compare a rerun with a saved run of the same mode, value by value.

    python -m scripts.compare_runs data/final/bracket_validation.json data/rerun/bracket.json

Every numeric result (margins, intervention effects) is compared at the same
location in both files. Environment, timing and provenance fields are ignored,
since they are expected to differ between machines.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

IGNORED = {
    "environment",
    "provenance",
    "source_snapshot",
    "elapsed_seconds",
    "torch",
    "gpu",
    "schema_version",
}


def flatten(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in IGNORED:
                yield from flatten(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{path}[{index}]")
    else:
        yield path, value


def compare(reference: dict, rerun: dict, tolerance: float) -> dict:
    left, right = dict(flatten(reference)), dict(flatten(rerun))
    # Fields only in the rerun come from later code versions that record more
    # detail; they are reported but are not a reproduction failure.
    missing = sorted(set(left) - set(right))
    added = sorted(set(right) - set(left))
    numeric, mismatched = [], []
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)):
            if a != b:
                mismatched.append((key, a, b))
        elif not (isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b)):
            if a != b:
                mismatched.append((key, a, b))
        else:
            numeric.append((abs(a - b), key, a, b))
    numeric.sort(reverse=True)
    return {
        "compared_values": len(numeric),
        "max_abs_difference": numeric[0][0] if numeric else 0.0,
        "largest": numeric[:5],
        "above_tolerance": sum(1 for d, *_ in numeric if d > tolerance),
        "non_numeric_mismatches": mismatched,
        "missing_in_rerun": missing,
        "added_in_rerun": added,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="absolute tolerance in logits for differences between GPUs",
    )
    args = parser.parse_args()
    result = compare(
        json.loads(args.reference.read_text()),
        json.loads(args.rerun.read_text()),
        args.tolerance,
    )
    print(f"Values compared:          {result['compared_values']}")
    print(f"Max absolute difference:  {result['max_abs_difference']:.3g}")
    print(f"Above tolerance ({args.tolerance:g}):   {result['above_tolerance']}")
    print(f"Non-numeric mismatches:   {len(result['non_numeric_mismatches'])}")
    print(f"Missing in rerun:         {len(result['missing_in_rerun'])}")
    print(f"Added in rerun (info):    {len(result['added_in_rerun'])}")
    for diff, key, a, b in result["largest"]:
        print(f"  {diff:.3g}  {key}: {a} vs {b}")
    for key, a, b in result["non_numeric_mismatches"][:10]:
        print(f"  mismatch {key}: {a!r} vs {b!r}")
    for key in result["missing_in_rerun"][:10]:
        print(f"  missing in rerun: {key}")
    ok = (
        result["above_tolerance"] == 0
        and not result["non_numeric_mismatches"]
        and not result["missing_in_rerun"]
    )
    print("RESULT:", "match" if ok else "differences found")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
