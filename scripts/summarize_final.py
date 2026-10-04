"""Regenerate all final numerical claims from immutable per-example run files."""

import json
from collections import defaultdict
from pathlib import Path

from scripts.research_statistics import accuracy_summary, ratio_summary

ROOT = Path(__file__).resolve().parents[1]


def summarize_behavior(rows, natural=False):
    if natural:
        return {
            "n": len(rows),
            "candidate_accuracy": accuracy_summary(rows),
            "full_accuracy": accuracy_summary(rows, "full_correct"),
        }
    prepared = [
        dict(
            r,
            correct=r["pair_correct"],
            broad_correct=r["clean_candidate_correct"]
            and r["corrupt_candidate_correct"],
            full_correct=r["clean_full_correct"] and r["corrupt_full_correct"],
        )
        for r in rows
    ]
    return {
        "n": len(rows),
        "pair_accuracy": accuracy_summary(prepared),
        "candidate_accuracy": accuracy_summary(prepared, "broad_correct"),
        "full_accuracy": accuracy_summary(prepared, "full_correct"),
        "aligned_n": sum(r["aligned"] for r in rows),
    }


def summarize_validation(rows):
    if not rows:
        return {}
    result = {}
    for subset, selected in (
        ("all_eligible", rows),
        ("pair_solved", [r for r in rows if r["pair_correct"]]),
        (
            "broad_solved",
            [
                r
                for r in rows
                if r["clean_candidate_correct"] and r["corrupt_candidate_correct"]
            ],
        ),
    ):
        keys = list(rows[0]["interventions"])
        interactions = sum(k.startswith("interaction/") for k in keys)
        values = {}
        for key in keys:
            prepared = [
                {
                    "family": r["family"],
                    "gap": r["gap"],
                    "effect": r["interventions"][key],
                }
                for r in selected
            ]
            values[key] = ratio_summary(
                prepared,
                "effect",
                alpha=0.05 / max(interactions, 1)
                if key.startswith("interaction/")
                else 0.05,
            )
        result[subset] = {"n": len(selected), "effects": values}
    return result


def main():
    summary = {}
    for path in sorted((ROOT / "data/final").glob("*.json")):
        data = json.loads(path.read_text())
        if "mode" not in data:
            continue
        result = {"mode": data["mode"], "elapsed_seconds": data["elapsed_seconds"]}
        for key in ("transfer", "natural", "legacy", "behavior"):
            if key not in data:
                continue
            rows = data[key]
            natural = key == "natural"
            result[key] = summarize_behavior(rows, natural)
            groups = defaultdict(list)
            for r in rows:
                label = (
                    r["task"]
                    if natural or "width" not in r
                    else f"{r['task']}|w{r['width']}|comments{r['comment_lines']}"
                )
                groups[label].append(r)
            result[key]["by_condition"] = {
                g: summarize_behavior(rr, natural) for g, rr in groups.items()
            }
        if "validation" in data:
            result["validation"] = summarize_validation(data["validation"])
            if data["mode"].startswith("transfer"):
                result["validation_by_task"] = {
                    task: summarize_validation(
                        [r for r in data["validation"] if r["task"] == task]
                    )
                    for task in ("increase", "continuation", "dedent")
                }
        for collection, value_field in (
            ("curves", "curve"),
            ("mediation", "mediation"),
            ("content", "content"),
        ):
            if collection not in data:
                continue
            rows = data[collection]
            result[collection] = {}
            groups = {"all": rows}
            if collection == "content":
                groups.update(
                    {
                        task: [r for r in rows if r["task"] == task]
                        for task in ("increase", "continuation", "dedent")
                    }
                )
            for group, group_rows in groups.items():
                keys = sorted({k for r in group_rows for k in r[value_field]})
                result[collection][group] = {
                    k: ratio_summary(
                        [
                            {
                                "family": r["family"],
                                "gap": r["gap"],
                                "effect": r[value_field][k],
                            }
                            for r in group_rows
                            if k in r[value_field]
                        ],
                        "effect",
                    )
                    for k in keys
                }
                if collection == "content":
                    result[collection][group].update(
                        {
                            k + "/absolute": ratio_summary(
                                [
                                    {
                                        "family": r["family"],
                                        "gap": r["gap"],
                                        "effect": abs(r[value_field][k]),
                                    }
                                    for r in group_rows
                                    if k in r[value_field]
                                ],
                                "effect",
                            )
                            for k in keys
                        }
                    )
        if data["mode"].startswith("delimiter_events"):
            rows = [r for r in data["behavior"] if r["event_interventions"]]
            result["event_effects"] = (
                {
                    k: ratio_summary(
                        [
                            {
                                "family": r["family"],
                                "gap": r["gap"],
                                "effect": r["event_interventions"][k],
                            }
                            for r in rows
                        ],
                        "effect",
                    )
                    for k in rows[0]["event_interventions"]
                }
                if rows
                else {}
            )
        summary[path.stem] = result
    (ROOT / "data/final/summary.json").write_text(json.dumps(summary, indent=2))
    print("Summarized:", ", ".join(summary))
    for name, r in summary.items():
        if "validation" in r:
            print(name, "n=", r["validation"]["all_eligible"]["n"])
            for k, v in r["validation"]["all_eligible"]["effects"].items():
                if k in (
                    "set_task/damage",
                    "set_task/restore",
                    "keep_task_mlps0/forward",
                    "keep_task_mlps1/forward",
                    "keep_union_mlps1/forward",
                ):
                    print(" ", k, v["estimate"], v["ci"])


if __name__ == "__main__":
    main()
