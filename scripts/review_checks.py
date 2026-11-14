"""Numbers added to the thesis after the advisor's first review, from saved results.

No GPU and no third-party packages:

    python -m scripts.review_checks

* Comment lines (M2): with eight comment lines between the statement and the gap,
  what is the top token in the whole vocabulary? (``data/final/boundary_run.json``)
* Detection ranking (M5): induction detection score of all 384 heads in the
  exploratory sweep (``outputs/induction_heads/induction_exhaustive_results.csv``).
* Delimiters by language (M6): candidate effects split by template language.
  Even templates are Python (swap offset +1), odd ones JavaScript (+2)
  (``data/final/bracket_validation.json``).
* Discovery sample: which pairs ranked the 384 heads (``outputs/*/``), and the
  candidates' delimiter effect by closer in the confirmation split.
* Ranking robustness: the discovery ranking redone on every solved discovery pair
  (``data/extra/discovery_*_all.json``, from ``bracket_modal.py --causal-kind heads``
  and ``indentation_modal.py`` without ``--causal-limit``) against the original.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {(14, 11), (8, 5), (8, 1), (5, 4), (12, 4), (9, 15)}


def comment_lines():
    rows = json.loads((ROOT / "data/final/boundary_run.json").read_text())["transfer"]
    out = {}
    for lines in (0, 8):
        subset = [r for r in rows if r["comment_lines"] == lines]
        tokens = Counter(r["clean_top_token"] for r in subset)
        tokens.update(r["corrupt_top_token"] for r in subset)
        out[lines] = {
            "pairs": len(subset),
            "top_tokens": tokens.most_common(3),
            "mean_target_probability_increase": sum(
                r["clean_probability"] for r in subset if r["task"] == "increase"
            )
            / sum(1 for r in subset if r["task"] == "increase"),
            "mean_target_probability_all": sum(r["clean_probability"] for r in subset)
            / len(subset),
        }
    return out


def detection_ranking(top=10):
    path = ROOT / "outputs/induction_heads/induction_exhaustive_results.csv"
    rows = [
        {k: float(v) for k, v in r.items()} for r in csv.DictReader(path.open())
    ]
    by_effect = sorted(rows, key=lambda r: -r["mean_causal_selective_drop"])
    effect_rank = {(r["layer"], r["head"]): i + 1 for i, r in enumerate(by_effect)}
    rows.sort(key=lambda r: -r["mean_detection_score"])
    return [
        {
            "rank": i + 1,
            "head": f"L{int(r['layer'])}H{int(r['head'])}",
            "candidate": (int(r["layer"]), int(r["head"])) in CANDIDATES,
            "detection": r["mean_detection_score"],
            "selective_effect": r["mean_causal_selective_drop"],
            "effect_rank": effect_rank[(r["layer"], r["head"])],
        }
        for i, r in enumerate(rows[:top])
    ]


def by_language():
    data = json.loads((ROOT / "data/final/bracket_validation.json").read_text())
    rows = [r for r in data["validation"] if r["aligned"] and r["gap"] >= 0.5]

    def ratio(subset, key):
        return sum(r["interventions"][key] for r in subset) / sum(
            r["gap"] for r in subset
        )

    keys = ("set_task/damage", "set_task/restore") + tuple(
        f"{h}/resample/damage" for h in ("L14H10", "L15H15", "L18H13")
    )
    out = {}
    for name, parity in (("python", 0), ("javascript", 1)):
        subset = [r for r in rows if int(r["family"]) % 2 == parity]
        out[name] = {"pairs": len(subset)} | {k: ratio(subset, k) for k in keys}
    return out


def discovery_sample():
    out = {}
    for task, path, target in (
        ("bracket", "outputs/bracket_matching/causal_discovery.json", "clean_target"),
        ("indent", "outputs/indentation/discovery_causal.json", "clean_target_spaces"),
    ):
        data = json.loads((ROOT / path).read_text())
        labels = set(data["causal_sample_labels"])
        rows = [r for r in data["behavioral_rows"] if r["label"] in labels]
        out[task] = {
            "pairs": len(rows),
            "by_target": dict(Counter(str(r[target]) for r in rows)),
            "by_template": dict(Counter(r["template_id"] for r in rows)),
        }
    return out


def by_closer():
    data = json.loads((ROOT / "data/final/bracket_validation.json").read_text())
    rows = [r for r in data["validation"] if r["aligned"] and r["gap"] >= 0.5]
    out = {}
    for closer in (")", "]", "}"):
        subset = [r for r in rows if r["clean_target"] == closer]
        gap = sum(r["gap"] for r in subset)
        out[closer] = {
            "pairs": len(subset),
            "damage": sum(r["interventions"]["set_task/damage"] for r in subset) / gap,
            "restore": sum(r["interventions"]["set_task/restore"] for r in subset) / gap,
        }
    return out


RANKINGS = {
    "bracket": (
        "outputs/bracket_matching/causal_discovery.json",
        "data/extra/discovery_bracket_all.json",
        36,
    ),
    "indent": (
        "outputs/indentation/discovery_causal.json",
        "data/extra/discovery_indent_all.json",
        49,
    ),
}


def _ranking(path):
    rows = json.loads((ROOT / path).read_text())["head_patching"]
    rows = sorted(rows, key=lambda r: -r["bidirectional_score"])
    return [(f"L{r['layer']}H{r['head']}", r["bidirectional_score"]) for r in rows]


def ranking_robustness():
    out = {}
    for task, (original, full, curve_size) in RANKINGS.items():
        a, b = _ranking(original), _ranking(full)
        ra = {h: i for i, (h, _) in enumerate(a)}
        rb = {h: i for i, (h, _) in enumerate(b)}
        n = len(ra)
        spearman = 1 - 6 * sum((ra[h] - rb[h]) ** 2 for h in ra) / (n * (n * n - 1))
        out[task] = {
            "full_sample_pairs": len(
                json.loads((ROOT / full).read_text())["causal_sample_labels"]
            ),
            "top10_full": [(h, round(s, 3)) for h, s in b[:10]],
            "original_top4_positions_in_full": {h: rb[h] + 1 for h, _ in a[:4]},
            "spearman": spearman,
            f"overlap_top{curve_size}": len(
                {h for h, _ in a[:curve_size]} & {h for h, _ in b[:curve_size]}
            ),
        }
    return out


def main():
    print(
        json.dumps(
            {
                "comment_lines": comment_lines(),
                "detection_ranking": detection_ranking(),
                "delimiters_by_language": by_language(),
                "discovery_sample": discovery_sample(),
                "delimiters_by_closer": by_closer(),
                "ranking_robustness": ranking_robustness(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
