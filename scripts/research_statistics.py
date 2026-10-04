"""Family-cluster uncertainty for saved raw intervention effects."""

from __future__ import annotations

import numpy as np


def ratio_summary(rows, numerator, denominator="gap", alpha=0.05, n_boot=10000):
    """Ratio of sums with whole-family bootstrap, not a mean of pair ratios."""
    if not rows:
        return {"n": 0, "families": 0, "estimate": None, "ci": None}
    families = sorted({str(r["family"]) for r in rows})
    totals = np.asarray(
        [
            [
                sum(r[numerator] for r in rows if str(r["family"]) == f),
                sum(r[denominator] for r in rows if str(r["family"]) == f),
            ]
            for f in families
        ],
        dtype=float,
    )
    rng = np.random.default_rng(20260906)
    samples = totals[rng.integers(0, len(families), (n_boot, len(families)))].sum(
        axis=1
    )
    valid = np.abs(samples[:, 1]) > 1e-10
    ratios = samples[valid, 0] / samples[valid, 1]
    den = totals[:, 1].sum()
    return {
        "n": len(rows),
        "families": len(families),
        "estimate": float(totals[:, 0].sum() / den) if abs(den) > 1e-10 else None,
        "ci": np.quantile(ratios, [alpha / 2, 1 - alpha / 2]).tolist()
        if len(ratios)
        else None,
        "alpha": alpha,
        "invalid_bootstrap_fraction": float(1 - valid.mean()),
    }


def accuracy_summary(rows, field="correct"):
    values = [dict(r, success=float(r[field]), unit=1.0) for r in rows]
    return ratio_summary(values, "success", "unit")
