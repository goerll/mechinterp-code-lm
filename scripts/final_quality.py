"""Check intervention endpoints and provenance, export set membership and budget."""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "data/final"
    checks = []
    ledger = []
    for path in sorted(root.glob("*.json")):
        d = json.loads(path.read_text())
        if not isinstance(d, dict) or "mode" not in d:
            continue
        ledger.append(
            {
                "file": path.name,
                "mode": d["mode"],
                "gpu_function_seconds": d["elapsed_seconds"],
                "client_wall_seconds": d.get("provenance", {}).get(
                    "client_wall_seconds"
                ),
                "gpu_only_runtime_estimate_usd": d["elapsed_seconds"] * 0.000306,
            }
        )
        for r in d.get("validation", []):
            for key, value in r["interventions"].items():
                if key == "self_patch_error" or key.startswith("keep_all_error"):
                    checks.append(
                        {
                            "run": path.name,
                            "label": r["label"],
                            "check": key,
                            "absolute_error": abs(value),
                        }
                    )
                    assert abs(value) < 1e-4, (path, r["label"], key, value)
        for r in d.get("curves", []):
            for key, value in r["curve"].items():
                if "/384/" in key:
                    error = abs(value / r["gap"] - 1)
                    checks.append(
                        {
                            "run": path.name,
                            "label": r["label"],
                            "check": key,
                            "absolute_error": error,
                        }
                    )
                    assert error < 1e-4, (path, r["label"], key, value)
        for source, text in d.get("source_snapshot", {}).items():
            assert (
                hashlib.sha256(text.encode()).hexdigest()
                == d["provenance"]["source_sha256"][source]
            )
    (root / "quality_checks.json").write_text(
        json.dumps(
            {
                "passed": True,
                "n_endpoint_checks": len(checks),
                "max_error": max((c["absolute_error"] for c in checks), default=0),
                "checks": checks,
            },
            indent=2,
        )
    )
    budget = {
        "pricing_source": "https://modal.com/pricing",
        "checked_on": "2026-09-06",
        "a10_usd_per_second": 0.000306,
        "runs": ledger,
        "new_gpu_function_seconds": sum(r["gpu_function_seconds"] for r in ledger),
        "gpu_only_runtime_estimate_usd": sum(
            r["gpu_only_runtime_estimate_usd"] for r in ledger
        ),
        "conservative_compute_allowance_used_usd": sum(
            (r["client_wall_seconds"] or r["gpu_function_seconds"]) * 0.00055
            for r in ledger
        ),
        "user_total_limit_usd": 30,
        "final_pass_allowance_usd": 8,
        "note": "Runtime estimate, not an invoice. GPU function time omits some cold starts; CPU, memory, volumes and earlier sessions are not fully reconciled. No new paid subscription.",
    }
    (root / "budget_ledger.json").write_text(json.dumps(budget, indent=2))
    ordering = {}
    for task, path in (
        ("indent", "outputs/indentation/discovery_causal.json"),
        ("bracket", "outputs/bracket_matching/causal_discovery.json"),
    ):
        rows = json.loads((ROOT / path).read_text())["head_patching"]
        ranked = [[r["layer"], r["head"]] for r in rows]
        all_heads = [[l, h] for l in range(24) for h in range(16)]
        ranked += [h for h in all_heads if h not in ranked]
        shuffled = all_heads.copy()
        random.Random(20260906).shuffle(shuffled)
        ordering[task] = {
            "discovery_source": path,
            "source_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "ranked": ranked,
            "random": shuffled,
            "tested_sizes": [0, 4, 8, 16, 32, 64, 128, 384],
        }
    (root / "candidate_sets.json").write_text(json.dumps(ordering, indent=2))
    print(
        "Endpoint checks:",
        len(checks),
        "max error:",
        max((c["absolute_error"] for c in checks), default=0),
    )
    print(
        "New GPU seconds:",
        round(budget["new_gpu_function_seconds"], 1),
        "runtime GPU estimate USD:",
        round(budget["gpu_only_runtime_estimate_usd"], 2),
    )


if __name__ == "__main__":
    main()
