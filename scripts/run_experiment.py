"""Run a final experiment on any machine with an NVIDIA GPU; never overwrites a run.

Platform-neutral counterpart of ``scripts/final_modal.py``: the same inputs go to
the same ``scripts.final_experiments.run``, so results are directly comparable.

    python -m scripts.run_experiment --mode bracket --save-path data/rerun/bracket.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODES = (
    "behavior",
    "boundary",
    "bracket",
    "indent",
    "indent_boundary",
    "transfer",
    "transfer_boundary",
    "curve_bracket",
    "curve_indent",
    "content",
    "delimiter_events",
    "delimiter_events_fim",
    "diagnostic",
)


def prepare_dataset(mode: str) -> dict:
    """Frozen datasets plus, for curve modes, the discovery-split head rankings."""
    dataset = json.loads((ROOT / "data/final/transfer_and_natural.json").read_text())
    dataset["rankings"] = {}
    if mode.startswith("curve_"):
        for task, source in (
            ("indent", "outputs/indentation/discovery_causal.json"),
            ("bracket", "outputs/bracket_matching/causal_discovery.json"),
        ):
            ranking = json.loads((ROOT / source).read_text())["head_patching"]
            dataset["rankings"][task] = [[r["layer"], r["head"]] for r in ranking]
    return dataset


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def attach_provenance(payload: dict, dataset: dict, started: float) -> dict:
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    payload["provenance"] = {
        "dataset_sha256": dataset["sha256"],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in scripts
        },
        "git_head": _git_head(),
        "client_wall_seconds": time.time() - started,
        "runtime_input_sha256": hashlib.sha256(
            json.dumps(dataset, sort_keys=True).encode()
        ).hexdigest(),
    }
    payload["source_snapshot"] = {
        str(p.relative_to(ROOT)): p.read_text() for p in scripts
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--save-path", required=True, type=Path)
    args = parser.parse_args()

    if args.save_path.exists():
        raise FileExistsError(f"{args.save_path} exists; choose a new path")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (e.g. a Colab GPU runtime).")

    from scripts.final_experiments import run

    dataset = prepare_dataset(args.mode)
    started = time.time()
    payload = attach_provenance(run(args.mode, dataset), dataset, started)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {args.save_path}; GPU seconds={payload['elapsed_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
