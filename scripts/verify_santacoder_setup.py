#!/usr/bin/env python3
"""Verify the pinned environment and SantaCoder loading path."""

from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.patch_santacoder import REVISION, ensure_santacoder_transformers_compat
from transformer_lens import HookedTransformer

EXPECTED_VERSIONS = {
    "torch": "2.6.0+cpu",
    "transformers": "4.57.6",
    "transformer-lens": "2.17.0",
}


def check_versions() -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} version mismatch: expected {expected}, got {actual}")


def main() -> int:
    check_versions()

    print("Compatibility patch results:")
    for result in ensure_santacoder_transformers_compat():
        print(f"  {result}")

    print("\nLoading SantaCoder with TransformerLens...")
    model = HookedTransformer.from_pretrained(
        "santacoder", device="cpu", revision=REVISION
    )

    print("\nLoaded model summary:")
    print(f"  device={model.cfg.device}")
    print(f"  dtype={next(model.parameters()).dtype}")
    print(f"  n_layers={model.cfg.n_layers}")
    print(f"  n_heads={model.cfg.n_heads}")
    print(f"  d_model={model.cfg.d_model}")
    print(f"  n_ctx={model.cfg.n_ctx}")
    print(f"  tokenizer={model.cfg.tokenizer_name}")
    print(f"  revision={REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
