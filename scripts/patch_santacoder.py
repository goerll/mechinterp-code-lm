#!/usr/bin/env python3
"""Patch SantaCoder remote code for newer transformers releases."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoConfig

REPO_ID = "bigcode/santacoder"
MODEL_FILE = "modeling_gpt2_mq.py"
TARGET = "from transformers.modeling_utils import PreTrainedModel, SequenceSummary"
REPLACEMENT = "from transformers.modeling_utils import PreTrainedModel"


def _hf_home() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser()
    return Path.home() / ".cache" / "huggingface"


def _dynamic_module_path(commit_hash: str) -> Path:
    cache_root = Path(os.environ.get("TRANSFORMERS_CACHE", _hf_home())).expanduser()
    return (
        cache_root
        / "modules"
        / "transformers_modules"
        / "bigcode"
        / "santacoder"
        / commit_hash
        / MODEL_FILE
    )


def _patch_file(path: Path) -> str:
    if not path.exists():
        return f"missing: {path}"

    content = path.read_text()
    if TARGET not in content:
        return f"already compatible: {path}"

    path.write_text(content.replace(TARGET, REPLACEMENT))
    return f"patched: {path}"


def _ensure_dynamic_module_copy(snapshot_path: Path, dynamic_module_path: Path) -> str:
    dynamic_module_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_module_path.write_text(snapshot_path.read_text())
    return f"synced snapshot to dynamic module cache: {dynamic_module_path}"


def ensure_santacoder_transformers_compat() -> list[str]:
    config = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    commit_hash = getattr(config, "_commit_hash", None)
    if not commit_hash:
        raise RuntimeError(f"Could not determine commit hash for {REPO_ID}")

    snapshot_path = Path(hf_hub_download(REPO_ID, MODEL_FILE, revision=commit_hash))
    results = [_patch_file(snapshot_path)]

    dynamic_module_path = _dynamic_module_path(commit_hash)
    if dynamic_module_path != snapshot_path:
        results.append(_ensure_dynamic_module_copy(snapshot_path, dynamic_module_path))
        results.append(_patch_file(dynamic_module_path))

    return results


def main() -> int:
    for result in ensure_santacoder_transformers_compat():
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
