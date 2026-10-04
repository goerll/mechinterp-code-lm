"""Budget-bounded final experiment launcher; never overwrites an existing run.

Modal only provides the GPU. Inputs and provenance come from
``scripts.run_experiment``, which also runs the same experiments without Modal.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("mechinterp-thesis-final")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.6",
        "transformer-lens==2.17.0",
        "typeguard==4.5.1",
        "einops==0.8.2",
        "numpy==1.26.4",
        "pandas==2.3.3",
        "tqdm==4.67.3",
        "huggingface_hub",
    )
    .add_local_python_source("scripts")
)
cache = modal.Volume.from_name("mechinterp-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/cache": cache})
def remote(mode, dataset):
    import os

    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    from scripts.final_experiments import run

    return run(mode, dataset)


@app.local_entrypoint()
def main(mode: str = "behavior", save_path: str = "data/final/behavior_run.json"):
    from scripts.run_experiment import attach_provenance, prepare_dataset

    path = Path(save_path)
    if path.exists():
        raise FileExistsError(path)
    dataset = prepare_dataset(mode)
    started = time.time()
    payload = attach_provenance(remote.remote(mode, dataset), dataset, started)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {path}; GPU function seconds={payload['elapsed_seconds']:.1f}")
