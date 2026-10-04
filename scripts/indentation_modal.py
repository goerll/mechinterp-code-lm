#!/usr/bin/env python3
"""Run Phase 3 indentation experiments on Modal."""
import json
from pathlib import Path
import modal

app=modal.App("mechinterp-santacoder-indentation")
image=(modal.Image.debian_slim(python_version="3.11").pip_install(
 "torch==2.6.0","transformers==4.57.6","transformer-lens==2.17.0","typeguard==4.5.1",
 "einops==0.8.2","numpy==1.26.4","pandas==2.3.3","tqdm==4.67.3","huggingface_hub").add_local_python_source("scripts"))
cache=modal.Volume.from_name("mechinterp-hf-cache",create_if_missing=True)

@app.function(image=image,gpu="A10G",timeout=60*30,volumes={"/cache":cache})
def remote(mode,split,causal_limit,layers):
 import os; os.environ.setdefault("HF_HOME","/cache/huggingface")
 from scripts.indentation_experiment import run
 return run("cuda",mode,split,causal_limit,layers)

@app.local_entrypoint()
def main(mode="behavior",split="discovery",causal_limit:int|None=None,layers="",save_path="outputs/indentation/discovery.json"):
 payload=remote.remote(mode,split,causal_limit,[int(x) for x in layers.split(",") if x] or None)
 path=Path(save_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2))
 print(json.dumps(payload["behavioral_summary"],indent=2)); print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}\nsaved={path}")
 if payload.get("head_patching"):
  print("Top heads:")
  for row in payload["head_patching"][:10]: print(row)
