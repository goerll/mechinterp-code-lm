#!/usr/bin/env python3
"""Behavioral and causal evaluation of matched FIM indentation prompts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import platform
import time

import numpy as np
import torch
from transformer_lens import utils as tl_utils

from scripts.benchmark import load_model
from scripts.indentation_dataset import IndentationPair, dataset_fingerprint, generate_indentation_pairs
from scripts.patch_santacoder import REVISION

torch.set_grad_enabled(False)
TARGET_SPACES = (8, 12, 16)


def _tokens(model, prompt):
    return model.to_tokens(prompt, prepend_bos=True)


def _target_ids(model):
    result = {}
    for count in TARGET_SPACES:
        ids = model.to_tokens(" " * count, prepend_bos=False)[0].tolist()
        if len(ids) != 1:
            raise RuntimeError(f"{count} spaces are not one token: {ids}")
        result[count] = ids[0]
    return result


def _candidate_logits(model, tokens, target_ids):
    logits = model(tokens)[0, -1]
    return {count: float(logits[token_id]) for count, token_id in target_ids.items()}


def behavioral_rows(model, pairs):
    target_ids = _target_ids(model)
    rows = []
    for pair in pairs:
        clean_tokens, corrupt_tokens = _tokens(model, pair.clean_prompt), _tokens(model, pair.corrupt_prompt)
        negative_tokens = _tokens(model, pair.negative_control_prompt)
        diffs = torch.where(clean_tokens[0] != corrupt_tokens[0])[0].tolist() if clean_tokens.shape == corrupt_tokens.shape else []
        clean = _candidate_logits(model, clean_tokens, target_ids)
        corrupt = _candidate_logits(model, corrupt_tokens, target_ids)
        negative = _candidate_logits(model, negative_tokens, target_ids)
        ct, kt = pair.clean_target_spaces, pair.corrupt_target_spaces
        cm = clean[ct] - clean[kt]
        km = corrupt[ct] - corrupt[kt]
        nm = negative[ct] - negative[kt]
        rows.append({**asdict(pair), "aligned": clean_tokens.shape == corrupt_tokens.shape,
                     "clean_length": clean_tokens.shape[1], "corrupt_length": corrupt_tokens.shape[1],
                     "negative_length": negative_tokens.shape[1], "differing_token_positions": diffs,
                     "clean_logits": clean, "corrupt_logits": corrupt, "negative_control_logits": negative,
                     "clean_margin": cm, "corrupt_clean_margin": km, "negative_control_margin": nm,
                     "structural_effect": cm-km, "negative_control_effect": cm-nm,
                     "clean_prediction": max(clean, key=clean.get), "corrupt_prediction": max(corrupt, key=corrupt.get)})
    return rows


def _lcb(values, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(values, dtype=float)
    return float(np.quantile([rng.choice(a, len(a), replace=True).mean() for _ in range(2000)], .05))


def summarize_behavior(rows):
    paired = [r["clean_prediction"] == r["clean_target_spaces"] and r["corrupt_prediction"] == r["corrupt_target_spaces"] for r in rows]
    structural = [r["structural_effect"] for r in rows]
    negative = [abs(r["negative_control_effect"]) for r in rows]
    cells = defaultdict(list)
    for r, ok in zip(rows, paired): cells[f'{r["depth_band"]}|{r["compound"]}'].append(ok)
    return {"n_pairs": len(rows), "aligned_fraction": np.mean([r["aligned"] for r in rows]),
            "single_difference_fraction": np.mean([len(r["differing_token_positions"]) == 1 for r in rows]),
            "paired_accuracy": float(np.mean(paired)), "paired_accuracy_bootstrap_95_lcb": _lcb(paired),
            "structural_sensitivity": float(np.mean([x > 0 for x in structural])),
            "mean_structural_effect": float(np.mean(structural)),
            "mean_absolute_negative_control_effect": float(np.mean(negative)),
            "negative_to_structural_effect_ratio": float(np.mean(negative)/max(np.mean(np.abs(structural)), 1e-12)),
            "paired_accuracy_by_condition": {k: float(np.mean(v)) for k,v in sorted(cells.items())}}


def select_solved(rows, limit=None):
    solved = [r for r in rows if r["clean_prediction"] == r["clean_target_spaces"] and r["corrupt_prediction"] == r["corrupt_target_spaces"]]
    return solved if limit is None else solved[:limit]


def _pair(row):
    return IndentationPair(**{k: row[k] for k in IndentationPair.__dataclass_fields__})


def component_patch_discovery(model, rows, layers=None):
    ids = _target_ids(model); layers = layers or list(range(model.cfg.n_layers))
    components = ("attn_out", "mlp_out", "resid_post"); values = defaultdict(list); reverse = defaultdict(list)
    for row in rows:
        pair = _pair(row); clean = _tokens(model,pair.clean_prompt); corrupt = _tokens(model,pair.corrupt_prompt)
        denominator = row["structural_effect"]
        if abs(denominator) < 1e-6: continue
        for layer in layers:
            names = [tl_utils.get_act_name(c,layer) for c in components]
            _, cc = model.run_with_cache(clean,names_filter=names); _, kc = model.run_with_cache(corrupt,names_filter=names)
            for component,name in zip(components,names):
                def forward(x,hook,name=name):
                    del hook; x=x.clone(); x[0,-1]=cc[name][0,-1]; return x
                def backward(x,hook,name=name):
                    del hook; x=x.clone(); x[0,-1]=kc[name][0,-1]; return x
                fl=model.run_with_hooks(corrupt,fwd_hooks=[(name,forward)])[0,-1]
                rl=model.run_with_hooks(clean,fwd_hooks=[(name,backward)])[0,-1]
                a,b=ids[pair.clean_target_spaces],ids[pair.corrupt_target_spaces]
                values[layer,component].append(float(((fl[a]-fl[b])-row["corrupt_clean_margin"])/denominator))
                reverse[layer,component].append(float((row["clean_margin"]-(rl[a]-rl[b]))/denominator))
    out=[]
    for key,v in values.items():
        rv=reverse[key]; out.append({"layer":key[0],"component":key[1],"clean_to_corrupt_restoration_mean":float(np.mean(v)),
          "corrupt_to_clean_damage_mean":float(np.mean(rv)),"bidirectional_score":float((np.mean(v)+np.mean(rv))/2),"n_pairs":len(v)})
    return sorted(out,key=lambda x:x["bidirectional_score"],reverse=True)


def head_patch_discovery(model, rows, layers=None):
    ids=_target_ids(model); layers=layers or list(range(model.cfg.n_layers)); vals=defaultdict(list); rev=defaultdict(list)
    for row in rows:
        pair=_pair(row); clean=_tokens(model,pair.clean_prompt); corrupt=_tokens(model,pair.corrupt_prompt); den=row["structural_effect"]
        if abs(den)<1e-6: continue
        a,b=ids[pair.clean_target_spaces],ids[pair.corrupt_target_spaces]
        for layer in layers:
            name=tl_utils.get_act_name("z",layer); _,cc=model.run_with_cache(clean,names_filter=[name]); _,kc=model.run_with_cache(corrupt,names_filter=[name]); nh=model.cfg.n_heads
            def fwd(z,hook):
                del hook; z=z.clone(); idx=torch.arange(nh,device=z.device); z[idx,:,idx,:]=cc[name][0,:,idx,:].permute(1,0,2); return z
            def back(z,hook):
                del hook; z=z.clone(); idx=torch.arange(nh,device=z.device); z[idx,:,idx,:]=kc[name][0,:,idx,:].permute(1,0,2); return z
            fl=model.run_with_hooks(corrupt.repeat(nh,1),fwd_hooks=[(name,fwd)])[:,-1]; rl=model.run_with_hooks(clean.repeat(nh,1),fwd_hooks=[(name,back)])[:,-1]
            for h in range(nh):
                vals[layer,h].append(float(((fl[h,a]-fl[h,b])-row["corrupt_clean_margin"])/den)); rev[layer,h].append(float((row["clean_margin"]-(rl[h,a]-rl[h,b]))/den))
    out=[]
    for key,v in vals.items():
        rv=rev[key]; out.append({"layer":key[0],"head":key[1],"clean_to_corrupt_restoration_mean":float(np.mean(v)),"corrupt_to_clean_damage_mean":float(np.mean(rv)),"bidirectional_score":float((np.mean(v)+np.mean(rv))/2),"n_pairs":len(v)})
    return sorted(out,key=lambda x:x["bidirectional_score"],reverse=True)


def run(device, mode="behavior", split="discovery", causal_limit=None, layers=None):
    start=time.time(); model=load_model(device=device); pairs=[p for p in generate_indentation_pairs() if p.split==split]; rows=behavioral_rows(model,pairs)
    payload={"schema_version":1,"metadata":{"model":"bigcode/santacoder","model_revision":REVISION,"torch":torch.__version__,"python":platform.python_version()},
             "dataset_fingerprint_sha256":dataset_fingerprint(generate_indentation_pairs()),"split":split,"behavioral_summary":summarize_behavior(rows),"behavioral_rows":rows}
    if mode=="causal":
        selected=select_solved(rows,causal_limit); payload["causal_sample_labels"]=[r["label"] for r in selected]
        payload["component_patching"]=component_patch_discovery(model,selected,layers); payload["head_patching"]=head_patch_discovery(model,selected,layers)
    payload["elapsed_seconds"]=time.time()-start; return payload
