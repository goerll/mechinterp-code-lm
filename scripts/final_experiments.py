"""Final exact-intervention experiments. Raw outcomes precede statistical summaries.

All hook mutations clone their input. Cached donors are fixed ordinary forward
passes; interventions operate on z (query-head output) and mlp_out in SantaCoder.
"""

from __future__ import annotations

import time
from collections import defaultdict
from itertools import combinations

import torch
from transformer_lens import utils as U

from scripts.benchmark import load_model
from scripts.patch_santacoder import REVISION

HEADS = ((14, 10), (15, 15), (18, 13), (18, 14), (15, 3))
CONTROLS = ((14, 0), (15, 0), (18, 0))
INDENT = ((14, 10), (18, 13), (18, 14), (15, 3))
BRACKET = ((14, 10), (15, 15), (18, 13))


def legacy_pairs(task):
    if task == "bracket":
        from scripts.bracket_dataset import generate_bracket_pairs

        return [
            {
                "label": p.label,
                "family": str(p.template_id),
                "task": task,
                "clean_prompt": p.clean_prompt,
                "corrupt_prompt": p.corrupt_prompt,
                "clean_target": p.clean_target,
                "corrupt_target": p.corrupt_target,
            }
            for p in generate_bracket_pairs()
            if p.split == "confirmation"
        ]
    from scripts.indentation_dataset import generate_indentation_pairs

    return [
        {
            "label": p.label,
            "family": str(p.template_id),
            "task": task,
            "clean_prompt": p.clean_prompt,
            "corrupt_prompt": p.corrupt_prompt,
            "clean_target": " " * p.clean_target_spaces,
            "corrupt_target": " " * p.corrupt_target_spaces,
        }
        for p in generate_indentation_pairs()
        if p.split == "confirmation"
    ]


def tokens(model, prompt):
    return model.to_tokens(prompt, prepend_bos=True)


def target_id(model, target):
    ids = model.to_tokens(target, prepend_bos=False)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"Target not single-token: {target!r} {ids}")
    return ids[0]


def vocab_candidates(model, task):
    strings = (
        (")", "]", "}") if task == "bracket" else tuple(" " * i for i in range(1, 25))
    )
    return {s: target_id(model, s) for s in strings}


def behavior(model, pairs):
    rows = []
    for index, p in enumerate(pairs):
        a, b = (
            target_id(model, p["clean_target"]),
            target_id(model, p["corrupt_target"]),
        )
        ct, kt = tokens(model, p["clean_prompt"]), tokens(model, p["corrupt_prompt"])
        c = model(ct)[0, -1]
        k = model(kt)[0, -1]
        ids = list(vocab_candidates(model, p["task"]).values())
        aligned = ct.shape == kt.shape
        rows.append(
            dict(
                p,
                clean_margin=float(c[a] - c[b]),
                corrupt_margin=float(k[a] - k[b]),
                gap=float(c[a] - c[b] - k[a] + k[b]),
                aligned=aligned,
                changed=torch.where(ct[0] != kt[0])[0].tolist() if aligned else [],
                clean_length=ct.shape[1],
                corrupt_length=kt.shape[1],
                pair_correct=bool(c[a] > c[b] and k[b] > k[a]),
                clean_candidate_correct=ids[int(c[ids].argmax())] == a,
                corrupt_candidate_correct=ids[int(k[ids].argmax())] == b,
                clean_full_correct=int(c.argmax()) == a,
                corrupt_full_correct=int(k.argmax()) == b,
                clean_probability=float(c.softmax(-1)[a]),
                corrupt_probability=float(k.softmax(-1)[b]),
            )
        )
        rows[-1]["clean_top_token"] = model.tokenizer.decode([int(c.argmax())])
        rows[-1]["corrupt_top_token"] = model.tokenizer.decode([int(k.argmax())])
        if (index + 1) % 64 == 0:
            print(f"behavior {index + 1}/{len(pairs)}", flush=True)
    return rows


def head_hooks(heads, donor, *, mode="resample", position="all", changed=()):
    by = defaultdict(list)
    for l, h in heads:
        by[l].append(h)
    hooks = []
    for layer, hh in by.items():
        name = U.get_act_name("z", layer)

        def hook(value, hook, name=name, hh=tuple(hh)):
            del hook
            result = value.clone()
            pos = (
                slice(None)
                if position == "all"
                else [-1]
                if position == "final"
                else list(changed)
            )
            for h in hh:
                source = donor[name][:, :, h, :]
                if mode == "zero":
                    result[:, pos, h, :] = 0
                elif mode == "temporal_mean":
                    result[:, pos, h, :] = source.mean(1, keepdim=True)
                else:
                    result[:, pos, h, :] = source[:, pos, :]
            return result

        hooks.append((name, hook))
    return hooks


def keep_hooks(model, retained, donor, keep_mlps=False):
    """Replace all nonmembers with fixed donor outputs, preserving clean embeddings."""
    hooks = []
    retained = set(retained)
    for l in range(model.cfg.n_layers):
        name = U.get_act_name("z", l)
        excluded = [h for h in range(model.cfg.n_heads) if (l, h) not in retained]

        def zh(value, hook, name=name, excluded=tuple(excluded)):
            del hook
            result = value.clone()
            if excluded:
                result[:, :, list(excluded), :] = donor[name][:, :, list(excluded), :]
            return result

        hooks.append((name, zh))
        if not keep_mlps:
            name = U.get_act_name("mlp_out", l)

            def mh(value, hook, name=name):
                del value, hook
                return donor[name].clone()

            hooks.append((name, mh))
    return hooks


def validate(model, rows, *, exhaustive=True):
    output = []
    eligible = [r for r in rows if r["aligned"] and r["gap"] >= 0.5]
    names = [
        U.get_act_name(kind, l)
        for l in range(model.cfg.n_layers)
        for kind in ("z", "mlp_out")
    ]
    for index, row in enumerate(eligible):
        ct, kt = (
            tokens(model, row["clean_prompt"]),
            tokens(model, row["corrupt_prompt"]),
        )
        a, b = (
            target_id(model, row["clean_target"]),
            target_id(model, row["corrupt_target"]),
        )
        _, cc = model.run_with_cache(ct, names_filter=names)
        _, kc = model.run_with_cache(kt, names_filter=names)

        def margin(tok, hooks, a=a, b=b):
            value = model.run_with_hooks(tok, fwd_hooks=hooks)[0, -1]
            return float(value[a] - value[b])

        base = row["clean_margin"]
        corrupt = row["corrupt_margin"]
        interventions = {}
        candidates = HEADS + CONTROLS
        for h in candidates:
            label = f"L{h[0]}H{h[1]}"
            modes = (
                ("resample", "zero", "temporal_mean") if exhaustive else ("resample",)
            )
            for mode in modes:
                interventions[f"{label}/{mode}/damage"] = base - margin(
                    ct, head_hooks([h], kc, mode=mode)
                )
            interventions[f"{label}/restore"] = (
                margin(kt, head_hooks([h], cc)) - corrupt
            )
            if exhaustive:
                for pos in ("final", "changed"):
                    interventions[f"{label}/{pos}/damage"] = base - margin(
                        ct, head_hooks([h], kc, position=pos, changed=row["changed"])
                    )
        task_set = BRACKET if row["task"] == "bracket" else INDENT
        sets = {"task": task_set, "union": HEADS, "controls": CONTROLS}
        for label, heads in sets.items():
            interventions[f"set_{label}/damage"] = base - margin(
                ct, head_hooks(heads, kc)
            )
            interventions[f"set_{label}/restore"] = (
                margin(kt, head_hooks(heads, cc)) - corrupt
            )
        if exhaustive:
            for left, right in combinations(task_set, 2):
                label = f"L{left[0]}H{left[1]}+L{right[0]}H{right[1]}"
                joint = base - margin(ct, head_hooks([left, right], kc))
                individual = sum(
                    interventions[f"L{h[0]}H{h[1]}/resample/damage"]
                    for h in (left, right)
                )
                interventions[f"interaction/{label}"] = joint - individual
            # Self-patch sanity and endpoint tests are run on every eligible row.
            interventions["self_patch_error"] = margin(ct, head_hooks(HEADS, cc)) - base
            for direction, tok, donor, origin in (
                ("forward", ct, kc, corrupt),
                ("reverse", kt, cc, base),
            ):
                sign = 1 if direction == "forward" else -1
                empty = margin(tok, keep_hooks(model, (), donor))
                interventions[f"keep_empty/{direction}"] = sign * (empty - origin)
                for label, heads in sets.items():
                    for mlps in (False, True):
                        m = margin(tok, keep_hooks(model, heads, donor, mlps))
                        interventions[f"keep_{label}_mlps{int(mlps)}/{direction}"] = (
                            sign * (m - empty)
                        )
                all_heads = [
                    (l, h)
                    for l in range(model.cfg.n_layers)
                    for h in range(model.cfg.n_heads)
                ]
                all_m = margin(tok, keep_hooks(model, all_heads, donor, True))
                interventions[f"keep_all_error/{direction}"] = all_m - (
                    base if direction == "forward" else corrupt
                )
        output.append(dict(row, interventions=interventions))
        del cc, kc
        if (index + 1) % 8 == 0:
            print(f"validate {row['task']} {index + 1}/{len(eligible)}", flush=True)
    return output


def natural_behavior(model, rows):
    result = []
    ids = list(vocab_candidates(model, "indent").values())
    for r in rows:
        a = target_id(model, r["target"])
        logits = model(tokens(model, r["prompt"]))[0, -1]
        predicted = ids[int(logits[ids].argmax())]
        result.append(
            dict(
                r,
                correct=predicted == a,
                full_correct=int(logits.argmax()) == a,
                probability=float(logits.softmax(-1)[a]),
                predicted_spaces=len(model.tokenizer.decode([predicted])),
            )
        )
    return result


def circuit_curves(model, rows, ranking):
    """Fixed discovery-ranked sizes versus fixed random ordering, all MLPs active."""
    import random

    all_heads = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]
    ranking = [tuple(h) for h in ranking]
    ranking += [h for h in all_heads if h not in ranking]
    random_order = all_heads.copy()
    random.Random(20260906).shuffle(random_order)
    names = [U.get_act_name("z", l) for l in range(model.cfg.n_layers)]
    output = []
    for index, row in enumerate(r for r in rows if r["aligned"] and r["gap"] >= 0.5):
        ct, kt = (
            tokens(model, row["clean_prompt"]),
            tokens(model, row["corrupt_prompt"]),
        )
        a, b = (
            target_id(model, row["clean_target"]),
            target_id(model, row["corrupt_target"]),
        )
        _, cc = model.run_with_cache(ct, names_filter=names)
        _, kc = model.run_with_cache(kt, names_filter=names)
        result = {}
        for kind, order in (("ranked", ranking), ("random", random_order)):
            for size in (0, 4, 8, 16, 32, 64, 128, 384):
                for direction, tok, donor in (("forward", ct, kc), ("reverse", kt, cc)):
                    logits = model.run_with_hooks(
                        tok, fwd_hooks=keep_hooks(model, order[:size], donor, True)
                    )[0, -1]
                    margin = float(logits[a] - logits[b])
                    effect = (
                        margin - row["corrupt_margin"]
                        if direction == "forward"
                        else row["clean_margin"] - margin
                    )
                    result[f"{kind}/{size}/{direction}"] = effect
        output.append(dict(row, curve=result))
        if (index + 1) % 16 == 0:
            print(f"curve {row['task']} {index + 1}", flush=True)
    return output


def mediation(model, rows, paths):
    results = []
    for row in [r for r in rows if r["aligned"] and r["gap"] >= 0.5]:
        ct, kt = (
            tokens(model, row["clean_prompt"]),
            tokens(model, row["corrupt_prompt"]),
        )
        a, b = (
            target_id(model, row["clean_target"]),
            target_id(model, row["corrupt_target"]),
        )
        names = sorted({U.get_act_name("z", h[0]) for pair in paths for h in pair})
        _, cc = model.run_with_cache(ct, names_filter=names)
        _, kc = model.run_with_cache(kt, names_filter=names)
        effects = {}
        for source, dest in paths:
            dn = U.get_act_name("z", dest[0])
            label = f"L{source[0]}H{source[1]}-L{dest[0]}H{dest[1]}"
            for direction, tok, donor in (("damage", ct, kc), ("restore", kt, cc)):
                with model.hooks(fwd_hooks=head_hooks([source], donor)):
                    _, propagated = model.run_with_cache(tok, names_filter=[dn])
                logits = model.run_with_hooks(
                    tok, fwd_hooks=head_hooks([dest], propagated)
                )[0, -1]
                margin = float(logits[a] - logits[b])
                effects[label + "/" + direction] = (
                    row["clean_margin"] - margin
                    if direction == "damage"
                    else margin - row["corrupt_margin"]
                )
        results.append(dict(row, mediation=effects))
    return results


def content_transfer(model, rows):
    """Final-position interchange: lexical invariance and absolute-column transfer."""
    valid = [r for r in rows if r["aligned"] and r["gap"] >= 0.5]
    names = sorted({U.get_act_name("z", h[0]) for h in INDENT})
    output = []
    for index, row in enumerate(valid):
        tok = tokens(model, row["clean_prompt"])
        a = target_id(model, row["clean_target"])
        b = target_id(model, row["corrupt_target"])
        effects = {}
        donor_labels = {}
        lexical = [
            r
            for r in rows
            if r["family"] != row["family"]
            and r["task"] == row["task"]
            and r["width"] == row["width"]
            and r["depth"] == row["depth"]
            and r["comment_lines"] == row["comment_lines"]
            and r["clean_target"] == row["clean_target"]
        ]
        column = [
            r
            for r in rows
            if r["family"] != row["family"]
            and r["task"] == row["task"]
            and r["width"] != row["width"]
            and r["comment_lines"] == row["comment_lines"]
            and r["clean_target"] == row["clean_target"]
        ]
        for condition, pool in (
            ("same_style", lexical),
            ("same_column_cross_style", column),
        ):
            if not pool:
                continue
            donor = pool[0]
            donor_labels[condition] = donor["label"]
            for side in ("clean", "corrupt"):
                _, cache = model.run_with_cache(
                    tokens(model, donor[side + "_prompt"]), names_filter=names
                )
                for head in INDENT:
                    logits = model.run_with_hooks(
                        tok, fwd_hooks=head_hooks([head], cache, position="final")
                    )[0, -1]
                    m = float(logits[a] - logits[b])
                    effects[f"{condition}/{side}/L{head[0]}H{head[1]}"] = (
                        row["clean_margin"] - m
                    )
        output.append(dict(row, content=effects, donors=donor_labels))
        if (index + 1) % 32 == 0:
            print(f"content {index + 1}/{len(valid)}", flush=True)
    return output


def run(mode, dataset=None):
    import importlib.metadata
    import platform

    torch.set_grad_enabled(False)
    started = time.time()
    model = load_model(device="cuda")
    payload = {
        "schema_version": 2,
        "mode": mode,
        "model_revision": REVISION,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "heads": HEADS,
    }
    payload["architecture"] = {
        k: getattr(model.cfg, k, None)
        for k in (
            "d_model",
            "n_layers",
            "n_heads",
            "n_key_value_heads",
            "d_head",
            "n_ctx",
            "d_mlp",
        )
    }
    payload["environment"] = {
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "dtype": str(next(model.parameters()).dtype),
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }
    if mode == "behavior":
        payload["transfer"] = behavior(model, dataset["transfer"])
        payload["natural"] = natural_behavior(model, dataset["natural"])
    elif mode in ("delimiter_events", "delimiter_events_fim"):
        from scripts.delimiter_completion import run as event_run
        from scripts.final_datasets import delimiter_completion_pairs

        pairs = delimiter_completion_pairs()
        if mode == "delimiter_events_fim":
            from scripts.final_datasets import fim

            for row in pairs:
                suffix = (")" if row["depth"] == 2 else "") + "\n\n"
                row["clean_prompt"] = fim(row["clean_prompt"], suffix)
                row["corrupt_prompt"] = fim(row["corrupt_prompt"], suffix)
                row["label"] += "_fim"
        payload.update(event_run(model, pairs))
    elif mode == "boundary":
        from scripts.final_datasets import shift_boundary

        payload["transfer"] = behavior(
            model, [shift_boundary(r) for r in dataset["transfer"]]
        )
        payload["natural"] = natural_behavior(
            model, [shift_boundary(r) for r in dataset["natural"]]
        )
        payload["legacy"] = behavior(
            model, [shift_boundary(r) for r in legacy_pairs("indent")]
        )
    elif mode == "indent_boundary":
        from scripts.final_datasets import shift_boundary

        rows = behavior(model, [shift_boundary(r) for r in legacy_pairs("indent")])
        payload["behavior"] = rows
        payload["validation"] = validate(model, rows)
    elif mode == "transfer_boundary":
        from scripts.final_datasets import shift_boundary

        rows = behavior(model, [shift_boundary(r) for r in dataset["transfer"]])
        payload["behavior"] = rows
        payload["validation"] = validate(model, rows, exhaustive=False)
    elif mode in ("curve_indent", "curve_bracket"):
        from scripts.final_datasets import shift_boundary

        task = mode.split("_")[1]
        pairs = legacy_pairs(task)
        if task == "indent":
            pairs = [shift_boundary(r) for r in pairs]
        rows = behavior(model, pairs)
        payload["behavior"] = rows
        payload["curves"] = circuit_curves(model, rows, dataset["rankings"][task])
        paths = (
            [((14, 10), (15, 15))]
            if task == "bracket"
            else [((14, 10), (15, 3)), ((14, 10), (18, 14))]
        )
        payload["mediation"] = mediation(model, rows, paths)
    elif mode == "content":
        from scripts.final_datasets import shift_boundary

        rows = behavior(model, [shift_boundary(r) for r in dataset["transfer"]])
        payload["behavior"] = rows
        payload["content"] = content_transfer(model, rows)
    elif mode == "diagnostic":
        selected = []
        for task in ("increase", "continuation", "dedent"):
            selected.extend(
                [
                    r
                    for r in dataset["transfer"]
                    if r["task"] == task and r["width"] == 4 and r["comment_lines"] == 0
                ][:2]
            )
        selected.extend(legacy_pairs("indent")[:2])
        examples = []
        for r in selected:
            tok = tokens(model, r["clean_prompt"])
            logits = model(tok)[0, -1]
            top = logits.softmax(-1).topk(8)
            generated = model.generate(
                tok, max_new_tokens=6, do_sample=False, verbose=False
            )
            examples.append(
                dict(
                    r,
                    top=[
                        {"text": model.tokenizer.decode([int(i)]), "p": float(p)}
                        for p, i in zip(top.values, top.indices)
                    ],
                    generated=model.tokenizer.decode(
                        generated[0, tok.shape[1] :].tolist()
                    ),
                )
            )
        payload["examples"] = examples
    elif mode in ("bracket", "indent"):
        rows = behavior(model, legacy_pairs(mode))
        payload["behavior"] = rows
        payload["validation"] = validate(model, rows)
    elif mode == "transfer":
        rows = behavior(model, dataset["transfer"])
        payload["behavior"] = rows
        payload["validation"] = validate(model, rows, exhaustive=False)
    else:
        raise ValueError(mode)
    payload["elapsed_seconds"] = time.time() - started
    return payload
