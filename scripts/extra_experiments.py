"""Follow-up experiments requested in the advisor's first review (Nov 2026).

Two questions about the sufficiency curves, both with all MLPs active and the same
eligible confirmation pairs, hooks and effect definitions as ``circuit_curves``:

* ``curve_*_seeds``: is the random-order curve stable across draws, and where
  between 32 and 64 discovery-ranked heads does recovery cross 80%?  Adds ranked
  sizes 40, 48 and 56 and 20 random orders at every grid size. The first random
  seed is the one used in the thesis, so its values must match ``curve_*.json``.
* ``addback_bracket``: with the 128 top-ranked heads, delimiters recover ~111%.
  Do some of the remaining 256 heads oppose the behavior?  Each remaining head is
  added back to the 128-head set on its own; an effect drop below the 128-head
  value means that head pulls the margin the other way.
* ``curve_*_fine``: decided at the advisor meeting (2026-11-12). The coarse runs
  bracket the minimum between 33 and 40 heads (delimiters) and 49 and 56
  (indentation); this mode tests every discovery-ranked size from 33 to 64.
* ``curve_*_small``: asked in the advisor's second review (2026-11-17). Tests every
  discovery-ranked size from 1 to 32, so that the smallest sufficient size is the
  minimum over all sizes up to 64 and not only over the tested grid.
"""

from __future__ import annotations

import random

from transformer_lens import utils as U

from scripts.final_experiments import keep_hooks, target_id, tokens

ORIGINAL_SEED = 20260906
RANDOM_SEEDS = (ORIGINAL_SEED,) + tuple(range(1, 20))
GRID = (4, 8, 16, 32, 64, 128)
EXTRA_RANKED = (40, 48, 56)
FINE_RANKED = tuple(range(33, 65))
SMALL_RANKED = tuple(range(1, 33))


def _all_heads(model):
    return [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]


def _full_ranking(model, ranking):
    ranking = [tuple(h) for h in ranking]
    return ranking + [h for h in _all_heads(model) if h not in ranking]


def _random_order(model, seed):
    order = _all_heads(model)
    random.Random(seed).shuffle(order)
    return order


def _eligible(rows):
    return [r for r in rows if r["aligned"] and r["gap"] >= 0.5]


def _effects(model, row, sets):
    """Effect of keeping each head set, in both directions (as in circuit_curves)."""
    names = [U.get_act_name("z", l) for l in range(model.cfg.n_layers)]
    ct, kt = tokens(model, row["clean_prompt"]), tokens(model, row["corrupt_prompt"])
    a, b = target_id(model, row["clean_target"]), target_id(model, row["corrupt_target"])
    _, cc = model.run_with_cache(ct, names_filter=names)
    _, kc = model.run_with_cache(kt, names_filter=names)
    result = {}
    for label, heads in sets:
        for direction, tok, donor in (("forward", ct, kc), ("reverse", kt, cc)):
            logits = model.run_with_hooks(
                tok, fwd_hooks=keep_hooks(model, heads, donor, True)
            )[0, -1]
            margin = float(logits[a] - logits[b])
            result[f"{label}/{direction}"] = (
                margin - row["corrupt_margin"]
                if direction == "forward"
                else row["clean_margin"] - margin
            )
    del cc, kc
    return result


def curve_seeds(model, rows, ranking):
    ranked = _full_ranking(model, ranking)
    sets = [(f"ranked/{n}", ranked[:n]) for n in EXTRA_RANKED]
    for seed in RANDOM_SEEDS:
        order = _random_order(model, seed)
        sets += [(f"random_s{seed}/{n}", order[:n]) for n in GRID]
    output = []
    for index, row in enumerate(_eligible(rows)):
        output.append(dict(row, curve=_effects(model, row, sets)))
        if (index + 1) % 8 == 0:
            print(f"seeds {row['task']} {index + 1}", flush=True)
    return output


def curve_fine(model, rows, ranking, sizes=FINE_RANKED):
    ranked = _full_ranking(model, ranking)
    sets = [(f"ranked/{n}", ranked[:n]) for n in sizes]
    output = []
    for index, row in enumerate(_eligible(rows)):
        output.append(dict(row, curve=_effects(model, row, sets)))
        if (index + 1) % 8 == 0:
            print(f"fine {row['task']} {index + 1}", flush=True)
    return output


def addback(model, rows, ranking, base_size=128):
    ranked = _full_ranking(model, ranking)
    base = ranked[:base_size]
    sets = [(f"base{base_size}", base)]
    sets += [(f"add/L{l}H{h}", base + [(l, h)]) for l, h in ranked[base_size:]]
    output = []
    for index, row in enumerate(_eligible(rows)):
        output.append(dict(row, curve=_effects(model, row, sets)))
        if (index + 1) % 8 == 0:
            print(f"addback {index + 1}", flush=True)
    return output


EXTRA_MODES = (
    "curve_bracket_seeds",
    "curve_indent_seeds",
    "addback_bracket",
    "curve_bracket_fine",
    "curve_indent_fine",
    "curve_bracket_small",
    "curve_indent_small",
)


def run_extra(model, mode, dataset):
    from scripts.final_datasets import shift_boundary
    from scripts.final_experiments import behavior, legacy_pairs

    task = "bracket" if "bracket" in mode else "indent"
    pairs = legacy_pairs(task)
    if task == "indent":
        # Same formulation as curve_indent (shifted gap boundary).
        pairs = [shift_boundary(r) for r in pairs]
    rows = behavior(model, pairs)
    ranking = dataset["rankings"][task]
    payload = {"behavior": rows, "random_seeds": RANDOM_SEEDS}
    if mode.endswith("_fine"):
        payload["curves"] = curve_fine(model, rows, ranking)
    elif mode.endswith("_small"):
        payload["curves"] = curve_fine(model, rows, ranking, SMALL_RANKED)
    elif mode.startswith("curve_"):
        payload["curves"] = curve_seeds(model, rows, ranking)
    else:
        payload["curves"] = addback(model, rows, ranking)
    return payload
