"""Test closer events, including merged tokens such as `)\\n`, on valid prefixes."""

from collections import defaultdict

from transformer_lens import utils as U

from scripts.final_experiments import BRACKET, CONTROLS, HEADS, head_hooks, tokens


def run(model, pairs):
    groups = defaultdict(list)
    for i in range(model.cfg.d_vocab):
        s = model.tokenizer.decode([i]).lstrip(" \t")
        if s and s[0] in ")]}":
            groups[s[0]].append(i)

    def scores(logits):
        return {c: float(logits[ids].logsumexp(-1)) for c, ids in groups.items()}

    names = sorted({U.get_act_name("z", h[0]) for h in HEADS + CONTROLS})
    out = []
    for index, p in enumerate(pairs):
        ct, kt = tokens(model, p["clean_prompt"]), tokens(model, p["corrupt_prompt"])
        cl, cc = model.run_with_cache(ct, names_filter=names)
        kl, kc = model.run_with_cache(kt, names_filter=names)
        c, k = cl[0, -1], kl[0, -1]
        cs, ks = scores(c), scores(k)
        a, b = p["clean_target"], p["corrupt_target"]
        cm, km = cs[a] - cs[b], ks[a] - ks[b]
        cp = model.tokenizer.decode([int(c.argmax())]).lstrip(" \t")
        kp = model.tokenizer.decode([int(k.argmax())]).lstrip(" \t")
        row = dict(
            p,
            clean_margin=cm,
            corrupt_margin=km,
            gap=cm - km,
            aligned=ct.shape == kt.shape,
            pair_correct=cs[a] > cs[b] and ks[b] > ks[a],
            clean_candidate_correct=max(cs, key=cs.get) == a,
            corrupt_candidate_correct=max(ks, key=ks.get) == b,
            clean_full_correct=cp.startswith(a),
            corrupt_full_correct=kp.startswith(b),
            clean_top_token=cp,
            corrupt_top_token=kp,
            clean_event_probability=float(c.softmax(-1)[groups[a]].sum()),
            corrupt_event_probability=float(k.softmax(-1)[groups[b]].sum()),
        )
        interventions = {}
        if row["aligned"] and row["gap"] >= 0.5:
            for label, heads in [
                (f"L{l}H{h}", [(l, h)]) for l, h in HEADS + CONTROLS
            ] + [("set_task", BRACKET)]:
                dl = model.run_with_hooks(ct, fwd_hooks=head_hooks(heads, kc))[0, -1]
                ds = scores(dl)
                rl = model.run_with_hooks(kt, fwd_hooks=head_hooks(heads, cc))[0, -1]
                rs = scores(rl)
                interventions[label + "/damage"] = cm - (ds[a] - ds[b])
                interventions[label + "/restore"] = (rs[a] - rs[b]) - km
        row["event_interventions"] = interventions
        out.append(row)
        if (index + 1) % 32 == 0:
            print(f"delimiter events {index + 1}/{len(pairs)}", flush=True)
    return {"behavior": out, "event_groups": {k: len(v) for k, v in groups.items()}}
