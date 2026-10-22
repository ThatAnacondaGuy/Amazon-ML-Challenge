"""Design probe: which classical remedies reduce the unseen-country penalty? (LOCO US<->India)
 V0 baseline | V1 drop country-fragile features | V2 V1 + monotone constraints | V3 V2 + self-training on the unseen
 country's UNLABELLED pairs (pseudo-labels from confident, unambiguous predictions; no target labels used)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from loco import load, f05_counts, PARAMS
D = {c: load(c) for c in ["India", "US"]}
names = list(D["India"]["names"])
FRAGILE = {"q_indic", "num_q", "num_s", "n_len_q", "n_len_s", "s_twins", "ret_score", "cos_name", "cos_addr", "q_twin_cands"}
MONO_UP = {"ret_ratio", "n_ratio", "n_tsort", "n_tset_core", "n_jw_core", "n_ns_ratio", "n_ns_partial", "n_skel_jacc",
           "n_tok_jacc", "a_tset", "a_jacc", "num_any", "num_first"}
MONO_DOWN = {"ret_rank", "ret_gap"}


def fit(X, y, cols, mono, w=None, rounds=500):
    p = dict(PARAMS)
    if mono:
        p["monotone_constraints"] = [1 if names[c] in MONO_UP else (-1 if names[c] in MONO_DOWN else 0) for c in cols]
        p["monotone_constraints_method"] = "advanced"
    return lgb.train(p, lgb.Dataset(X[:, cols], y, weight=w), num_boost_round=rounds)


def best(d, p):
    r = [(f05_counts(d, p, t)[0], t) for t in [0.5, 0.6, 0.7, 0.8, 0.9]]
    return max(r)


for src, tgt in [("US", "India"), ("India", "US")]:
    S, T = D[src], D[tgt]
    trm = S["role"][S["P_s"]] == 1
    all_cols = list(range(len(names)))
    robust = [i for i, n in enumerate(names) if n not in FRAGILE]
    out = []
    for lab, cols, mono in [("V0 all features", all_cols, False), ("V1 drop fragile", robust, False), ("V2 V1+monotone", robust, True)]:
        m = fit(S["X"][trm], S["y"][trm], cols, mono)
        p = m.predict(T["X"][:, cols], num_threads=10)
        out.append((lab, best(T, p)))
    # V3: self-training on the target's unlabelled candidate pairs (transductive, like France at test time)
    m = fit(S["X"][trm], S["y"][trm], robust, True)
    p = m.predict(T["X"][:, robust], num_threads=10)
    for it in range(2):
        P_q = T["P_q"]
        order = np.lexsort((-p, P_q)); first = np.zeros(len(p), bool)
        fq = np.concatenate([[True], P_q[order][1:] != P_q[order][:-1]]); first[order[fq]] = True
        pos = first & (p >= 0.9); neg = (p <= 0.1) | (~first & (p <= 0.5))
        sel = pos | neg
        Xs = np.vstack([S["X"][trm][:, robust], T["X"][sel][:, robust]])
        ys = np.concatenate([S["y"][trm], pos[sel].astype(np.int8)])
        ws = np.concatenate([np.ones(trm.sum()), np.full(sel.sum(), 0.5)])
        m = lgb.train({**PARAMS, "monotone_constraints": [1 if names[c] in MONO_UP else (-1 if names[c] in MONO_DOWN else 0) for c in robust],
                       "monotone_constraints_method": "advanced"}, lgb.Dataset(Xs, ys, weight=ws), num_boost_round=500)
        p = m.predict(T["X"][:, robust], num_threads=10)
        out.append((f"V3 self-train round {it+1} (pseudo pos={pos.sum():,} neg={neg.sum():,})", best(T, p)))
    # in-domain reference with robust features (does dropping fragile features cost in-domain accuracy?)
    trt = T["role"][T["P_s"]] == 1
    m = fit(T["X"][trt], T["y"][trt], robust, True)
    out.append(("REF in-domain robust+monotone", best(T, m.predict(T["X"][:, robust], num_threads=10))))
    print(f"\n{src} -> {tgt}")
    for lab, (f, t) in out:
        print(f"   {lab:60s} F0.5={f:.4f} (t={t})")
