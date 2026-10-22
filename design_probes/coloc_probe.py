"""Design probe: France over-merging mechanism = co-located businesses (33% of France S1 share an exact address vs
~7% in train).  Q1: in-domain, are errors concentrated on co-located S1 entities?  Q2: do address-twin features
(how many S1 share the candidate's address; is the candidate the best NAME match among the query's candidates that
share that address) fix them?  Uses the dictionary-enabled India npz (+ US npz)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl, lightgbm as lgb
from load import load_source
from loco import f05_counts, PARAMS
SP = os.path.dirname(os.path.abspath(__file__))
FILES = {"India": "probe_model_India_drop0.0.npz", "US": "probe_model_US_drop0.0.npz"}


def addr_groups(country):
    s1 = load_source("train", 1).filter(pl.col("country") == country)
    ak = s1.select(pl.col("business_address").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.split(" ")
                   .list.eval(pl.element().filter(pl.element().str.len_chars() > 1)).list.sort().list.join(" ").alias("ak"))["ak"]
    codes = ak.cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    size = np.bincount(codes)[codes]
    return codes, size


for c, f in FILES.items():
    d = dict(np.load(f"{SP}/{f}", allow_pickle=True))
    X, y, P_q, P_s, role, names = d["X"], d["y"], d["P_q"], d["P_s"], d["role"], list(d["names"])
    codes, gsize = addr_groups(c)
    # pairwise extras: address-twin count of candidate S1; is it the best name match among the query's candidates in the same address group
    a_twins = gsize[P_s].astype(np.float32)
    grp = codes[P_s]; nsim = X[:, names.index("n_tsort")]
    key = P_q * (codes.max() + 1) + grp
    order = np.lexsort((-nsim, key)); first = np.zeros(len(key), bool)
    fk = np.concatenate([[True], key[order][1:] != key[order][:-1]]); first[order[fk]] = True
    ncolo = np.bincount(np.unique(key, return_inverse=True)[1])[np.unique(key, return_inverse=True)[1]].astype(np.float32)
    X2 = np.column_stack([X, a_twins, first.astype(np.float32), ncolo])
    tr = role[P_s] == 1
    m1 = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), num_boost_round=500)
    m2 = lgb.train(PARAMS, lgb.Dataset(X2[tr], y[tr]), num_boost_round=500)
    p1, p2 = m1.predict(X, num_threads=10), m2.predict(X2, num_threads=10)
    # slice validation S1 entities by co-location of the S1 itself
    colo = gsize > 1
    for lab, sl in [("all val S1", None), ("co-located val S1", colo), ("unique-address val S1", ~colo)]:
        dd = dict(d)
        if sl is not None:
            r = d["role"].copy(); r[(r == 2) & ~sl] = 0; dd["role"] = r
        f1 = max(f05_counts(dd, p1, t)[0] for t in [0.6, 0.7, 0.8]); f2 = max(f05_counts(dd, p2, t)[0] for t in [0.6, 0.7, 0.8])
        sfp1 = f05_counts(dd, p1, 0.7)[1]; sfp2 = f05_counts(dd, p2, 0.7)[1]
        nval = int(((dd["role"] == 2)).sum())
        print(f"[{c}] {lab:22s} n={nval:6,}  F0.5 base={f1:.4f} +addr-twin feats={f2:.4f} ({(f2-f1)*100:+.2f} pts) | singleton-FP rate {sfp1*100:.1f}% -> {sfp2*100:.1f}%")
