"""Design probe: leave-one-country-out transfer + decision-rule comparison, from saved probe_model npz files.

For each (train_country -> eval_country): train LightGBM on train-role pairs of train_country, evaluate macro F0.5
on val-role S1 entities of eval_country (exact formula incl. singletons and blocking misses via true counts).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from probe_core import load_country
SP = os.path.dirname(os.path.abspath(__file__))
FILES = {"India": os.environ.get("INDIA_NPZ", "probe_model_India.npz"), "US": "probe_model_US_drop0.0.npz"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=7)


def load(c):
    d = dict(np.load(f"{SP}/{FILES[c]}", allow_pickle=True))
    if "tcount" not in d:
        s1, s23, truth_q = load_country("train", c)
        d["tcount"] = np.bincount(truth_q[truth_q >= 0], minlength=s1.height)
    return d


def f05_counts(d, p, t, margin=0.0):
    """rule B (+ optional ambiguity margin) evaluated on val S1 of dataset d; uses candidate labels + true counts."""
    P_q, P_s, y, role, tcount = d["P_q"], d["P_s"], d["y"], d["role"], d["tcount"]
    order = np.lexsort((-p, P_q)); Pq, Ps, pp, yy = P_q[order], P_s[order], p[order], y[order]
    first = np.concatenate([[True], Pq[1:] != Pq[:-1]])
    second = np.zeros_like(pp); nxt = np.concatenate([Pq[1:] == Pq[:-1], [False]])
    second[np.where(first)[0]] = np.where(nxt[np.where(first)[0]], pp[np.minimum(np.where(first)[0] + 1, len(pp) - 1)], 0.0)
    sel = first & (pp >= t) & ((pp - second) >= margin)
    va = np.where(role == 2)[0]
    npred = np.bincount(Ps[sel], minlength=len(role)); ntp = np.bincount(Ps[sel], weights=yy[sel], minlength=len(role))
    T, P, TP = tcount[va].astype(float), npred[va].astype(float), ntp[va]
    f = np.zeros(len(va))
    both0 = (T == 0) & (P == 0); f[both0] = 1.0
    ok = (T > 0) & (P > 0) & (TP > 0)
    pr, rc = TP[ok] / P[ok], TP[ok] / T[ok]; f[ok] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f.mean(), (P[T == 0] > 0).mean(), P.mean()



if __name__ == "__main__":
    D = {c: load(c) for c in FILES}
    names = list(D["India"]["names"])
    models = {}
    for c, d in D.items():
        tr = d["role"][d["P_s"]] == 1
        models[c] = lgb.train(PARAMS, lgb.Dataset(d["X"][tr], d["y"][tr], feature_name=names), num_boost_round=500)
    Xb = {c: D[c]["X"][D[c]["role"][D[c]["P_s"]] == 1] for c in D}; yb = {c: D[c]["y"][D[c]["role"][D[c]["P_s"]] == 1] for c in D}
    models["both"] = lgb.train(PARAMS, lgb.Dataset(np.vstack([Xb["India"], Xb["US"]]), np.concatenate([yb["India"], yb["US"]]), feature_name=names), num_boost_round=500)
    models["both"].save_model(f"{SP}/probe_lgb_both{'_dict' if 'INDIA_NPZ' in os.environ else ''}.txt")
    ts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for ev in D:
        for tr_c in ["India", "US", "both"]:
            p = models[tr_c].predict(D[ev]["X"], num_threads=10)
            res = [f05_counts(D[ev], p, t) for t in ts]
            best = max(range(len(ts)), key=lambda i: res[i][0])
            tag = "IN-DOMAIN" if tr_c == ev else ("LOCO" if tr_c != "both" else "pooled")
            print(f"eval={ev:5s} train={tr_c:5s} [{tag:9s}] " + " ".join(f"t={t}:{r[0]:.4f}" for t, r in zip(ts, res)) +
                  f" | best t={ts[best]} F={res[best][0]:.4f} singletonFP={res[best][1]*100:.1f}% pred/S1={res[best][2]:.2f}")
    # ambiguity margin on pooled model
    for ev in D:
        p = models["both"].predict(D[ev]["X"], num_threads=10)
        print(f"eval={ev} pooled, margin sweep @t=0.7: " + " ".join(f"m={m}:{f05_counts(D[ev], p, 0.7, m)[0]:.4f}" for m in [0.0, 0.1, 0.2, 0.3, 0.5]))
