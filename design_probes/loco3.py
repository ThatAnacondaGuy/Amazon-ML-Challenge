"""Design probe, round 2 of unseen-country remedies (after loco2 showed dropping features / monotone / self-training
on a reduced feature set all HURT).  Arms, evaluated LOCO both directions, full feature set throughout:
  A0  baseline (all features)
  A1  per-country percentile normalisation of scale-dependent features (unsupervised, within each country's own
      candidate pairs) — fixes score-scale shift without discarding signal
  A2  self-training on the target's unlabelled pairs, full features, 1 round
and three thresholds each: fixed 0.7, cardinality-matched (label-free), oracle-best (uses labels; reference only).
Premise for cardinality matching (measured): match-cardinality is identical across train countries
(mean 3.462 vs 3.461; singletons 5.57% vs 5.59%)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from scipy.stats import rankdata
from loco import load, f05_counts, PARAMS

D = {c: load(c) for c in ["India", "US"]}
names = list(D["India"]["names"])
SCALE = ["ret_score", "cos_name", "cos_addr", "ret_gap", "n_len_q", "n_len_s", "num_q", "num_s", "s_twins", "q_twin_cands"]
TS = np.round(np.arange(0.40, 0.96, 0.01), 2)


def pct_norm(X):
    X = X.copy()
    for n in SCALE:
        j = names.index(n)
        X[:, j] = (rankdata(X[:, j], method="average") / len(X)).astype(np.float32)
    return X


def evaluate(S, T, ps, pt):
    card_src = f05_counts(S, ps, 0.7)[2]
    cards = np.array([f05_counts(T, pt, t)[2] for t in TS]); t_match = TS[np.argmin(np.abs(cards - card_src))]
    fs = np.array([f05_counts(T, pt, t)[0] for t in TS])
    return f05_counts(T, pt, 0.7)[0], (t_match, f05_counts(T, pt, t_match)[0]), (TS[fs.argmax()], fs.max())


for src, tgt in [("US", "India"), ("India", "US")]:
    S, T = D[src], D[tgt]
    trm = S["role"][S["P_s"]] == 1
    rows = []
    for arm in ["A0 baseline", "A1 per-country percentile norm", "A2 self-train (full features)"]:
        XS, XT = (pct_norm(S["X"]), pct_norm(T["X"])) if arm.startswith("A1") else (S["X"], T["X"])
        m = lgb.train(PARAMS, lgb.Dataset(XS[trm], S["y"][trm]), num_boost_round=500)
        ps, pt = m.predict(XS, num_threads=10), m.predict(XT, num_threads=10)
        if arm.startswith("A2"):
            P_q = T["P_q"]; order = np.lexsort((-pt, P_q)); first = np.zeros(len(pt), bool)
            fq = np.concatenate([[True], P_q[order][1:] != P_q[order][:-1]]); first[order[fq]] = True
            pos = first & (pt >= 0.9); neg = (pt <= 0.1) | (~first & (pt <= 0.5)); sel = pos | neg
            m = lgb.train(PARAMS, lgb.Dataset(np.vstack([XS[trm], XT[sel]]), np.concatenate([S["y"][trm], pos[sel].astype(np.int8)]),
                                              weight=np.concatenate([np.ones(trm.sum()), np.full(sel.sum(), 0.5)])), num_boost_round=500)
            ps, pt = m.predict(XS, num_threads=10), m.predict(XT, num_threads=10)
        # in-domain cost of the arm (train on target's own train-role pairs, same transformation)
        trt = T["role"][T["P_s"]] == 1
        mi = lgb.train(PARAMS, lgb.Dataset(XT[trt], T["y"][trt]), num_boost_round=500)
        f_in = max(f05_counts(T, mi.predict(XT, num_threads=10), t)[0] for t in [0.6, 0.7, 0.8])
        rows.append((arm, evaluate(S, T, ps, pt), f_in))
    print(f"\n{src} -> {tgt}")
    for arm, (ffix, (tm, fm), (tb, fb)), f_in in rows:
        print(f"   {arm:34s} fixed0.7={ffix:.4f}  card-matched(t={tm:.2f})={fm:.4f}  oracle(t={tb:.2f})={fb:.4f}  | in-domain={f_in:.4f}")
