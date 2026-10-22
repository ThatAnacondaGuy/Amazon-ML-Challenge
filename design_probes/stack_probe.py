"""Design probe: (1) does a stage-2 model with collective (cross-candidate / cross-query) features help?
(2) does per-S1 expected-F0.5 subset selection beat a global threshold?  Uses saved probe npz (one country)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
SP = os.path.dirname(os.path.abspath(__file__))
F = sys.argv[1]
d = dict(np.load(f"{SP}/{F}", allow_pickle=True))
if "tcount" not in d:
    from probe_core import load_country
    s1, s23, truth_q = load_country("train", "India")
    d["tcount"] = np.bincount(truth_q[truth_q >= 0], minlength=s1.height)
X, y, P_q, P_s, role, tcount = d["X"], d["y"], d["P_q"], d["P_s"], d["role"], d["tcount"]
names = list(d["names"])
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=7)
ent_role = role[P_s]
sampled = ent_role > 0
# ---- stage-1 out-of-fold predictions (5 folds grouped by S1 entity over train+val entities); competitor pairs get fold-avg
fold_of_ent = np.random.default_rng(3).integers(0, 5, len(role))
fold = fold_of_ent[P_s]
p1 = np.zeros(len(y))
cnt = np.zeros(len(y))
for k in range(5):
    tr = sampled & (fold != k) & (ent_role == 1)          # learn only from TRAIN-role entities
    m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), num_boost_round=400)
    te = ~(sampled & (fold != k) & (ent_role == 1))       # predict everything not used for training this fold
    p1[te] += m.predict(X[te], num_threads=10); cnt[te] += 1
p1 /= np.maximum(cnt, 1)
# ---- collective features
nq = P_q.max() + 1; ns = len(role)
order = np.lexsort((-p1, P_q)); inv = np.empty_like(order); inv[order] = np.arange(len(order))
Pq_o, p_o = P_q[order], p1[order]
first = np.concatenate([[True], Pq_o[1:] != Pq_o[:-1]])
grp_start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
rank_p = (np.arange(len(order)) - grp_start)[inv]                        # rank of this S1 among the query's candidates by p1
qbest = np.zeros(nq); np.maximum.at(qbest, P_q, p1)
qsecond = np.zeros(nq)
is_best = rank_p == 0
tmp = p1.copy(); tmp[is_best] = -1; np.maximum.at(qsecond, P_q, tmp); qsecond = np.maximum(qsecond, 0)
margin = np.where(is_best, p1 - qsecond[P_q], p1 - qbest[P_q])          # >0 only for the query's winner
# per S1 support: how many OTHER queries pick this S1 as their argmax with high confidence
win = is_best & (p1 >= 0.5)
s_support = np.bincount(P_s[win], weights=p1[win], minlength=ns)
s_nstrong = np.bincount(P_s[is_best & (p1 >= 0.9)], minlength=ns)
own = np.where(win, p1, 0.0)
own_strong = (is_best & (p1 >= 0.9)).astype(float)
C = np.column_stack([p1, rank_p, margin, qbest[P_q], s_support[P_s] - own, s_nstrong[P_s] - own_strong])
cnames = ["p1", "p1_rank", "p1_margin", "q_best_p1", "s_support_others", "s_nstrong_others"]
X2 = np.column_stack([X, C]); n2 = names + cnames
tr2 = ent_role == 1
m2 = lgb.train(PARAMS, lgb.Dataset(X2[tr2], y[tr2], feature_name=n2), num_boost_round=500)
p2 = m2.predict(X2, num_threads=10)


def evaluate(p, t, rule="B"):
    order = np.lexsort((-p, P_q)); Pq, Ps, pp, yy = P_q[order], P_s[order], p[order], y[order]
    first = np.concatenate([[True], Pq[1:] != Pq[:-1]])
    sel = first & (pp >= t)
    va = np.where(role == 2)[0]
    npred = np.bincount(Ps[sel], minlength=len(role)); ntp = np.bincount(Ps[sel], weights=yy[sel], minlength=len(role))
    T, P, TP = tcount[va].astype(float), npred[va].astype(float), ntp[va]
    f = np.zeros(len(va)); f[(T == 0) & (P == 0)] = 1
    ok = (T > 0) & (P > 0) & (TP > 0); pr, rc = TP[ok] / P[ok], TP[ok] / T[ok]; f[ok] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f.mean()


def expected_f_select(p, cal):
    """per S1 (val): candidates = queries whose argmax is this S1; choose top-j by calibrated p maximizing
    E[F0.5] ~= 1.25*sum_top_j(q) / (0.25*(sum_all_q + miss) + j), j=0 scores P(no true match)=prod(1-q)."""
    order = np.lexsort((-p, P_q)); Pq, Ps, pp, yy = P_q[order], P_s[order], p[order], y[order]
    first = np.concatenate([[True], Pq[1:] != Pq[:-1]])
    Pq, Ps, pp, yy = Pq[first], Ps[first], pp[first], yy[first]
    q = cal.predict(pp)
    va = np.where(role == 2)[0]; vset = np.zeros(len(role), bool); vset[va] = True
    keep = vset[Ps]; Ps, q, yy = Ps[keep], q[keep], yy[keep]
    o = np.lexsort((-q, Ps)); Ps, q, yy = Ps[o], q[o], yy[o]
    bounds = np.flatnonzero(np.concatenate([[True], Ps[1:] != Ps[:-1], [True]]))
    TPs = np.zeros(len(role)); Ps_n = np.zeros(len(role))
    MISS = 0.1   # expected true matches outside candidates (blocking miss prior)
    for a, b in zip(bounds[:-1], bounds[1:]):
        qs = q[a:b]; tot = qs.sum() + MISS
        best_j, best_v = 0, np.prod(1 - qs) * np.exp(-MISS)
        cs = np.cumsum(qs)
        for j in range(1, len(qs) + 1):
            v = 1.25 * cs[j - 1] / (0.25 * tot + j)
            if v > best_v: best_j, best_v = j, v
        s = Ps[a]; Ps_n[s] = best_j; TPs[s] = yy[a:a + best_j].sum()
    T, P, TP = tcount[va].astype(float), Ps_n[va], TPs[va]
    f = np.zeros(len(va)); f[(T == 0) & (P == 0)] = 1
    ok = (T > 0) & (P > 0) & (TP > 0); pr, rc = TP[ok] / P[ok], TP[ok] / T[ok]; f[ok] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f.mean()


ts = [0.5, 0.6, 0.7, 0.8, 0.9]
for lab, p in [("stage-1 (OOF)", p1), ("stage-2 collective", p2)]:
    r = [evaluate(p, t) for t in ts]
    print(f"{lab:20s} rule B: " + " ".join(f"t={t}:{v:.4f}" for t, v in zip(ts, r)) + f" | best {max(r):.4f}")
    trm = ent_role == 1
    cal = IsotonicRegression(out_of_bounds="clip").fit(p[trm], y[trm])
    print(f"{lab:20s} expected-F0.5 per-S1 selection (isotonic-calibrated): {expected_f_select(p, cal):.4f}")
imp = sorted(zip(m2.feature_importance('gain'), n2), reverse=True)[:10]
print("stage-2 top gain:", [(n, round(g / sum(m2.feature_importance('gain')) * 100, 1)) for g, n in imp])
