import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from load import load_source, load_gt
from probe_lib import name_tokens
SP = os.path.dirname(os.path.abspath(__file__)); C = sys.argv[1]; T = float(sys.argv[2]); F = sys.argv[3]
d = np.load(f"{SP}/{F}", allow_pickle=True)
X, y, p, P_q, P_s, role, names = d["X"], d["y"], d["p"], d["P_q"], d["P_s"], d["role"], list(d["names"])
s1 = load_source("train", 1).filter(pl.col("country") == C)
s23 = pl.concat([load_source("train", 2), load_source("train", 3)]).filter(pl.col("country") == C)
gt = load_gt()
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
s1_pos = {e: i for i, e in enumerate(s1["entity_id"].to_list())}; q_pos = {e: i for i, e in enumerate(s23["entity_id"].to_list())}
truth_q = np.full(s23.height, -1, np.int64)
for a, b in pairs.filter(pl.col("s1").is_in(list(s1_pos))).select("s1", "rid").iter_rows(): truth_q[q_pos[b]] = s1_pos[a]
qn, qa = s23["business_name"].to_list(), s23["business_address"].to_list(); sn, sa = s1["business_name"].to_list(), s1["business_address"].to_list()
va = np.where(role == 2)[0]; vaset = set(va.tolist())
# rule B at threshold T
order = np.lexsort((-p, P_q)); Pq, Ps, pp, yy, XX = P_q[order], P_s[order], p[order], y[order], X[order]
first = np.concatenate([[True], Pq[1:] != Pq[:-1]])
sel = first & (pp >= T)
assigned = {}
for q, s, pr in zip(Pq[sel].tolist(), Ps[sel].tolist(), pp[sel].tolist()): assigned[q] = (s, pr)
T_sets = {s: set() for s in va.tolist()}
for q in np.where(np.isin(truth_q, va))[0].tolist(): T_sets[truth_q[q]].add(q)
P_sets = {s: set() for s in va.tolist()}
for q, (s, pr) in assigned.items():
    if s in vaset: P_sets[s].add(q)
tp = sum(len(T_sets[s] & P_sets[s]) for s in va.tolist()); fp = sum(len(P_sets[s] - T_sets[s]) for s in va.tolist()); fn = sum(len(T_sets[s] - P_sets[s]) for s in va.tolist())
print(f"[{C} t={T}] val S1={len(va):,}  micro: TP={tp:,} FP={fp:,} FN={fn:,}  precision={tp/(tp+fp):.4f} recall={tp/(tp+fn):.4f}")
# per-entity F0.5 buckets
def f05(Tt, Pp):
    if not Tt and not Pp: return 1.0
    if not Tt or not Pp: return 0.0
    t = len(Tt & Pp)
    if t == 0: return 0.0
    pr, rc = t / len(Pp), t / len(Tt); return 1.25 * pr * rc / (0.25 * pr + rc)
fs = np.array([f05(T_sets[s], P_sets[s]) for s in va.tolist()]); sing = np.array([len(T_sets[s]) == 0 for s in va.tolist()])
print(f"  macro F0.5={fs.mean():.4f}  singletons: n={sing.sum()} mean F={fs[sing].mean():.3f} (lost {(1-fs[sing]).sum()/len(va)*100:.2f} pts) | non-singletons mean F={fs[~sing].mean():.4f} (lost {(1-fs[~sing]).sum()/len(va)*100:.2f} pts)")
# FP categories
cand_of = {}
for q, s in zip(P_q.tolist(), P_s.tolist()): cand_of.setdefault(q, set()).add(s)
fps = [(q, s) for s in va.tolist() for q in P_sets[s] - T_sets[s]]
cat = {"distractor(query has no S1)": 0, "query belongs to another S1": 0}
for q, s in fps: cat["distractor(query has no S1)" if truth_q[q] < 0 else "query belongs to another S1"] += 1
print("  FP breakdown:", cat)
fns = [(q, s) for s in va.tolist() for q in T_sets[s] - P_sets[s]]
fcat = {"blocking miss": 0, "below threshold": 0, "lost argmax to another S1": 0}
for q, s in fns:
    if s not in cand_of.get(q, ()): fcat["blocking miss"] += 1
    elif q in assigned: fcat["lost argmax to another S1"] += 1
    else: fcat["below threshold"] += 1
print("  FN breakdown:", fcat)
ind = re.compile(r"[ऀ-෿]")
print("  FN with indic query name: %.1f%%, empty query addr: %.1f%% | FP with empty query addr: %.1f%%" % (
    np.mean([bool(ind.search(qn[q])) for q, s in fns]) * 100, np.mean([not qa[q].strip() for q, s in fns]) * 100, np.mean([not qa[q].strip() for q, s in fps]) * 100))
rng = np.random.default_rng(1)
print("\n  --- sample FPs (query -> wrongly assigned S1 ; [true S1 if any])")
for i in rng.choice(len(fps), 10, replace=False):
    q, s = fps[i]; t = truth_q[q]
    print(f"   p={assigned[q][1]:.2f} Q: {qn[q][:40]} | {qa[q][:55]}\n          S: {sn[s][:40]} | {sa[s][:55]}" + (f"\n          TRUE: {sn[t][:40]} | {sa[t][:55]}" if t >= 0 else "  [distractor]"))
print("\n  --- sample FNs below threshold (query -> its true S1)")
bt = [(q, s) for q, s in fns if s in cand_of.get(q, ()) and q not in assigned]
for i in rng.choice(len(bt), 10, replace=False):
    q, s = bt[i]; k = np.where((P_q == q) & (P_s == s))[0][0]
    print(f"   p={p[k]:.2f} Q: {qn[q][:40]} | {qa[q][:55]}\n          S: {sn[s][:40]} | {sa[s][:55]}")
