"""Shared design-probe core (throwaway): per-country blocking + probe features, usable on train or test splits."""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NG", "4"); os.environ.setdefault("CONJ", "1")
import numpy as np, polars as pl
from multiprocessing import Pool
from scipy import sparse
from collections import Counter
from rapidfuzz import process, fuzz, distance
from load import load_source, load_gt
from probe_lib import name_tokens, addr_tokens, skel, NB
import probe_block2 as pb

INDIC = re.compile(r"[ऀ-෿]")
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
CAP, WA = 2000, 0.5


def feat_arrays(rows):
    fs = pb.featurize(rows)
    out = []
    for part in (0, 1):
        lens = np.array([len(f[part]) for f in fs], np.int64)
        idx = np.fromiter((x for f in fs for x in f[part]), np.int64, count=int(lens.sum()))
        out.append((lens, idx))
    return out


def build(names, addrs):
    rows = list(zip(names, addrs))
    chunks = [rows[i:i + 25000] for i in range(0, len(rows), 25000)]
    with Pool(10) as pool:
        res = pool.map(feat_arrays, chunks)
    mats = []
    for part in (0, 1):
        lens = np.concatenate([r[part][0] for r in res]); idx = np.concatenate([r[part][1] for r in res])
        indptr = np.concatenate([[0], np.cumsum(lens)])
        m = sparse.csr_matrix((np.ones(len(idx), np.float32), idx.astype(np.int32), indptr.astype(np.int64)), shape=(len(rows), NB))
        m.sum_duplicates(); mats.append(m)
    return mats


def text_feats(names, addrs):
    out = []
    for n, a in zip(names, addrs):
        toks, core = name_tokens(n)
        words, nums = addr_tokens(a)
        out.append((" ".join(toks), " ".join(core), "".join(core), frozenset(skel(t) for t in core), frozenset(core),
                    " ".join(words), frozenset(words), tuple(nums), "".join(t[0] for t in core if t)))
    return out


def load_country(split, country, drop=0.0):
    s1 = load_source(split, 1).filter(pl.col("country") == country)
    if drop > 0:
        s1 = s1.sample(fraction=1 - drop, seed=123, shuffle=False)
    s23 = pl.concat([load_source(split, 2), load_source(split, 3)]).filter(pl.col("country") == country)
    truth_q = np.full(s23.height, -1, np.int64)
    if split == "train":
        gt = load_gt()
        pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                   .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
        s1_pos = {e: i for i, e in enumerate(s1["entity_id"].to_list())}; q_pos = {e: i for i, e in enumerate(s23["entity_id"].to_list())}
        for a, b in pairs.filter(pl.col("s1").is_in(list(s1_pos))).select("s1", "rid").iter_rows():
            truth_q[q_pos[b]] = s1_pos[a]
    return s1, s23, truth_q


def run_blocking(s1, s23, role, K):
    """Full-country retrieval; keeps sparse query rows only for queries relevant to entities with role>0."""
    N = s1.height
    In, Ia = build(s1["business_name"].to_list(), s1["business_address"].to_list())
    df_n = np.bincount(In.indices, minlength=NB); df_a = np.bincount(Ia.indices, minlength=NB)
    idf_n = (np.log((N + 1) / (df_n + 1)) + 1).astype(np.float32); idf_a = (np.log((N + 1) / (df_a + 1)) + 1).astype(np.float32)
    Wn, Wa = pb.weight(In, idf_n), pb.weight(Ia, idf_a); del In, Ia
    Wi = sparse.hstack([Wn * np.sqrt(1 - WA), Wa * np.sqrt(WA)]).tocsr()
    inv = Wi.tocsc(); inv.sort_indices()
    keep = np.bincount(Wi.indices, minlength=Wi.shape[1]) <= CAP
    lens = np.diff(inv.indptr) * keep; ip = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    mask = np.repeat(keep, np.diff(inv.indptr)); ii = inv.indices[mask].astype(np.int32); idd = inv.data[mask].astype(np.float32)
    del inv, Wi
    nq = s23.height
    top_i = np.full((nq, K), -1, np.int32); top_s = np.zeros((nq, K), np.float32)
    qn_all = s23["business_name"].to_list(); qa_all = s23["business_address"].to_list()
    Qn_parts, Qa_parts, rel_parts = [], [], []
    CH = 400000; t = time.time()
    for c0 in range(0, nq, CH):
        Qn, Qa = build(qn_all[c0:c0 + CH], qa_all[c0:c0 + CH])
        Qn, Qa = pb.weight(Qn, idf_n), pb.weight(Qa, idf_a)
        Wq = sparse.hstack([Qn * np.sqrt(1 - WA), Qa * np.sqrt(WA)]).tocsr()
        oi = -np.ones((Wq.shape[0], K), np.int32); os_ = np.zeros((Wq.shape[0], K), np.float32); work = np.zeros(Wq.shape[0], np.int64)
        pb.search(Wq.indptr.astype(np.int64), Wq.indices.astype(np.int64), Wq.data.astype(np.float32), ip, ii, idd, N, K, 40, oi, os_, work)
        top_i[c0:c0 + CH] = oi; top_s[c0:c0 + CH] = os_
        relc = np.where((role[np.clip(oi, 0, None)] * (oi >= 0)).max(1) > 0)[0]
        Qn_parts.append(Qn[relc]); Qa_parts.append(Qa[relc]); rel_parts.append(relc + c0)
    log(f"retrieval: {nq:,} queries in {time.time()-t:.0f}s ({nq/(time.time()-t):,.0f} q/s)")
    Qn = sparse.vstack(Qn_parts).tocsr(); Qa = sparse.vstack(Qa_parts).tocsr(); rel_all = np.concatenate(rel_parts)
    qrow = np.full(nq, -1, np.int64); qrow[rel_all] = np.arange(len(rel_all))
    return dict(top_i=top_i, top_s=top_s, Qn=Qn, Qa=Qa, qrow=qrow, Wn=Wn, Wa=Wa, rel=rel_all)


def make_features(s1, s23, B, K):
    top_i, top_s, qrow = B["top_i"], B["top_s"], B["qrow"]
    rel_q = B["rel"]
    P_q = np.repeat(rel_q, K); P_s = top_i[rel_q].ravel(); P_sc = top_s[rel_q].ravel(); P_rank = np.tile(np.arange(K), len(rel_q))
    ok = P_s >= 0; P_q, P_s, P_sc, P_rank = P_q[ok], P_s[ok], P_sc[ok], P_rank[ok]
    cos_n = np.asarray(B["Qn"][qrow[P_q]].multiply(B["Wn"][P_s]).sum(1)).ravel()
    cos_a = np.asarray(B["Qa"][qrow[P_q]].multiply(B["Wa"][P_s]).sum(1)).ravel()
    qn_all = s23["business_name"].to_list(); qa_all = s23["business_address"].to_list()
    s1n = s1["business_name"].to_list(); s1a = s1["business_address"].to_list()
    uq = np.unique(P_q); us = np.unique(P_s)
    TQ = dict(zip(uq.tolist(), text_feats([qn_all[i] for i in uq], [qa_all[i] for i in uq])))
    TS = dict(zip(us.tolist(), text_feats([s1n[i] for i in us], [s1a[i] for i in us])))
    core_all = [" ".join(name_tokens(n)[1]) for n in s1n]; cnt = Counter(core_all)
    tq = [TQ[i] for i in P_q.tolist()]; ts = [TS[i] for i in P_s.tolist()]
    W = -1; F = {}
    F["ret_score"] = P_sc; F["ret_rank"] = P_rank.astype(np.float32); F["cos_name"] = cos_n; F["cos_addr"] = cos_a
    top1 = top_s[P_q, 0]; F["ret_ratio"] = P_sc / np.maximum(top1, 1e-6); F["ret_gap"] = top1 - P_sc
    F["n_ratio"] = process.cpdist([a[0] for a in tq], [b[0] for b in ts], scorer=fuzz.ratio, workers=W)
    F["n_tsort"] = process.cpdist([a[0] for a in tq], [b[0] for b in ts], scorer=fuzz.token_sort_ratio, workers=W)
    F["n_tset_core"] = process.cpdist([a[1] for a in tq], [b[1] for b in ts], scorer=fuzz.token_set_ratio, workers=W)
    F["n_jw_core"] = process.cpdist([a[1] for a in tq], [b[1] for b in ts], scorer=distance.JaroWinkler.similarity, workers=W)
    F["n_ns_ratio"] = process.cpdist([a[2] for a in tq], [b[2] for b in ts], scorer=fuzz.ratio, workers=W)
    F["n_ns_partial"] = process.cpdist([a[2] for a in tq], [b[2] for b in ts], scorer=fuzz.partial_ratio, workers=W)
    F["n_skel_jacc"] = np.array([len(a[3] & b[3]) / max(1, len(a[3] | b[3])) for a, b in zip(tq, ts)], np.float32)
    F["n_tok_jacc"] = np.array([len(a[4] & b[4]) / max(1, len(a[4] | b[4])) for a, b in zip(tq, ts)], np.float32)
    F["n_initials"] = np.array([float(len(a[2]) <= 5 and len(b[8]) >= 2 and a[2] == b[8]) for a, b in zip(tq, ts)], np.float32)
    F["n_len_q"] = np.array([len(a[4]) for a in tq], np.float32); F["n_len_s"] = np.array([len(b[4]) for b in ts], np.float32)
    F["s_twins"] = np.array([cnt[core_all[s]] for s in P_s.tolist()], np.float32)
    F["a_tset"] = process.cpdist([a[5] for a in tq], [b[5] for b in ts], scorer=fuzz.token_set_ratio, workers=W)
    F["a_jacc"] = np.array([len(a[6] & b[6]) / max(1, len(a[6] | b[6])) for a, b in zip(tq, ts)], np.float32)
    F["num_any"] = np.array([float(bool(set(a[7]) & set(b[7]))) for a, b in zip(tq, ts)], np.float32)
    F["num_first"] = np.array([float(bool(a[7]) and bool(b[7]) and a[7][0] == b[7][0]) for a, b in zip(tq, ts)], np.float32)
    F["num_fuzzy"] = np.array([float(any(x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x)) for x in a[7] for y in b[7])) for a, b in zip(tq, ts)], np.float32)
    F["num_q"] = np.array([len(a[7]) for a in tq], np.float32); F["num_s"] = np.array([len(b[7]) for b in ts], np.float32)
    F["q_addr_empty"] = np.array([float(not a[5]) for a in tq], np.float32)
    F["q_indic"] = np.array([float(bool(INDIC.search(qn_all[q]))) for q in P_q.tolist()], np.float32)
    tw = (F["n_tset_core"] >= 95).astype(np.int32)
    qt = np.zeros(s23.height, np.int32); np.add.at(qt, P_q, tw); F["q_twin_cands"] = qt[P_q].astype(np.float32)
    X = np.column_stack([np.asarray(F[k], np.float32) for k in F])
    return X, list(F), P_q, P_s


def decide(p, P_q, P_s, t):
    """rule B: each query -> argmax S1 if p>=t. returns dict s1 -> set(queries)"""
    order = np.lexsort((-p, P_q)); Pq, Ps, pp = P_q[order], P_s[order], p[order]
    first = np.concatenate([[True], Pq[1:] != Pq[:-1]])
    sel = first & (pp >= t)
    pred = {}
    for q, s in zip(Pq[sel].tolist(), Ps[sel].tolist()):
        pred.setdefault(s, set()).add(q)
    return pred


def f05_macro(true_sets, pred):
    tot = 0.0
    for s, T in true_sets.items():
        Pp = pred.get(s, set())
        if not T and not Pp: tot += 1; continue
        if not T or not Pp: continue
        tp = len(T & Pp)
        if tp == 0: continue
        pr, rc = tp / len(Pp), tp / len(T); tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(true_sets)
