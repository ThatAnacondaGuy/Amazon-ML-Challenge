"""Design probe (throwaway): end-to-end on ONE country to estimate achievable macro F0.5 and test decision rules.

blocking (full country, reverse direction S2/S3 -> S1, top-K) -> ~30 classical features -> LightGBM -> decision rules
Usage: probe_model.py <country> <n_train_s1> <n_val_s1> <K>
"""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("NG", "4"); os.environ.setdefault("CONJ", "1")
import numpy as np, polars as pl, lightgbm as lgb
from multiprocessing import Pool
from scipy import sparse
from rapidfuzz import process, fuzz, distance
from load import load_source, load_gt
from probe_lib import name_tokens, addr_tokens, skel, NB
import probe_block2 as pb

COUNTRY = sys.argv[1]; NTR = int(sys.argv[2]); NVA = int(sys.argv[3]); K = int(sys.argv[4]); CAP = 2000; WA = 0.5
SP = os.path.dirname(__file__)
INDIC = re.compile(r"[ऀ-෿]")
log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


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
        m = sparse.csr_matrix((np.ones(len(idx), np.float32), idx.astype(np.int32), indptr.astype(np.int64)), shape=(len(rows), NB)); m.sum_duplicates()
        mats.append(m)
    return mats


def text_feats(names, addrs):
    """normalized strings used by pairwise features"""
    out = []
    for n, a in zip(names, addrs):
        toks, core = name_tokens(n)
        words, nums = addr_tokens(a)
        out.append((" ".join(toks), " ".join(core), "".join(core), frozenset(skel(t) for t in core), frozenset(core),
                    " ".join(words), frozenset(words), tuple(nums), "".join(t[0] for t in core if t)))
    return out


def main():
    t0 = time.time()
    s1 = load_source("train", 1).filter(pl.col("country") == COUNTRY)
    DROP = float(os.environ.get("DROP", "0"))
    if DROP > 0:
        s1 = s1.sample(fraction=1 - DROP, seed=123, shuffle=False)
    s23 = pl.concat([load_source("train", 2), load_source("train", 3)]).filter(pl.col("country") == COUNTRY)
    gt = load_gt()
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
    s1_ids = s1["entity_id"].to_list(); s1_pos = {e: i for i, e in enumerate(s1_ids)}
    q_ids = s23["entity_id"].to_list(); q_pos = {e: i for i, e in enumerate(q_ids)}
    truth_q = np.full(len(q_ids), -1, np.int64)
    for a, b in pairs.filter(pl.col("s1").is_in(s1_ids)).select("s1", "rid").iter_rows():
        truth_q[q_pos[b]] = s1_pos[a]
    N = len(s1_ids)
    log(f"{COUNTRY}: S1={N:,} queries={len(q_ids):,}")
    # ---------------- index
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
    log("index built")
    rng = np.random.default_rng(0)
    perm = rng.permutation(N); tr_s1 = perm[:NTR]; va_s1 = perm[NTR:NTR + NVA]
    role = np.zeros(N, np.int8); role[tr_s1] = 1; role[va_s1] = 2
    # ---------------- retrieval over ALL queries in chunks
    top_i = np.full((len(q_ids), K), -1, np.int32); top_s = np.zeros((len(q_ids), K), np.float32)
    qn_all = s23["business_name"].to_list(); qa_all = s23["business_address"].to_list()
    Qn_parts, Qa_parts, rel_parts = [], [], []
    CH = 400000
    for c0 in range(0, len(q_ids), CH):
        Qn, Qa = build(qn_all[c0:c0 + CH], qa_all[c0:c0 + CH])
        Qn, Qa = pb.weight(Qn, idf_n), pb.weight(Qa, idf_a)
        Wq = sparse.hstack([Qn * np.sqrt(1 - WA), Qa * np.sqrt(WA)]).tocsr()
        oi = -np.ones((Wq.shape[0], K), np.int32); os_ = np.zeros((Wq.shape[0], K), np.float32); work = np.zeros(Wq.shape[0], np.int64)
        pb.search(Wq.indptr.astype(np.int64), Wq.indices.astype(np.int64), Wq.data.astype(np.float32), ip, ii, idd, N, K, 40, oi, os_, work)
        top_i[c0:c0 + CH] = oi; top_s[c0:c0 + CH] = os_
        relc = np.where((role[np.clip(oi, 0, None)] * (oi >= 0)).max(1) > 0)[0]
        Qn_parts.append(Qn[relc]); Qa_parts.append(Qa[relc]); rel_parts.append(relc + c0)
        log(f"  retrieved {min(c0 + CH, len(q_ids)):,}")
    Qn = sparse.vstack(Qn_parts).tocsr(); Qa = sparse.vstack(Qa_parts).tocsr(); rel_all = np.concatenate(rel_parts); del Qn_parts, Qa_parts
    qrow = np.full(len(q_ids), -1, np.int64); qrow[rel_all] = np.arange(len(rel_all))
    m = truth_q >= 0
    hit = (top_i[m] == truth_q[m][:, None]).any(1)
    log(f"pair recall@{K} over all matched queries: {hit.mean()*100:.2f}%   total candidate pairs: {(top_i >= 0).sum():,}")
    # ---------------- sample S1 train / val entities (group split by S1)
    rel_q = np.where((role[np.clip(top_i, 0, None)] * (top_i >= 0)).max(1) > 0)[0]
    # also include queries whose TRUE s1 is in the sampled sets (needed to count misses in recall)
    log(f"relevant queries: {len(rel_q):,}")
    P_q = np.repeat(rel_q, K); P_s = top_i[rel_q].ravel(); P_sc = top_s[rel_q].ravel(); P_rank = np.tile(np.arange(K), len(rel_q))
    ok = P_s >= 0; P_q, P_s, P_sc, P_rank = P_q[ok], P_s[ok], P_sc[ok], P_rank[ok]
    y = (truth_q[P_q] == P_s).astype(np.int8)
    log(f"pairs to featurize: {len(P_q):,}  positives: {y.sum():,}")
    # ---------------- features
    cos_n = np.asarray(Qn[qrow[P_q]].multiply(Wn[P_s]).sum(1)).ravel(); cos_a = np.asarray(Qa[qrow[P_q]].multiply(Wa[P_s]).sum(1)).ravel()
    uq = np.unique(P_q); us = np.unique(P_s)
    s1n = s1["business_name"].to_list(); s1a = s1["business_address"].to_list()
    TQ = dict(zip(uq.tolist(), text_feats([qn_all[i] for i in uq], [qa_all[i] for i in uq])))
    TS = dict(zip(us.tolist(), text_feats([s1n[i] for i in us], [s1a[i] for i in us])))
    # name frequency (twins) among S1 index
    core_all = [" ".join(name_tokens(n)[1]) for n in s1n]
    from collections import Counter
    cnt = Counter(core_all)
    tq = [TQ[i] for i in P_q.tolist()]; ts = [TS[i] for i in P_s.tolist()]
    W = -1
    F = {}
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
    # query-level context: how many of this query's candidates look like name twins
    tw = (F["n_tset_core"] >= 95).astype(np.int32)
    qt = np.zeros(len(q_ids), np.int32); np.add.at(qt, P_q, tw); F["q_twin_cands"] = qt[P_q].astype(np.float32)
    X = np.column_stack([np.asarray(F[k], np.float32) for k in F]); names = list(F)
    log(f"features: {len(names)}")
    # ---------------- train / eval
    # train on pairs whose S1 is a TRAIN entity ; evaluate the whole decision pipeline for VAL entities
    trm = role[P_s] == 1
    dtr = lgb.Dataset(X[trm], y[trm], feature_name=names, free_raw_data=False)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10)
    bst = lgb.train(params, dtr, num_boost_round=500)
    p = bst.predict(X, num_threads=10)
    imp = sorted(zip(bst.feature_importance("gain"), names), reverse=True)
    log("top features (gain):", [(n, round(g / sum(x for x, _ in imp) * 100, 1)) for g, n in imp[:14]])
    # decision rules evaluated on VAL S1 entities; competition uses scores of ALL candidates of each query
    va_set = set(va_s1.tolist())
    true_sets = {s: set() for s in va_s1.tolist()}
    for qq in np.where(np.isin(truth_q, va_s1))[0].tolist():
        true_sets[truth_q[qq]].add(qq)

    def f05(pred):
        tot = 0.0
        for s, T in true_sets.items():
            Pp = pred.get(s, set())
            if not T and not Pp: tot += 1; continue
            if not T or not Pp: continue
            tp = len(T & Pp)
            if tp == 0: continue
            pr, rc = tp / len(Pp), tp / len(T)
            tot += 1.25 * pr * rc / (0.25 * pr + rc)
        return tot / len(true_sets)

    # rule A: independent threshold
    # rule B: each query assigned to argmax S1 only, if p >= t
    order = np.lexsort((-p, P_q)); Pq_o, Ps_o, p_o = P_q[order], P_s[order], p[order]
    first = np.concatenate([[True], Pq_o[1:] != Pq_o[:-1]])
    res = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for rule in ["A", "B"]:
            sel = (p_o >= t) & (first if rule == "B" else True)
            pred = {}
            for qq, ss in zip(Pq_o[sel].tolist(), Ps_o[sel].tolist()):
                if ss in va_set: pred.setdefault(ss, set()).add(qq)
            res[(rule, t)] = f05(pred)
    for rule in ["A", "B"]:
        log(f"rule {rule}: " + "  ".join(f"t={t}:{res[(rule, t)]:.4f}" for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    # oracle: perfect matcher on our candidate set (blocking ceiling)
    pred = {}
    for qq, ss, yy in zip(P_q.tolist(), P_s.tolist(), y.tolist()):
        if yy and ss in va_set: pred.setdefault(ss, set()).add(qq)
    log(f"oracle F0.5 given candidates (blocking ceiling): {f05(pred):.4f};  val S1={len(true_sets):,} singletons={sum(1 for T in true_sets.values() if not T)/len(true_sets)*100:.1f}%")
    tcount = np.bincount(truth_q[truth_q >= 0], minlength=N)
    np.savez_compressed(f"{SP}/probe_model_{COUNTRY}_drop{DROP}.npz", X=X, y=y, p=p, P_q=P_q, P_s=P_s, role=role, names=np.array(names), tcount=tcount, top_s=top_s[P_q])


if __name__ == "__main__":
    main()
