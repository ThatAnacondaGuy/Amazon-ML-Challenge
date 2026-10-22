"""Stage-2: out-of-fold stage-1 scores -> collective + sibling-agreement features -> stage-2 LightGBM ->
validation (exact macro F0.5) -> test inference with cardinality-matched thresholds -> submission files.

Requires the caches written by ber.pipeline (work/train_<c>.parquet, work/test_<c>.parquet, ...).
Usage: python -m ber.stage2 --data-dir ../../student_resource/dataset --work-dir work --out-dir ../../output
"""
import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import blocking, decide, io

log = logging.getLogger("ber")
META = ("y", "q_id", "s1_id", "qi", "si", "country")
P1 = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=7, deterministic=True,
          force_row_wise=True)
P2 = dict(P1, learning_rate=0.05)


def fold_of(ids):
    return np.array([zlib.crc32(x.encode()) % 5 for x in ids], np.int8)


def countries(work, split):
    return sorted(f[len(split) + 1:-8] for f in os.listdir(work)
                  if f.startswith(split + "_") and f.endswith(".parquet") and f.count("_") == 1)


def norm_frame(work, tag, pool, s1c, s23c, td, which):
    path = os.path.join(work, f"{tag}_{which}.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    src = s23c if which == "qn" else s1c
    df = blocking.normalize_only(pool, src["business_name"].to_list(), src["business_address"].to_list(), td)
    df.write_parquet(path)
    return df


def collective(D, p1, qn):
    """Per-partition collective features. D: partition pairs (qi, si); p1 aligned; qn normalized queries."""
    q = D["qi"].to_numpy().astype(np.int64); s = D["si"].to_numpy().astype(np.int64)
    sc = D["ret_score"].to_numpy()
    n = len(q)
    order = np.lexsort((s, -sc, -p1, q))
    qo = q[order]
    first = np.ones(n, bool); first[1:] = qo[1:] != qo[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.empty(n, np.int32); rank[order] = np.arange(n) - start
    nq = q.max() + 1
    qbest = np.zeros(nq); np.maximum.at(qbest, q, p1)
    tmp = np.where(rank == 0, -1.0, p1); qsec = np.zeros(nq); np.maximum.at(qsec, q, tmp)
    margin = np.where(rank == 0, p1 - qsec[q], p1 - qbest[q])
    qstrong = np.bincount(q, weights=(p1 >= 0.5), minlength=nq)
    win = (rank == 0) & (p1 >= 0.5)
    ns = s.max() + 1
    s_sup = np.bincount(s[win], weights=p1[win], minlength=ns)
    s_nwin = np.bincount(s[win], minlength=ns)
    s_nstr = np.bincount(s[win & (p1 >= 0.9)], minlength=ns)
    own = np.where(win, p1, 0.0)
    F = {"p1": p1, "p1_rank": rank, "p1_margin": margin, "q_best_p1": qbest[q], "q_nstrong": qstrong[q],
         "s_support_oth": s_sup[s] - own, "s_nwin_oth": s_nwin[s] - win, "s_nstrong_oth": s_nstr[s] - (win & (p1 >= 0.9))}
    # sibling agreement: compare q with the top-3 other confident winners of the same S1
    W = pl.DataFrame({"s": s[win], "sib": q[win], "ps": p1[win]}).sort(["s", "ps"], descending=[False, True])
    W = W.with_columns(pl.int_range(pl.len()).over("s").alias("r")).filter(pl.col("r") < 4)
    X = pl.DataFrame({"pid": np.arange(n), "q": q, "s": s}).join(W, on="s", how="inner").filter(pl.col("sib") != pl.col("q"))
    sib_name = np.full(n, -1.0, np.float32); sib_addr = np.full(n, -1.0, np.float32); sib_hn = np.full(n, -1.0, np.float32)
    sib_cnt = np.zeros(n, np.float32)
    CH = 4_000_000
    for c0 in range(0, X.height, CH):
        Xc = X.slice(c0, CH)
        a = qn[Xc["q"].to_numpy()]; b = qn[Xc["sib"].to_numpy()]
        nm = process.cpdist(a["n_full"].to_list(), b["n_full"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        ad = process.cpdist(a["a_full"].to_list(), b["a_full"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        hn = pl.DataFrame({"x": a["a_hn"], "y": b["a_hn"]}).select(
            (pl.col("x").list.set_intersection(pl.col("y")).list.len() > 0).cast(pl.Float32))["x"].to_numpy()
        pid = Xc["pid"].to_numpy()
        np.maximum.at(sib_name, pid, nm.astype(np.float32)); np.maximum.at(sib_addr, pid, ad.astype(np.float32))
        np.maximum.at(sib_hn, pid, hn); np.add.at(sib_cnt, pid, 1)
        del a, b
    F.update(sib_name=sib_name, sib_addr=sib_addr, sib_hn=sib_hn, sib_cnt=sib_cnt)
    return pl.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})


def group_features(D, p1, qn, pmin=0.3, top=12):
    """Group-level reasoning: within each S1's candidate records, link records that are near-duplicates of each other
    (copies of ONE business) and split them into groups. An orphan business (absent from S1) shows up as a coherent
    group that competes with the true group — so each record gets its group's size, strength, consensus with the S1,
    and the strength of the best competing group."""
    n = D.height
    q = D["qi"].to_numpy().astype(np.int64); s = D["si"].to_numpy().astype(np.int64)
    sel = np.where(p1 >= pmin)[0]
    T = (pl.DataFrame({"pid": sel, "q": q[sel], "s": s[sel], "p": p1[sel],
                       "nt": D["n_tsort"].to_numpy()[sel], "hn": D["hn_all_equal"].to_numpy()[sel]})
         .sort(["s", "p"], descending=[False, True])
         .with_columns(pl.int_range(pl.len()).over("s").alias("r")).filter(pl.col("r") < top)
         .with_row_index("node"))
    E = (T.select("node", "q", "s").join(T.select("node", "q", "s"), on="s", suffix="_b")
         .filter(pl.col("node") < pl.col("node_b")))
    link = np.zeros(E.height, bool)
    CH = 4_000_000
    for c0 in range(0, E.height, CH):
        e = E.slice(c0, CH)
        a = qn[e["q"].to_numpy()]; b = qn[e["q_b"].to_numpy()]
        nm = process.cpdist(a["n_full"].to_list(), b["n_full"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        ad = process.cpdist(a["a_full"].to_list(), b["a_full"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        hn = pl.DataFrame({"x": a["a_hn"], "y": b["a_hn"]}).select(
            (pl.col("x").list.set_intersection(pl.col("y")).list.len() > 0))["x"].to_numpy()
        empty = (a["a_full"] == "").to_numpy() | (b["a_full"] == "").to_numpy()
        # copies of one business: similar names AND (same house number or similar address);
        # an address-less copy joins on a near-exact name (38% of missed matches have no address)
        link[c0:c0 + CH] = ((nm >= 85) & (hn | (ad >= 85))) | ((nm >= 95) & empty)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    L = E.filter(pl.Series(link))
    N = T.height
    g = coo_matrix((np.ones(L.height), (L["node"].to_numpy(), L["node_b"].to_numpy())), shape=(N, N))
    _, comp = connected_components(g, directed=False)
    T = T.with_columns(pl.Series("g", comp))
    G = T.group_by("g").agg(pl.len().alias("g_size"), pl.col("p").sum().alias("g_sum_p"), pl.col("p").max().alias("g_max_p"),
                            pl.col("hn").mean().alias("g_hn_share"), pl.col("nt").mean().alias("g_name_mean"), pl.col("s").first())
    G = G.with_columns(pl.col("g_sum_p").max().over("s").alias("s_best_sum"),
                       (pl.col("g_max_p") >= 0.5).sum().over("s").alias("s_n_groups"))
    # strength of best OTHER group of the same S1
    G = G.with_columns(pl.col("g_sum_p").rank("ordinal", descending=True).over("s").alias("g_rank"))
    second = G.filter(pl.col("g_rank") == 2).select("s", pl.col("g_sum_p").alias("s_second_sum"))
    G = G.join(second, on="s", how="left").with_columns(pl.col("s_second_sum").fill_null(0.0))
    G = G.with_columns(pl.when(pl.col("g_rank") == 1).then(pl.col("s_second_sum")).otherwise(pl.col("s_best_sum")).alias("g_best_other"))
    T = T.join(G.select("g", "g_size", "g_sum_p", "g_hn_share", "g_name_mean", "g_rank", "s_n_groups", "g_best_other"), on="g")
    out = {k: np.full(n, -1.0, np.float32) for k in ["g_size", "g_sum_p", "g_hn_share", "g_name_mean", "g_rank", "s_n_groups", "g_best_other"]}
    pid = T["pid"].to_numpy()
    for k in out:
        out[k][pid] = T[k].cast(pl.Float32).to_numpy()
    return pl.DataFrame(out)


def evaluate(parts, p, tcounts):
    grid = np.round(np.arange(0.40, 0.96, 0.05), 2)
    res = {}
    for c, (D, off) in parts.items():
        pp = p[off:off + D.height]
        q = D["qi"].to_numpy(); s = D["si"].to_numpy(); y = D["y"].to_numpy(); sc = D["ret_score"].to_numpy()
        tc, val = tcounts[c]
        best = decide.best_per_query(q, s, pp, sc)
        res[c] = {}
        for t in grid:
            b = best[pp[best] >= t]
            pc = np.bincount(s[b], minlength=len(tc)); tp = np.bincount(s[b], weights=y[b], minlength=len(tc))
            res[c][float(t)] = (decide.f05_macro(tc[val], pc[val], tp[val]), float(pc[val].mean()))
    return res, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--lr1", type=float, default=0.08)
    ap.add_argument("--tag", default="", help="prefix for model/validation files (keeps experiments separate)")
    ap.add_argument("--aug", action="store_true", help="append <split>_<c>_aug.parquet noise-operation features")
    ap.add_argument("--group", action="store_true", help="add group-level (record-group) features to stage-2")
    ap.add_argument("--s1-tag", default=None, help="reuse stage-1 models saved under this tag (default: --tag)")
    cfg = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(cfg.work_dir, "stage2.log"))])
    T0 = time.time()
    P1["learning_rate"] = cfg.lr1
    pool = mp.get_context("spawn").Pool(10)
    td = json.load(open(os.path.join(cfg.work_dir, "translit_dict.json")))
    tr_s1, tr_s23 = io.load_split(cfg.data_dir, "train")
    # ---------------- load train partitions + normalized queries
    parts, qns, tcounts = {}, {}, {}
    off = 0
    for c in countries(cfg.work_dir, "train"):
        D = pl.read_parquet(os.path.join(cfg.work_dir, f"train_{c}.parquet"))
        if cfg.aug:
            D = D.hstack(pl.read_parquet(os.path.join(cfg.work_dir, f"train_{c}_aug.parquet")))
        qns[c] = norm_frame(cfg.work_dir, f"train_{c}", pool, tr_s1.filter(pl.col("country") == c),
                            tr_s23.filter(pl.col("country") == c), td, "qn")
        tcd = pl.read_parquet(os.path.join(cfg.work_dir, f"train_{c}_tcount.parquet"))
        tcounts[c] = (tcd["tcount"].to_numpy(), fold_of(tcd["s1_id"].to_list()) == 0)
        parts[c] = (D, off); off += D.height
        log.info("train/%s: %d pairs", c, D.height)
    del tr_s23; gc.collect()
    feat1 = [c for c in parts[next(iter(parts))][0].columns if c not in META]
    N = sum(D.height for D, _ in parts.values())
    X = np.empty((N, len(feat1)), np.float32)
    for D, o in parts.values():
        X[o:o + D.height] = D.select(feat1).to_numpy().astype(np.float32)
    y = np.concatenate([D["y"].to_numpy() for D, _ in parts.values()])
    fold = np.concatenate([fold_of(D["s1_id"].to_list()) for D, _ in parts.values()])
    # memory-lean: one binned Dataset, training/validation via row subsets (no matrix copies)
    base = lgb.Dataset(X, label=y, feature_name=feat1, free_raw_data=False).construct()
    va_idx = np.where(fold == 0)[0]

    def predict_rows(m, idx, extra=None, ch=2_000_000):
        out = np.empty(len(idx))
        for c0 in range(0, len(idx), ch):
            r = idx[c0:c0 + ch]
            M = X[r] if extra is None else np.hstack([X[r], extra[r]])
            out[c0:c0 + ch] = m.predict(M, num_threads=10)
        return out
    # ---------------- stage-1: M_all on folds 1-4 (for val + test), M_k on folds 1-4 minus k (OOF for folds 1-4)
    p1 = np.zeros(N)
    models1 = {}
    for k in [0, 1, 2, 3, 4]:
        path = os.path.join(cfg.work_dir, f"{cfg.tag if cfg.s1_tag is None else cfg.s1_tag}s1_fold{k}.txt")
        if os.path.exists(path):
            m = lgb.Booster(model_file=path)
        else:
            t = time.time()
            tr_idx = np.where((fold != 0) & (fold != k))[0]
            m = lgb.train(P1, base.subset(tr_idx), num_boost_round=cfg.rounds, valid_sets=[base.subset(va_idx)],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            m.save_model(path)
            log.info("stage-1 model k=%d: %d trees (%.0fs)", k, m.num_trees(), time.time() - t)
        models1[k] = m
        tgt = np.where((fold == 0) if k == 0 else (fold == k))[0]
        p1[tgt] = predict_rows(m, tgt)
    # ---------------- collective features
    Cdf = pl.concat([collective(D, p1[o:o + D.height], qns[c]).hstack(group_features(D, p1[o:o + D.height], qns[c]))
                     if cfg.group else collective(D, p1[o:o + D.height], qns[c]) for c, (D, o) in parts.items()])
    feat2 = feat1 + Cdf.columns
    C = Cdf.to_numpy().astype(np.float32); del Cdf; gc.collect()
    res1, grid = evaluate(parts, p1, tcounts)
    for c, r in res1.items():
        log.info("VAL stage-1 %s: %s", c, " ".join(f"{t:.2f}:{v[0]:.4f}" for t, v in r.items()))
    # ---------------- stage-2
    path2 = os.path.join(cfg.work_dir, f"{cfg.tag}s2.txt")
    if os.path.exists(path2):
        m2 = lgb.Booster(model_file=path2)
    else:
        t = time.time()
        base.add_features_from(lgb.Dataset(C, feature_name=feat2[len(feat1):], free_raw_data=False).construct())
        tr_idx = np.where(fold != 0)[0]
        m2 = lgb.train(P2, base.subset(tr_idx), num_boost_round=cfg.rounds, valid_sets=[base.subset(va_idx)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        m2.save_model(path2)
        log.info("stage-2 model: %d trees (%.0fs)", m2.num_trees(), time.time() - t)
    del base; gc.collect()
    p2 = predict_rows(m2, np.arange(N), extra=C)
    gain = sorted(zip(m2.feature_importance("gain"), feat2), reverse=True); tot = sum(g for g, _ in gain)
    log.info("stage-2 top gain: %s", ", ".join(f"{n}={g / tot * 100:.1f}%" for g, n in gain[:12]))
    res2, grid = evaluate(parts, p2, tcounts)
    for c, r in res2.items():
        log.info("VAL stage-2 %s: %s", c, " ".join(f"{t:.2f}:{v[0]:.4f}" for t, v in r.items()))
    t_star = max(grid, key=lambda t: min(res2[c][float(t)][0] for c in res2))
    ref_card = float(np.mean([res2[c][float(t_star)][1] for c in res2]))
    log.info("t* = %.2f  worst-country F0.5 %.4f  per-country %s  ref cardinality %.3f", t_star,
             min(res2[c][float(t_star)][0] for c in res2), {c: round(res2[c][float(t_star)][0], 4) for c in res2}, ref_card)
    json.dump({"t_star": float(t_star), "ref_card": ref_card,
               "stage1": {c: {str(k): v for k, v in r.items()} for c, r in res1.items()},
               "stage2": {c: {str(k): v for k, v in r.items()} for c, r in res2.items()}},
              open(os.path.join(cfg.work_dir, f"{cfg.tag}validation_stage2.json"), "w"), indent=1)
    del X, C, p1, p2; gc.collect()
    # ---------------- test
    te_s1, te_s23 = io.load_split(cfg.data_dir, "test")
    mparts, cparts = [], []
    for c in countries(cfg.work_dir, "test"):
        D = pl.read_parquet(os.path.join(cfg.work_dir, f"test_{c}.parquet"))
        if cfg.aug:
            D = D.hstack(pl.read_parquet(os.path.join(cfg.work_dir, f"test_{c}_aug.parquet")))
        s1c = te_s1.filter(pl.col("country") == c); s23c = te_s23.filter(pl.col("country") == c)
        qn = norm_frame(cfg.work_dir, f"test_{c}", pool, s1c, s23c, td, "qn")
        Xt = D.select(feat1).to_numpy().astype(np.float32)
        pt1 = models1[0].predict(Xt, num_threads=10)
        Ct = collective(D, pt1, qn)
        if cfg.group:
            Ct = Ct.hstack(group_features(D, pt1, qn))
        Xt2 = np.hstack([Xt, Ct.to_numpy().astype(np.float32)])
        pt = m2.predict(Xt2, num_threads=10)
        q = D["qi"].to_numpy(); s = D["si"].to_numpy(); sc = D["ret_score"].to_numpy()
        best = decide.best_per_query(q, s, pt, sc)
        t_c, _ = decide.card_matched_threshold(best, pt, s1c.height, ref_card, t_star)
        b = best[pt[best] >= t_c]
        log.info("test/%s: card@t*=%.3f  t_c=%.2f -> %.3f matches/S1, %.1f%% S1 empty", c,
                 decide.cardinality(best, pt, s1c.height, t_star), t_c, len(b) / s1c.height,
                 100 * (1 - len(np.unique(s[b])) / s1c.height))
        mparts.append(pl.DataFrame({"s1": D["s1_id"].gather(b), "rid": D["q_id"].gather(b)}))
        cparts.append(D.select(pl.col("s1_id").alias("s1"), pl.col("q_id").alias("rid")))
    all_s1 = te_s1["entity_id"].to_list()
    io.write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"], all_s1, pl.concat(mparts))
    io.write_id_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"], all_s1, pl.concat(cparts))
    pool.close()
    log.info("stage-2 total %.1f min", (time.time() - T0) / 60)


if __name__ == "__main__":
    main()
