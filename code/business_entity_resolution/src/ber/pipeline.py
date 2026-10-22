"""End-to-end pipeline: train split -> candidates + features -> LightGBM -> validation -> test inference -> outputs.

Usage (from code/business_entity_resolution):
    python -m ber.pipeline --data-dir ../../student_resource/dataset --work-dir work --out-dir ../../output
"""
import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

from . import blocking, decide, features, io, learn

log = logging.getLogger("ber")
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=7,
                  deterministic=True, force_row_wise=True)


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def is_val(ids):
    return np.array([zlib.crc32(x.encode()) % 5 == 0 for x in ids])


def countries_of(df):
    return sorted(df["country"].unique().to_list())


NORM_KEEP = ["n_full", "n_core", "n_ns", "a_full", "a_hn", "a_key"]


def build_partition(pool, s1c, s23c, td, cfg, tag=None):
    s1n, qn, pairs = blocking.block_partition(pool, s1c, s23c, td, k=cfg.k, ratio=cfg.ratio, cap=cfg.cap, wa=cfg.wa)
    noise = features.noise_lexicon(s1n, qn)
    q_src2 = s23c["entity_id"].str.starts_with("S2-").to_numpy()
    F = features.pair_features(s1n, qn, pairs, q_src2, noise) if pairs.height else pl.DataFrame()
    log.info("  %d pairs x %d features, noise lexicon %d tokens, RSS %.1f GB", F.height, F.width, len(noise), rss_gb())
    if tag:  # normalized records needed by stage-2 sibling features
        qn.select(NORM_KEEP).write_parquet(os.path.join(cfg.work_dir, f"{tag}_qn.parquet"))
        s1n.select(NORM_KEEP).write_parquet(os.path.join(cfg.work_dir, f"{tag}_s1n.parquet"))
    return pairs.select("q", "s"), F


def stage_train_partitions(pool, cfg, tr_s1, tr_s23, gt, td):
    for c in countries_of(tr_s1):
        path = os.path.join(cfg.work_dir, f"train_{c}.parquet")
        if os.path.exists(path):
            log.info("train/%s: cached", c)
            continue
        t0 = time.time()
        s1c = tr_s1.filter(pl.col("country") == c)
        s23c = tr_s23.filter(pl.col("country") == c)
        log.info("train/%s: S1=%d records=%d", c, s1c.height, s23c.height)
        pairs, F = build_partition(pool, s1c, s23c, td, cfg, tag=f"train_{c}")
        s1_ids = s1c["entity_id"]; q_ids = s23c["entity_id"]
        truth = (s23c.select("entity_id").with_row_index("q").join(gt, left_on="entity_id", right_on="rid")
                 .join(s1c.select("entity_id").with_row_index("s"), left_on="s1", right_on="entity_id"))
        tq = np.full(s23c.height, -1, np.int64)
        tq[truth["q"].to_numpy()] = truth["s"].to_numpy()
        y = (tq[pairs["q"].to_numpy()] == pairs["s"].to_numpy()).astype(np.int8)
        rec = tq[tq >= 0]
        found = y.sum()
        log.info("  pair recall (candidates): %.4f  (%d of %d true pairs)", found / max(1, len(rec)), found, len(rec))
        out = F.with_columns(pl.Series("y", y), pl.Series("q_id", q_ids.gather(pairs["q"])),
                             pl.Series("s1_id", s1_ids.gather(pairs["s"])), pl.Series("qi", pairs["q"]), pl.Series("si", pairs["s"]))
        out.write_parquet(path)
        tc = np.bincount(rec, minlength=s1c.height)
        pl.DataFrame({"s1_id": s1_ids, "tcount": tc}).write_parquet(os.path.join(cfg.work_dir, f"train_{c}_tcount.parquet"))
        log.info("train/%s done in %.0fs", c, time.time() - t0)
        del pairs, F, out; gc.collect()


def load_train(cfg, tr_s1):
    parts = []
    for c in countries_of(tr_s1):
        d = pl.read_parquet(os.path.join(cfg.work_dir, f"train_{c}.parquet")).with_columns(pl.lit(c).alias("country"))
        parts.append(d)
    return pl.concat(parts, how="vertical_relaxed")


def evaluate(cfg, D, p, tr_s1):
    """Macro F0.5 on validation S1 entities per country, over a threshold grid. Returns table + t* + ref cardinality."""
    res = {}
    grid = np.round(np.arange(0.40, 0.96, 0.05), 2)
    for c in countries_of(tr_s1):
        m = (D["country"] == c).to_numpy()
        q = D["qi"].to_numpy()[m]; s = D["si"].to_numpy()[m]; y = D["y"].to_numpy()[m]
        sc = D["ret_score"].to_numpy()[m]; pp = p[m]
        tcd = pl.read_parquet(os.path.join(cfg.work_dir, f"train_{c}_tcount.parquet"))
        val = is_val(tcd["s1_id"].to_list())
        tc = tcd["tcount"].to_numpy()
        best = decide.best_per_query(q, s, pp, sc)
        row = {}
        for t in grid:
            b = best[pp[best] >= t]
            pc = np.bincount(s[b], minlength=len(tc)); tp = np.bincount(s[b], weights=y[b], minlength=len(tc))
            row[float(t)] = (decide.f05_macro(tc[val], pc[val], tp[val]), pc[val].mean())
        res[c] = row
    return res, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--wa", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--shift", action="store_true",
                    help="drop a per-country fraction of train S1 so records-per-S1 matches test (their matches become distractors)")
    ap.add_argument("--train-only", action="store_true", help="only build train partitions (for ber.stage2)")
    ap.add_argument("--test-only", action="store_true", help="only build and cache test partitions (for ber.stage2)")
    cfg = ap.parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True); os.makedirs(cfg.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(cfg.work_dir, "run.log"))])
    T0 = time.time()
    pool = mp.get_context("spawn").Pool(cfg.procs)

    if cfg.test_only:
        td = json.load(open(os.path.join(cfg.work_dir, "translit_dict.json")))
        te_s1, te_s23 = io.load_split(cfg.data_dir, "test")
        for c in countries_of(te_s1):
            if os.path.exists(os.path.join(cfg.work_dir, f"test_{c}.parquet")):
                log.info("test/%s: cached", c); continue
            s1c = te_s1.filter(pl.col("country") == c); s23c = te_s23.filter(pl.col("country") == c)
            log.info("test/%s: S1=%d records=%d", c, s1c.height, s23c.height)
            if s1c.height == 0 or s23c.height == 0:
                continue
            pairs, F = build_partition(pool, s1c, s23c, td, cfg, tag=f"test_{c}")
            F.with_columns(pl.Series("q_id", s23c["entity_id"].gather(pairs["q"])), pl.Series("s1_id", s1c["entity_id"].gather(pairs["s"])),
                           pl.Series("qi", pairs["q"]), pl.Series("si", pairs["s"])).write_parquet(os.path.join(cfg.work_dir, f"test_{c}.parquet"))
            del pairs, F; gc.collect()
        pool.close(); log.info("test partitions cached (%.1f min)", (time.time() - T0) / 60); return
    # ---------------- train side
    tr_s1, tr_s23 = io.load_split(cfg.data_dir, "train")
    gt = io.load_gt_pairs(cfg.data_dir)
    if cfg.shift:
        te_s1_, te_s23_ = io.load_split(cfg.data_dir, "test")
        keep = []
        for c in countries_of(tr_s1):
            r_tr = tr_s23.filter(pl.col("country") == c).height / max(1, tr_s1.filter(pl.col("country") == c).height)
            n_te = te_s1_.filter(pl.col("country") == c).height
            r_te = te_s23_.filter(pl.col("country") == c).height / max(1, n_te) if n_te else r_tr
            d = min(0.4, max(0.0, 1 - r_tr / r_te))
            sc = tr_s1.filter(pl.col("country") == c)
            # never drop validation-fold entities; drop the fraction d from the rest (deterministic)
            ids = sc["entity_id"].to_list()
            h = np.array([zlib.crc32(("drop" + x).encode()) % 10000 for x in ids]) / 10000
            v = is_val(ids)
            keep.append(sc.filter(pl.Series(v | (h >= d / 0.8))))
            log.info("shift: %s records/S1 train %.2f test %.2f -> drop %.1f%% of non-val S1", c, r_tr, r_te, 100 * d / 0.8)
        tr_s1 = pl.concat(keep)
        del te_s1_, te_s23_
    td_path = os.path.join(cfg.work_dir, "translit_dict.json")
    if os.path.exists(td_path):
        td = json.load(open(td_path))
    else:
        val_ids = set(tr_s1["entity_id"].filter(pl.Series(is_val(tr_s1["entity_id"].to_list()))).to_list())
        td = learn.learn_translit_dict(tr_s1, tr_s23, gt, exclude_s1=val_ids)
        json.dump(td, open(td_path, "w"))
    stage_train_partitions(pool, cfg, tr_s1, tr_s23, gt, td)
    del tr_s23; gc.collect()
    if cfg.train_only:
        pool.close(); log.info("train partitions done (%.1f min)", (time.time() - T0) / 60); return

    # ---------------- model
    D = load_train(cfg, tr_s1)
    feat_cols = [c for c in D.columns if c not in ("y", "q_id", "s1_id", "qi", "si", "country")]
    val_pair = pl.Series(is_val(D["s1_id"].to_list())).to_numpy()
    X = D.select(feat_cols).to_numpy().astype(np.float32); y = D["y"].to_numpy()
    model_path = os.path.join(cfg.work_dir, "model_stage1.txt")
    if os.path.exists(model_path):
        bst = lgb.Booster(model_file=model_path)
    else:
        t0 = time.time()
        dtr = lgb.Dataset(X[~val_pair], y[~val_pair], feature_name=feat_cols, free_raw_data=True)
        dva = lgb.Dataset(X[val_pair], y[val_pair], reference=dtr)
        bst = lgb.train(LGB_PARAMS, dtr, num_boost_round=cfg.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        bst.save_model(model_path)
        log.info("model: %d trees, %d train rows, %.0fs", bst.num_trees(), (~val_pair).sum(), time.time() - t0)
    p = bst.predict(X, num_threads=10)
    gain = sorted(zip(bst.feature_importance("gain"), feat_cols), reverse=True)
    tot = sum(g for g, _ in gain)
    log.info("top gain: %s", ", ".join(f"{n}={g / tot * 100:.1f}%" for g, n in gain[:12]))
    res, grid = evaluate(cfg, D, p, tr_s1)
    for c, row in res.items():
        log.info("VAL %s: %s", c, "  ".join(f"t={t:.2f}:{v[0]:.4f}" for t, v in row.items()))
    # t* maximizes the worst country; reference cardinality = mean predicted matches per val S1 at t*
    t_star = max(grid, key=lambda t: min(res[c][float(t)][0] for c in res))
    ref_card = float(np.mean([res[c][float(t_star)][1] for c in res]))
    log.info("t* = %.2f (worst-country F0.5 %.4f), reference cardinality %.3f", t_star,
             min(res[c][float(t_star)][0] for c in res), ref_card)
    json.dump({"t_star": float(t_star), "ref_card": ref_card, "val": {c: {str(k): v for k, v in r.items()} for c, r in res.items()}},
              open(os.path.join(cfg.work_dir, "validation.json"), "w"), indent=1)
    del D, X, y, p; gc.collect()

    # ---------------- test side
    te_s1, te_s23 = io.load_split(cfg.data_dir, "test")
    match_parts, cand_parts = [], []
    for c in countries_of(te_s1):
        s1c = te_s1.filter(pl.col("country") == c); s23c = te_s23.filter(pl.col("country") == c)
        log.info("test/%s: S1=%d records=%d", c, s1c.height, s23c.height)
        if s1c.height == 0 or s23c.height == 0:
            continue
        pairs, F = build_partition(pool, s1c, s23c, td, cfg, tag=f"test_{c}")
        if pairs.height == 0:
            continue
        F.with_columns(pl.Series("q_id", s23c["entity_id"].gather(pairs["q"])), pl.Series("s1_id", s1c["entity_id"].gather(pairs["s"])),
                       pl.Series("qi", pairs["q"]), pl.Series("si", pairs["s"])).write_parquet(os.path.join(cfg.work_dir, f"test_{c}.parquet"))
        pl.DataFrame({"s1_id": s1c["entity_id"]}).write_parquet(os.path.join(cfg.work_dir, f"test_{c}_s1ids.parquet"))
        pp = bst.predict(F.select(feat_cols).to_numpy().astype(np.float32), num_threads=10)
        q = pairs["q"].to_numpy(); s = pairs["s"].to_numpy(); sc = F["ret_score"].to_numpy()
        best = decide.best_per_query(q, s, pp, sc)
        t_c, cards = decide.card_matched_threshold(best, pp, s1c.height, ref_card, t_star)
        card_at_star = decide.cardinality(best, pp, s1c.height, t_star)
        b = best[pp[best] >= t_c]
        log.info("  %s: cardinality at t*=%.3f; cardinality-matched t_c=%.2f -> %.3f matches/S1, singletons %.1f%%",
                 c, card_at_star, t_c, len(b) / s1c.height,
                 100 * (1 - len(np.unique(s[b])) / s1c.height))
        s1_ids = s1c["entity_id"]; q_ids = s23c["entity_id"]
        match_parts.append(pl.DataFrame({"s1": s1_ids.gather(s[b]), "rid": q_ids.gather(q[b])}))
        cand_parts.append(pl.DataFrame({"s1": s1_ids.gather(s), "rid": q_ids.gather(q)}))
        del pairs, F; gc.collect()
    matches = pl.concat(match_parts) if match_parts else pl.DataFrame({"s1": [], "rid": []}, schema={"s1": pl.Utf8, "rid": pl.Utf8})
    cands = pl.concat(cand_parts) if cand_parts else matches
    all_s1 = te_s1["entity_id"].to_list()
    io.write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"], all_s1, matches)
    io.write_id_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"], all_s1, cands)
    pool.close()
    log.info("total runtime %.1f min, RSS %.1f GB", (time.time() - T0) / 60, rss_gb())


if __name__ == "__main__":
    main()
