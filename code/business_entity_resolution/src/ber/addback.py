"""France recall add-back (final step). France randomizes house numbers on genuine copies, which the US/India-trained
model reads as a decoy signal. Re-add, for the France partition, each record's best candidate when the names are
identical (core token-set 100, token-sort >= 95), the street/city words agree (address token-set >= 97, numbers ignored),
there is no word substitution/short-token mismatch, and the model score is in [0.3, 0.95).

Usage: python -m ber.addback --data-dir <dataset> --work-dir work --in-dir <output> --out-dir <output2> --partition France --tag cg_
"""
import argparse
import os
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl

from . import decide, io, stage2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--work-dir", default="work")
    ap.add_argument("--in-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--partition", default="France"); ap.add_argument("--tag", default="cg_")
    cfg = ap.parse_args()
    c, W = cfg.partition, cfg.work_dir
    D = pl.read_parquet(os.path.join(W, f"test_{c}.parquet")).hstack(pl.read_parquet(os.path.join(W, f"test_{c}_aug.parquet")))
    qn = pl.read_parquet(os.path.join(W, f"test_{c}_qn.parquet"))
    feat1 = [x for x in D.columns if x not in stage2.META]
    X = D.select(feat1).to_numpy().astype(np.float32)
    p1 = lgb.Booster(model_file=os.path.join(W, f"{cfg.tag}s1_fold0.txt")).predict(X, num_threads=10)
    C = stage2.collective(D, p1, qn).hstack(stage2.group_features(D, p1, qn))
    p = lgb.Booster(model_file=os.path.join(W, f"{cfg.tag}s2.txt")).predict(np.hstack([X, C.to_numpy().astype(np.float32)]), num_threads=10)
    best = decide.best_per_query(D["qi"].to_numpy(), D["si"].to_numpy(), p, D["ret_score"].to_numpy())
    B = D[best].with_columns(pl.Series("p", p[best]))
    M = (pl.read_csv(os.path.join(cfg.in_dir, "matching_results.tsv"), separator="\t", infer_schema=False,
                     missing_utf8_is_empty_string=True)
         .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "")
         .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("q_id")))
    cand = (B.filter((pl.col("p") >= 0.3) & (pl.col("p") < 0.95) & (pl.col("n_tset_core") >= 100) & (pl.col("n_tsort") >= 95)
                     & (pl.col("a_tset") >= 97) & (pl.col("q_addr_empty") == 0) & (pl.col("tok_subst_vocab") == 0)
                     & (pl.col("tok_short_mismatch") == 0) & (pl.col("tok_subst_other") == 0))
            .join(M.select("q_id"), on="q_id", how="anti"))
    print(f"{c}: re-added {cand.height:,} matches")
    te1, _ = io.load_split(cfg.data_dir, "test")
    out = pl.concat([M, cand.select("s1_id", "q_id")]).rename({"s1_id": "s1", "q_id": "rid"})
    os.makedirs(cfg.out_dir, exist_ok=True)
    io.write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"],
                      te1["entity_id"].to_list(), out)
    shutil.copy(os.path.join(cfg.in_dir, "candidate_pairs.tsv"), os.path.join(cfg.out_dir, "candidate_pairs.tsv"))


if __name__ == "__main__":
    main()
