"""Post-processing for label-free partitions: remove 'pure template swaps'.

Diagnosed on the unlabelled France partition (manual inspection of ~90 predicted pairs): its names are built from a
small template vocabulary (club, amicale, ecole, collectif, comite, ...), and its decoy records swap exactly one
template word for another at the same address ("Kilti Amis" vs S1 "Kilti Culture"), while genuine copies drop a word
and/or add an injected noise suffix ("Et Fils", "Services", "& Associes", "France"). In the labelled partitions
(US/India) such swaps are ~98-99% true matches, so the filter is applied only to the partitions passed explicitly.

A matched pair is removed when, after removing legal forms and the partition's injected-noise tokens (tokens
over-represented in S2/S3 names vs S1 names by a ratio >= 1.15, measured on the partition itself), the record and the
S1 name differ by exactly one token each, both tokens are ordinary vocabulary words (>= 20 S1 names, >= 4 chars), the
difference is not a typo (normalized edit distance > 0.34), the names share at least one other token, and the
house numbers coincide.

Usage: python -m ber.postprocess --data-dir <dataset> --work-dir work --in-dir <output> --out-dir <output2> --partitions France
"""
import argparse
import os
import shutil

import polars as pl
from rapidfuzz.distance import Levenshtein

from . import io, text


def partition_lexicons(work, c, ratio=1.15):
    s1n = pl.read_parquet(os.path.join(work, f"test_{c}_s1n.parquet"))
    qn = pl.read_parquet(os.path.join(work, f"test_{c}_qn.parquet"))
    t1 = s1n.select(pl.col("n_core").str.split(" ").list.unique()).explode("n_core").group_by("n_core").len()
    t2 = qn.select(pl.col("n_core").str.split(" ").list.unique()).explode("n_core").group_by("n_core").len()
    m = t2.join(t1, on="n_core", how="left", suffix="_1").with_columns(pl.col("len_1").fill_null(0))
    m = m.with_columns(((pl.col("len") / qn.height) / ((pl.col("len_1") + 5) / s1n.height)).alias("r"))
    noise = set(m.filter((pl.col("r") >= ratio) & (pl.col("len") >= 200))["n_core"].to_list())
    vocab = set(t1.filter(pl.col("len") >= 20)["n_core"].to_list())
    return s1n, qn, noise, vocab


def is_template_swap(qcore, score_, qhn, shn, noise, vocab, any_number=False):
    Q = set(qcore.split()) - noise - text.LEGAL
    S = set(score_.split()) - noise - text.LEGAL
    ua, ub = Q - S, S - Q
    if len(ua) != 1 or len(ub) != 1 or not (Q & S):
        return False
    x, y = next(iter(ua)), next(iter(ub))
    if len(x) < 4 or len(y) < 4 or x not in vocab or y not in vocab or Levenshtein.normalized_distance(x, y) <= 0.34:
        return False
    # France randomizes house numbers on genuine copies, so a swap is a decoy signal regardless of the number
    return True if any_number else bool(set(qhn or []) & set(shn or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--work-dir", default="work")
    ap.add_argument("--in-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--partitions", nargs="+", required=True)
    ap.add_argument("--any-number", action="store_true", help="remove swaps even when house numbers differ/are missing")
    cfg = ap.parse_args()
    te1, te23 = io.load_split(cfg.data_dir, "test")
    M = (pl.read_csv(os.path.join(cfg.in_dir, "matching_results.tsv"), separator="\t", infer_schema=False,
                     missing_utf8_is_empty_string=True)
         .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "")
         .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rid")))
    removed = []
    for c in cfg.partitions:
        s1n, qn, noise, vocab = partition_lexicons(cfg.work_dir, c)
        s1c = te1.filter(pl.col("country") == c); s23c = te23.filter(pl.col("country") == c)
        sp = {e: i for i, e in enumerate(s1c["entity_id"].to_list())}; qp = {e: i for i, e in enumerate(s23c["entity_id"].to_list())}
        sc, qc, sh, qh = s1n["n_core"].to_list(), qn["n_core"].to_list(), s1n["a_hn"].to_list(), qn["a_hn"].to_list()
        n0 = len(removed)
        for s1, rid in M.filter(pl.col("s1").is_in(list(sp))).iter_rows():
            i, j = sp[s1], qp[rid]
            if is_template_swap(qc[j], sc[i], qh[j], sh[i], noise, vocab, cfg.any_number):
                removed.append((s1, rid))
        print(f"{c}: removed {len(removed) - n0:,} template-swap matches")
    rem = pl.DataFrame(removed, schema=["s1", "rid"], orient="row")
    out = M.join(rem, on=["s1", "rid"], how="anti")
    os.makedirs(cfg.out_dir, exist_ok=True)
    io.write_id_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"],
                      te1["entity_id"].to_list(), out)
    shutil.copy(os.path.join(cfg.in_dir, "candidate_pairs.tsv"), os.path.join(cfg.out_dir, "candidate_pairs.tsv"))


if __name__ == "__main__":
    main()
