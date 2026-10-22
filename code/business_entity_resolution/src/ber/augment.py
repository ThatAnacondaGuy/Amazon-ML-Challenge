"""Noise-operation features (cycle 3): identify HOW a record differs from a candidate S1 — the generator's genuine
noise (typos, truncation, inserted generic words, invented trade names) versus decoy perturbations (whole-word
substitution, initials change, house-number nudge). Computed from cached pairs + normalized records.

Usage: python -m ber.augment --data-dir ../../student_resource/dataset --work-dir work
Writes <split>_<country>_aug.parquet aligned row-by-row with <split>_<country>.parquet.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from . import blocking, io

log = logging.getLogger("ber")
AUG = ["tok_typo", "tok_subst_vocab", "tok_subst_other", "tok_short_mismatch", "q_only_vocab", "s_only_vocab",
       "hn_rel_equal", "hn_rel_trunc", "hn_rel_nudge", "hn_rel_numclose", "hn_q_extra", "hn_s_extra",
       "q_pseudo", "q_oov_share"]


def _hn_digits(t):
    return "".join(ch for ch in t if ch.isdigit())


def pair_ops(args):
    """Worker: per pair, classify token differences and house-number relations."""
    qt, st, qh, sh, vocab = args
    out = np.zeros((len(qt), len(AUG)), np.float32)
    for i in range(len(qt)):
        a, b = set(qt[i] or ()), set(st[i] or ())
        ua, ub = sorted(a - b), sorted(b - a)
        typo = subv = subo = short = 0
        cands = sorted((Levenshtein.normalized_distance(x, y), x, y) for x in ua for y in ub)
        ux, uy = set(), set()
        for best, x, y in cands:  # global best-first pairing of unmatched tokens
            if x in ux or y in uy:
                continue
            ux.add(x); uy.add(y)
            if len(x) <= 3 or len(y) <= 3:
                short += 1
            elif best <= 0.34:
                typo += 1
            elif x in vocab and y in vocab:
                subv += 1
            else:
                subo += 1
        qv = sum(1 for x in ua if x in vocab)
        sv = sum(1 for y in ub if y in vocab)
        # house numbers (canonical compound tokens)
        ha, hb = list(qh[i] or ()), list(sh[i] or ())
        eq = tr = nu = nc = 0
        if ha and hb:
            eq = int(bool(set(ha) & set(hb)))
            for x in set(ha) - set(hb):  # relations among the NON-shared numbers (nudge hides behind a shared one)
                for y in set(hb) - set(ha):
                    if x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x):
                        tr = 1
                    elif len(x) == len(y) and sum(c1 != c2 for c1, c2 in zip(x, y)) == 1:
                        nu = 1
                    dx, dy = _hn_digits(x), _hn_digits(y)
                    if dx and dy and len(dx) < 10 and len(dy) < 10 and abs(int(dx) - int(dy)) <= 10:
                        nc = 1
        qe = len(set(ha) - set(hb)); se = len(set(hb) - set(ha))
        oov = [x for x in (qt[i] or ()) if x not in vocab and not x.isdigit()]
        n = len(qt[i] or ())
        pseudo = float(n > 0 and len(oov) == n)
        out[i] = (typo, subv, subo, short, qv, sv, eq, tr, nu, nc, qe, se, pseudo, len(oov) / max(1, n))
    return out


def augment_partition(pool, D, qn, s1n, chunk=200_000):
    # partition vocabulary: tokens used by >= 20 S1 names (real words / business vocabulary)
    tok = s1n.select(pl.col("n_core").str.split(" ").list.unique()).explode("n_core").group_by("n_core").len()
    vocab = frozenset(tok.filter(pl.col("len") >= 20)["n_core"].to_list())
    qi = D["qi"].to_numpy(); si = D["si"].to_numpy()
    qt = qn["n_core"].gather(qi).str.split(" ").to_list(); st = s1n["n_core"].gather(si).str.split(" ").to_list()
    qh = qn["a_hn"].gather(qi).to_list(); sh = s1n["a_hn"].gather(si).to_list()
    jobs = [(qt[i:i + chunk], st[i:i + chunk], qh[i:i + chunk], sh[i:i + chunk], vocab) for i in range(0, len(qt), chunk)]
    res = pool.map(pair_ops, jobs)
    return pl.DataFrame(np.vstack(res), schema=AUG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--work-dir", default="work")
    cfg = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    pool = mp.get_context("spawn").Pool(8)
    td = json.load(open(os.path.join(cfg.work_dir, "translit_dict.json")))
    for split in ["train", "test"]:
        s1, s23 = io.load_split(cfg.data_dir, split)
        for f in sorted(os.listdir(cfg.work_dir)):
            if not (f.startswith(split + "_") and f.endswith(".parquet") and f.count("_") == 1):
                continue
            c = f[len(split) + 1:-8]
            out = os.path.join(cfg.work_dir, f"{split}_{c}_aug.parquet")
            if os.path.exists(out):
                continue
            t = time.time()
            D = pl.read_parquet(os.path.join(cfg.work_dir, f), columns=["qi", "si", "s1_id"])
            s1c = s1.filter(pl.col("country") == c); s23c = s23.filter(pl.col("country") == c)
            # S1 frame must match the index order used when the partition was built (possibly shift-reduced)
            ids = pl.read_parquet(os.path.join(cfg.work_dir, f"{split}_{c}_tcount.parquet"))["s1_id"] if split == "train" \
                else s1c["entity_id"]
            s1c = pl.DataFrame({"entity_id": ids}).join(s1c, on="entity_id", how="left", maintain_order="left")
            qp = os.path.join(cfg.work_dir, f"{split}_{c}_qn.parquet")
            sp = os.path.join(cfg.work_dir, f"{split}_{c}_s1n.parquet")
            qn = pl.read_parquet(qp) if os.path.exists(qp) else blocking.normalize_only(pool, s23c["business_name"].to_list(), s23c["business_address"].to_list(), td)
            s1n = pl.read_parquet(sp) if os.path.exists(sp) else blocking.normalize_only(pool, s1c["business_name"].to_list(), s1c["business_address"].to_list(), td)
            A = augment_partition(pool, D, qn, s1n)
            A.write_parquet(out)
            log.info("%s/%s: %d pairs augmented in %.0fs", split, c, A.height, time.time() - t)
    pool.close()


if __name__ == "__main__":
    main()
