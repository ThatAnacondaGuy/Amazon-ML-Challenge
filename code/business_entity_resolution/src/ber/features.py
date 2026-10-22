"""Pairwise features for (S2/S3 record, S1 candidate) pairs. Country-agnostic: country is never a feature."""
import logging

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

log = logging.getLogger("ber")


def noise_lexicon(s1n, qn, min_ratio=5.0):
    """Tokens over-represented in S2/S3 names vs S1 names of the same partition (unsupervised)."""
    def df(frame):
        return (frame.select(pl.col("n_tok").list.unique()).explode("n_tok").drop_nulls()
                .group_by("n_tok").len())
    if s1n.height == 0 or qn.height == 0:
        return []
    a, b = df(s1n).rename({"len": "d1"}), df(qn).rename({"len": "d2"})
    m = b.join(a, on="n_tok", how="left").with_columns(pl.col("d1").fill_null(0))
    m = m.with_columns(((pl.col("d2") / qn.height) / ((pl.col("d1") + 5) / s1n.height)).alias("r"))
    lex = m.filter((pl.col("r") >= min_ratio) & (pl.col("d2") >= max(200, int(1e-4 * qn.height))))["n_tok"].to_list()
    return lex


def pair_features(s1n, qn, pairs, q_src2, noise, chunk=3_000_000):
    """pairs: DataFrame(q, s, score, rank, top1, cos_n, cos_a). Returns float32 matrix + names."""
    s_twins = s1n.group_by("n_core").len().rename({"len": "s_twins"})
    a_twins = s1n.filter(pl.col("a_key") != "").group_by("a_key").len().rename({"len": "a_twins"})
    s1x = (s1n.with_row_index("s").join(s_twins, on="n_core", how="left")
              .join(a_twins, on="a_key", how="left").with_columns(pl.col("a_twins").fill_null(0)))
    qx = qn.with_columns(pl.col("n_tok").list.eval(pl.element().filter(pl.element().is_in(noise))).alias("q_noise"))
    out = []
    for c0 in range(0, pairs.height, chunk):
        P = pairs.slice(c0, chunk)
        qi = P["q"].to_numpy(); si = P["s"].to_numpy()
        A = qx[qi]
        B = s1x[si]
        W = -1
        f = {}
        f["ret_score"] = P["score"].to_numpy()
        f["ret_rank"] = P["rank"].to_numpy()
        f["ret_ratio"] = P["score"].to_numpy() / np.maximum(P["top1"].to_numpy(), 1e-6)
        f["ret_gap"] = (P["top1"] - P["score"]).to_numpy()
        f["cos_name"] = P["cos_n"].to_numpy(); f["cos_addr"] = P["cos_a"].to_numpy()
        qa, sb = A["n_full"].to_list(), B["n_full"].to_list()
        f["n_ratio"] = process.cpdist(qa, sb, scorer=fuzz.ratio, workers=W)
        f["n_tsort"] = process.cpdist(qa, sb, scorer=fuzz.token_sort_ratio, workers=W)
        qa, sb = A["n_core"].to_list(), B["n_core"].to_list()
        f["n_tset_core"] = process.cpdist(qa, sb, scorer=fuzz.token_set_ratio, workers=W)
        f["n_jw_core"] = process.cpdist(qa, sb, scorer=distance.JaroWinkler.normalized_similarity, workers=W)
        qa, sb = A["n_ns"].to_list(), B["n_ns"].to_list()
        f["n_ns_ratio"] = process.cpdist(qa, sb, scorer=fuzz.ratio, workers=W)
        f["n_ns_partial"] = process.cpdist(qa, sb, scorer=fuzz.partial_ratio, workers=W)
        f["a_tset"] = process.cpdist(A["a_full"].to_list(), B["a_full"].to_list(), scorer=fuzz.token_set_ratio, workers=W)
        J = pl.DataFrame({
            "qt": A["n_tok"], "st": B["n_tok"], "qk": A["n_sk"], "sk": B["n_sk"], "qa": A["a_tok"], "sa": B["a_tok"],
            "qn": A["a_num"], "sn": B["a_num"], "qh": A["a_hn"], "sh": B["a_hn"], "qz": A["q_noise"],
            "qns": A["n_ns"], "sinit": B["n_init"], "qkey": A["a_key"], "skey": B["a_key"]})

        def jac(a, b):
            return (pl.col(a).list.set_intersection(pl.col(b)).list.len() /
                    pl.max_horizontal(pl.col(a).list.set_union(pl.col(b)).list.len(), pl.lit(1)))
        qf, sf = pl.col("qn").list.first(), pl.col("sn").list.first()
        J = J.select(
            jac("qt", "st").alias("n_tok_jacc"), jac("qk", "sk").alias("n_skel_jacc"), jac("qa", "sa").alias("a_jacc"),
            pl.col("qt").list.set_difference(pl.col("st")).list.len().alias("q_extra_tok"),
            pl.col("st").list.set_difference(pl.col("qt")).list.len().alias("s_missing_tok"),
            pl.col("qz").list.set_difference(pl.col("st")).list.len().alias("q_extra_noise"),
            (pl.col("qn").list.set_intersection(pl.col("sn")).list.len() > 0).alias("num_any"),
            (qf == sf).fill_null(False).alias("num_first"),
            ((qf.str.starts_with(sf) | sf.str.starts_with(qf) | qf.str.ends_with(sf) | sf.str.ends_with(qf))
             & (qf != sf)).fill_null(False).alias("num_contain"),
            ((qf.cast(pl.Float64, strict=False) - sf.cast(pl.Float64, strict=False)).abs() + 1).log()
            .fill_null(-1).alias("num_logdiff"),
            pl.col("qh").list.set_intersection(pl.col("sh")).list.len().alias("hn_common"),
            ((pl.col("qh").list.len() > 0) & (pl.col("qh").list.sort() == pl.col("sh").list.sort())).alias("hn_all_equal"),
            pl.col("qh").list.len().alias("hn_q"), pl.col("sh").list.len().alias("hn_s"),
            pl.col("qn").list.len().alias("num_q"), pl.col("sn").list.len().alias("num_s"),
            pl.col("qt").list.len().alias("n_len_q"), pl.col("st").list.len().alias("n_len_s"),
            pl.col("qa").list.len().alias("a_len_q"), pl.col("sa").list.len().alias("a_len_s"),
            ((pl.col("qns") == pl.col("sinit")) & (pl.col("qns").str.len_chars() <= 5) & (pl.col("sinit").str.len_chars() >= 2))
            .alias("n_initials"),
            ((pl.col("qkey") == pl.col("skey")) & (pl.col("qkey") != "")).alias("addr_key_equal"),
        )
        for c in J.columns:
            f[c] = J[c].cast(pl.Float32).to_numpy()
        f["q_addr_empty"] = (A["a_full"] == "").cast(pl.Float32).to_numpy()
        f["q_domain"] = A["n_domain"].cast(pl.Float32).to_numpy()
        f["q_nonlatin"] = A["n_nonlatin"].cast(pl.Float32).to_numpy()
        f["q_src2"] = q_src2[qi].astype(np.float32)
        f["s_twins"] = np.log1p(B["s_twins"].cast(pl.Float32).to_numpy())
        f["a_twins"] = np.log1p(B["a_twins"].cast(pl.Float32).to_numpy())
        out.append(pl.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()}))
        log.info("    features %d/%d", min(c0 + chunk, pairs.height), pairs.height)
    F = pl.concat(out) if out else pl.DataFrame()
    # query-level context: candidates per query, name twins among them, co-location among them
    ctx = pl.DataFrame({"q": pairs["q"], "tw": (F["n_tset_core"] >= 95).cast(pl.Int32), "ak": s1n["a_key"].gather(pairs["s"]),
                        "nt": F["n_tsort"]})
    ctx = ctx.with_columns(
        pl.len().over("q").alias("n_cands"),
        pl.col("tw").sum().over("q").alias("q_twin_cands"),
        pl.when(pl.col("ak") != "").then(pl.len().over(["q", "ak"])).otherwise(1).alias("coloc_n"),
        (pl.col("nt") >= pl.col("nt").max().over(["q", "ak"])).alias("coloc_best"))
    F = F.with_columns([ctx[c].cast(pl.Float32) for c in ["n_cands", "q_twin_cands", "coloc_n", "coloc_best"]])
    return F
