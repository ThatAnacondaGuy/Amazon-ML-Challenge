import sys, time
sys.path.insert(0, "/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad")
from load import *
import polars as pl, numpy as np
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(220)

s1 = load_source("train", 1); s2 = load_source("train", 2); s3 = load_source("train", 3)
gt = load_gt()
print("GT rows", gt.height, "unique S1", gt["source1_entity_id"].n_unique(),
      "S1 ids all in GT:", s1.join(gt, left_on="entity_id", right_on="source1_entity_id", how="anti").height == 0)
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
           .rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
print("positive pairs:", f"{pairs.height:,}", " dup pairs:", pairs.height - pairs.unique().height)
k = gt.select(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("k"))["k"]
print("\nmatches per S1: mean=%.3f" % k.mean())
vc = k.value_counts().sort("k"); tot = vc["count"].sum()
print("  ", {r[0]: f"{r[1]:,} ({r[1]/tot*100:.2f}%)" for r in vc.iter_rows() if r[0] <= 15}, " max=", k.max())
print("  k>15:", (k > 15).sum(), " k>30:", (k > 30).sum())
pairs = pairs.with_columns(pl.col("rid").str.slice(0, 2).alias("src"))
print("pair source split:", dict(pairs.group_by("src").len().iter_rows()))
per = pairs.group_by("s1").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
print("S1 with >=1 match: has S2 only %.1f%%, S3 only %.1f%%, both %.1f%%" % (
    ((per["n2"] > 0) & (per["n3"] == 0)).mean()*100, ((per["n2"] == 0) & (per["n3"] > 0)).mean()*100, ((per["n2"] > 0) & (per["n3"] > 0)).mean()*100))
print("  per-source multiplicity within one S1: n2 dist", dict(per["n2"].value_counts().sort("n2").head(8).iter_rows()),
      "\n                                            n3 dist", dict(per["n3"].value_counts().sort("n3").head(8).iter_rows()))
# exclusivity: each S2/S3 record in at most one S1 list?
multi = pairs.group_by("rid").len().filter(pl.col("len") > 1)
print("\nS2/S3 ids appearing under >1 S1:", multi.height)
# coverage of S2/S3
allr = pl.concat([s2.select("entity_id", "country"), s3.select("entity_id", "country")])
known = allr.join(pairs.select("rid"), left_on="entity_id", right_on="rid", how="semi")
print("GT ids missing from S2/S3 files:", pairs.join(allr, left_on="rid", right_on="entity_id", how="anti").height)
print("fraction of S2 records matched: %.1f%%, S3: %.1f%%" % (
    s2.join(pairs, left_on="entity_id", right_on="rid", how="semi").height / s2.height * 100,
    s3.join(pairs, left_on="entity_id", right_on="rid", how="semi").height / s3.height * 100))
# country consistency
pc = pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c1")), on="s1").join(
    allr.rename({"entity_id": "rid", "country": "c2"}), on="rid")
print("pairs with country mismatch:", (pc["c1"] != pc["c2"]).sum())
# singleton / cardinality by country
kc = gt.join(s1, left_on="source1_entity_id", right_on="entity_id").with_columns(k.alias("k"))
print("\nby country:", kc.group_by("country").agg(pl.len(), (pl.col("k") == 0).mean().alias("singleton_rate"), pl.col("k").mean().alias("mean_k"), pl.col("k").quantile(.99).alias("p99_k"), pl.col("k").max().alias("max_k")))
# unmatched S2/S3 by country
un = allr.join(pairs, left_on="entity_id", right_on="rid", how="anti")
print("unmatched S2/S3 by country:", dict(un.group_by("country").len().iter_rows()), " total unmatched:", f"{un.height:,}")
# ID leakage check: rank correlation of numeric id parts in matched pairs
num = pairs.with_columns(pl.col("s1").str.slice(3).cast(pl.Int64).alias("a"), pl.col("rid").str.slice(3).cast(pl.Int64).alias("b"))
print("\nSpearman(id S1, id matched) =", np.corrcoef(num["a"].rank().to_numpy(), num["b"].rank().to_numpy())[0, 1])
# row-order leakage
s1r = s1.with_row_index("r1").select(pl.col("entity_id").alias("s1"), "r1")
s2r = pl.concat([s2.with_row_index("r2").select(pl.col("entity_id").alias("rid"), "r2"), s3.with_row_index("r2").select(pl.col("entity_id").alias("rid"), "r2")])
ro = pairs.join(s1r, on="s1").join(s2r, on="rid")
print("Spearman(row S1, row matched) =", np.corrcoef(ro["r1"].rank().to_numpy(), ro["r2"].rank().to_numpy())[0, 1])
print("id numeric range S1:", num["a"].min(), num["a"].max(), " S2/S3:", num["b"].min(), num["b"].max())
