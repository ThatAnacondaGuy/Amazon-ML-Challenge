import sys, re, collections
sys.path.insert(0, "/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad")
from load import *; from norm_probe_lib import *
import polars as pl, numpy as np
from rapidfuzz import fuzz
s1tr = load_source("train", 1); s1te = load_source("test", 1)
s23 = pl.concat([load_source("train", 2), load_source("train", 3)])
# vocabulary repetitiveness of S1 names per country (train vs test coverage)
def toks(df):
    return collections.Counter(t for s in df["business_name"].to_list() for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
for c in ["US", "India"]:
    vtr = toks(s1tr.filter(pl.col("country") == c)); vte = toks(s1te.filter(pl.col("country") == c))
    tot = sum(vte.values()); cov = sum(v for t, v in vte.items() if t in vtr)
    print(f"{c}: train S1 vocab={len(vtr):,} test S1 vocab={len(vte):,}; test token occurrences covered by train vocab={cov/tot*100:.2f}%; "
          f"test vocab types unseen in train={sum(1 for t in vte if t not in vtr):,}")
vfr = toks(s1te.filter(pl.col("country") == "France"))
print(f"France test S1 vocab={len(vfr):,}; top 40:", vfr.most_common(40))
# zero-name-overlap true pairs (latin)
gt = load_gt()
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
p = pairs.join(s1tr.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"), "country"), on="s1") \
         .join(s23.select(pl.col("entity_id").alias("rid"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")), on="rid") \
         .filter(~pl.col("n2").str.contains(r"[ऀ-෿]")).sample(100000, seed=2)
rows = [(a, b, x, y) for a, b, x, y in zip(p["n1"], p["n2"], p["a1"], p["a2"]) if feats(a, b)[1] == 0]
print(f"\nzero skeleton overlap among latin true pairs: {len(rows)/p.height*100:.1f}%  examples:")
for r in rows[:25]: print("   ", r[0], "||", r[1], "||", r[2][:60], "||", r[3][:60])
