"""Unsupervised injected-noise token detector: tokens over-represented in S2/S3 names vs S1 names of the SAME country.
No labels. For train countries, compare against the labelled insertion rate (token present in an S2/S3 name but absent
from its true S1 name)."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from load import load_source, load_gt
from probe_lib import base_clean
def toks(df):
    t = df.select(pl.col("business_name").map_elements(lambda s: sorted(set(re.sub(r"[^a-z0-9]+", " ", base_clean(s)).split())), return_dtype=pl.List(pl.Utf8)).alias("t"))
    return t.explode("t").group_by("t").len()
for split, c in [("test", "France"), ("train", "US"), ("train", "India")]:
    s1 = load_source(split, 1).filter(pl.col("country") == c)
    s23 = pl.concat([load_source(split, k).filter(pl.col("country") == c) for k in (2, 3)])
    a, b = toks(s1).rename({"len": "d1"}), toks(s23).rename({"len": "d23"})
    m = b.join(a, on="t", how="left").with_columns(pl.col("d1").fill_null(0))
    m = m.with_columns(((pl.col("d23") / s23.height) / ((pl.col("d1") + 5) / s1.height)).alias("ratio")).filter(pl.col("d23") >= 2000)
    top = m.sort("ratio", descending=True).head(22)
    print(f"[{split}/{c}] tokens over-represented in S2/S3 vs S1 (ratio, S2/S3 df):")
    print("   ", ", ".join(f"{t}({r:.0f}x,{d:,})" for t, r, d in zip(top["t"], top["ratio"], top["d23"])))
