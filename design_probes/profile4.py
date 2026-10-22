import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad")
from load import *
import polars as pl
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(200)
LEGAL = r"\b(pvt|private|ltd|limited|llp|llc|inc|incorporated|corp|corporation|co|company|lp|plc|pllc|sas|sarl|sasu|eurl|sa|the|and|of)\b"
def crude(col):
    return (pl.col(col).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(LEGAL, " ")
            .str.replace_all(r"\s+", " ").str.strip_chars().alias("cn"))
out = {}
for split in ["train", "test"]:
    s1 = load_source(split, 1).with_columns(crude("business_name"))
    s23 = pl.concat([load_source(split, 2), load_source(split, 3)]).with_columns(crude("business_name"))
    print(f"\n==== {split}")
    for c in s1["country"].unique().sort().to_list():
        a = s1.filter(pl.col("country") == c); b = s23.filter(pl.col("country") == c)
        vc = a.group_by("cn").len()
        print(f"  {c:7s} S1 n={a.height:>9,} uniq crude names={vc.height:>9,} ({vc.height/a.height*100:5.1f}%)  "
              f"S1 in name-collision group={a.join(vc.filter(pl.col('len')>1), on='cn', how='semi').height/a.height*100:5.1f}%  "
              f"largest group={vc['len'].max()}  S2/S3 uniq={b['cn'].n_unique()/b.height*100:5.1f}%")
        print("      top S1 names:", vc.sort("len", descending=True).head(6).rows())
    if split == "train":
        gt = load_gt()
        pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                   .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
        p = pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("cn").alias("n1"), "country"), on="s1") \
                 .join(s23.select(pl.col("entity_id").alias("rid"), pl.col("cn").alias("n2"), pl.col("business_name").alias("raw2")), on="rid")
        latin = p.filter(~p["raw2"].str.contains(r"[^\x00-\x7FÀ-ɏ]"))
        print(f"  true pairs: crude-name exact equal = {(latin['n1']==latin['n2']).mean()*100:.1f}% (latin-script S2/S3 only, n={latin.height:,})")
        # distractors: unmatched S2/S3 whose crude name equals some S1 crude name in same country
        un = s23.join(pairs, left_on="entity_id", right_on="rid", how="anti")
        s1n = s1.select("cn", "country").unique()
        print(f"  unmatched S2/S3: {un.height:,}; with crude name == some S1 name (same country): "
              f"{un.join(s1n, on=['cn','country'], how='semi').height/un.height*100:.1f}%")
        matched = s23.join(pairs, left_on="entity_id", right_on="rid", how="semi")
        print(f"  matched   S2/S3: {matched.height:,}; with crude name == some S1 name: "
              f"{matched.join(s1n, on=['cn','country'], how='semi').height/matched.height*100:.1f}%")
        print("  sample unmatched:", un.sample(12, seed=3).select("business_name", "business_address", "country").rows())
