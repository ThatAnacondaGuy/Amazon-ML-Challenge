import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad")
from load import *
import polars as pl, random
s1 = load_source("train", 1); s2 = load_source("train", 2); s3 = load_source("train", 3)
gt = load_gt()
rec = {r[0]: (r[1], r[2]) for df in (s2, s3) for r in df.select("entity_id", "business_name", "business_address").iter_rows()}
s1d = {r[0]: (r[1], r[2], r[3]) for r in s1.iter_rows()}
random.seed(7)
rows = gt.sample(40, seed=11).iter_rows()
shown = {"US": 0, "India": 0}
for sid, m in rows:
    n, a, c = s1d[sid]
    if shown[c] >= 7: continue
    shown[c] += 1
    print(f"\n### {sid} [{c}]  {n}  |  {a}")
    for rid in (m.split(",") if m else []):
        rn, ra = rec[rid]
        print(f"     {rid[:2]}  {rn}  |  {ra}")
