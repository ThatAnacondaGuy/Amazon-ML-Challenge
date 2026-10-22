"""Learn the romanized-token -> Latin-token dictionary from TRAIN India pairs, EXCLUDING the probe's validation
S1 entities (same selection as probe_model.py: default_rng(0) permutation, positions [60000:90000]) -> no leakage."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from collections import Counter, defaultdict
from rapidfuzz import fuzz
from load import load_source, load_gt
from probe_lib import base_clean, skel
SP = os.path.dirname(os.path.abspath(__file__))
toks = lambda s: re.sub(r"[^a-z0-9]+", " ", base_clean(s)).split()
s1 = load_source("train", 1).filter(pl.col("country") == "India")
N = s1.height; perm = np.random.default_rng(0).permutation(N); va_ids = set(s1["entity_id"].gather(perm[60000:90000]).to_list())
s23 = pl.concat([load_source("train", 2), load_source("train", 3)]).filter(pl.col("country") == "India")
gt = load_gt()
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
P = (s23.filter(pl.col("business_name").str.contains(r"[ऀ-෿]")).join(pairs, left_on="entity_id", right_on="rid")
       .join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1")), on="s1")
       .filter(~pl.col("s1").is_in(list(va_ids))))
def align(r, l):
    if len(r) == len(l): return list(zip(r, l))
    c = sorted(((0.5 * fuzz.ratio(skel(a), skel(b)) + 0.5 * fuzz.ratio(a, b), i, j) for i, a in enumerate(r) for j, b in enumerate(l)), reverse=True)
    ui, uj, out = set(), set(), []
    for sc, i, j in c:
        if sc < 50 or i in ui or j in uj: continue
        ui.add(i); uj.add(j); out.append((r[i], l[j]))
    return out
cnt = defaultdict(Counter)
for a, b in zip(P["business_name"].to_list(), P["n1"].to_list()):
    for x, y in align(toks(a), toks(b)): cnt[x][y] += 1
D = {a: c.most_common(1)[0][0] for a, c in cnt.items() if sum(c.values()) >= 3 and c.most_common(1)[0][1] / sum(c.values()) >= 0.6}
json.dump(D, open(f"{SP}/translit_dict.json", "w"))
print(f"learned {len(D)} entries from {P.height:,} indic pairs (val entities excluded: {len(va_ids):,})")
