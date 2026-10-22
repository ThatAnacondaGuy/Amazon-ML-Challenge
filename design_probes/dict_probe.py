import sys, os, re, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
from collections import Counter, defaultdict
from rapidfuzz import fuzz
from load import load_source, load_gt
from probe_lib import base_clean, skel, name_tokens
IND = re.compile(r"[ऀ-෿]")
toks = lambda s: re.sub(r"[^a-z0-9]+", " ", base_clean(s)).split()
s1 = load_source("train", 1).filter(pl.col("country") == "India")
s23 = pl.concat([load_source("train", 2), load_source("train", 3)]).filter(pl.col("country") == "India")
ind = s23.filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))
gt = load_gt()
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
P = ind.join(pairs, left_on="entity_id", right_on="rid").join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1")), on="s1")
held = np.array([zlib.crc32(x.encode()) % 5 == 0 for x in P["s1"].to_list()])
print(f"indic-name true pairs: {P.height:,} (learn {(~held).sum():,} / held-out {held.sum():,})")
def align(r, l):
    """greedy one-to-one alignment of romanized tokens r to latin tokens l by skeleton+string similarity"""
    if len(r) == len(l): return list(zip(r, l))
    cands = sorted(((0.5 * fuzz.ratio(skel(a), skel(b)) + 0.5 * fuzz.ratio(a, b), i, j) for i, a in enumerate(r) for j, b in enumerate(l)), reverse=True)
    used_i, used_j, out = set(), set(), []
    for sc, i, j in cands:
        if sc < 50 or i in used_i or j in used_j: continue
        used_i.add(i); used_j.add(j); out.append((r[i], l[j]))
    return out
cnt = defaultdict(Counter)
qn, n1 = P["business_name"].to_list(), P["n1"].to_list()
for k in np.where(~held)[0]:
    for a, b in align(toks(qn[k]), toks(n1[k])): cnt[a][b] += 1
D = {}
for a, c in cnt.items():
    tot = sum(c.values()); b, n = c.most_common(1)[0]
    if tot >= 3 and n / tot >= 0.6: D[a] = b
print(f"dictionary entries: {len(D):,}; examples:", list(D.items())[:25])
def mapped(s): return " ".join(D.get(t, t) for t in toks(s))
cov = []; before = []; after = []
for k in np.where(held)[0]:
    tt = toks(qn[k]); cov += [t in D for t in tt]
    L = " ".join(toks(n1[k]))
    before.append(fuzz.token_set_ratio(" ".join(tt), L)); after.append(fuzz.token_set_ratio(mapped(qn[k]), L))
before, after = np.array(before), np.array(after)
print(f"held-out: token coverage={np.mean(cov)*100:.1f}%  token_set_ratio p10/p50: before={np.percentile(before,10):.0f}/{np.median(before):.0f} after={np.percentile(after,10):.0f}/{np.median(after):.0f}  exact-equal after={np.mean(after==100)*100:.1f}%")
# test-set coverage of the learned dictionary (unlabelled test India indic names)
te = pl.concat([load_source("test", 2), load_source("test", 3)]).filter((pl.col("country") == "India") & pl.col("business_name").str.contains(r"[ऀ-෿]")).sample(100000, seed=1)
tc = [t in D for s in te["business_name"].to_list() for t in toks(s)]
print(f"TEST India indic names: token coverage by train-learned dictionary = {np.mean(tc)*100:.1f}%")
