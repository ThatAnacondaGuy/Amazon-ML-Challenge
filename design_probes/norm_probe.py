import sys, re, unicodedata, random
sys.path.insert(0, "/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad")
from load import *; from translit import romanize_words
import polars as pl, numpy as np
from rapidfuzz import fuzz
ABBR = {"pvt": "private", "pra": "private", "ltd": "limited", "li": "limited", "co": "company", "corp": "corporation",
        "inc": "incorporated", "intl": "international", "mfg": "manufacturing", "svcs": "services", "bros": "brothers"}
LEGAL = {"private", "limited", "llp", "llc", "incorporated", "corporation", "company", "lp", "plc", "pllc", "the", "and", "of"}
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    if re.search(r"[ऀ-෿]", s): s = romanize_words(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    s = re.sub(r"https?://|www\.|\.(com|in|net|org|co)\b", " ", s)
    toks = [ABBR.get(t, t) for t in re.sub(r"[^a-z0-9 ]", " ", s).split()]
    return [t for t in toks if t not in LEGAL] or toks
def skel(t):
    t = t.replace("ph", "f").replace("x", "ks").replace("sh", "s").replace("ck", "k").replace("q", "k").replace("w", "v").replace("z", "j")
    t = re.sub(r"c(?=[eiy])", "s", t).replace("c", "k"); t = re.sub(r"g(?=[eiy])", "j", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t[0] + re.sub(r"[aeiouyh]", "", t[1:]) if t else t
def feats(a, b):
    na, nb = norm(a), norm(b); sa, sb = {skel(t) for t in na}, {skel(t) for t in nb}
    return (fuzz.token_sort_ratio(" ".join(na), " ".join(nb)) / 100, len(sa & sb) / max(1, len(sa | sb)))
s1 = load_source("train", 1); s23 = pl.concat([load_source("train", 2), load_source("train", 3)])
gt = load_gt()
pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
p = pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), "country"), on="s1") \
         .join(s23.select(pl.col("entity_id").alias("rid"), pl.col("business_name").alias("n2")), on="rid")
p = p.with_columns(pl.col("n2").str.contains(r"[ऀ-෿]").alias("indic"))
rng = random.Random(0)
for label, sub in [("US latin", p.filter(pl.col("country") == "US")), ("India latin", p.filter((pl.col("country") == "India") & ~pl.col("indic"))),
                   ("India indic-script", p.filter(pl.col("indic")))]:
    smp = sub.sample(20000, seed=1)
    f = np.array([feats(a, b) for a, b in zip(smp["n1"], smp["n2"])])
    # random negatives: same country, shuffled S1 names
    n1s = smp["n1"].to_list(); rng.shuffle(n1s)
    g = np.array([feats(a, b) for a, b in zip(n1s, smp["n2"])])
    print(f"{label:20s} TRUE: tsr p10/p50={np.percentile(f[:,0],10):.2f}/{np.median(f[:,0]):.2f} skelJac p10/p50={np.percentile(f[:,1],10):.2f}/{np.median(f[:,1]):.2f} "
          f"frac skelJac=0: {(f[:,1]==0).mean()*100:4.1f}% | RANDOM: tsr p50/p99={np.median(g[:,0]):.2f}/{np.percentile(g[:,0],99):.2f} skelJac p99={np.percentile(g[:,1],99):.2f}")
# show the hardest indic true pairs
smp = p.filter(pl.col("indic")).sample(3000, seed=5)
res = sorted(((feats(a, b)[0], a, b, " ".join(norm(b))) for a, b in zip(smp["n1"], smp["n2"])))[:12]
for r in res: print(f"  {r[0]:.2f} | {r[1]} | {r[2]} -> {r[3]}")
