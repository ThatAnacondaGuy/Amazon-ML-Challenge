"""Unsupervised, language-agnostic abbreviation learner: token substitutions observed in identical local contexts
(left token, right token, right+1 token) across records of ONE country; kept only if abbreviation-shaped."""
import sys, os, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from load import load_source
from probe_lib import base_clean
C = sys.argv[1]; SPLIT = sys.argv[2]; FIELD = sys.argv[3]
t0 = time.time()
df = pl.concat([load_source(SPLIT, k).filter(pl.col("country") == C).select(FIELD) for k in (1, 2, 3)])
docs = [re.sub(r"[^a-z0-9]+", " ", base_clean(s).replace("'", "").replace("’", "")).split() for s in df[FIELD].to_list()]
rows = []
for d in docs:
    d2 = ["^"] + d + ["$", "$"]
    for i in range(1, len(d2) - 2):
        t = d2[i]
        if t.isalpha():
            rows.append((f"{d2[i-1]}|{d2[i+1]}|{d2[i+2]}", t))
sig = pl.DataFrame(rows, schema=["sig", "tok"], orient="row").unique()
g = sig.group_by("sig").agg(pl.col("tok")).filter(pl.col("tok").list.len().is_between(2, 20))
pairs = (g.explode("tok").join(g.explode("tok"), on="sig").filter(pl.col("tok") < pl.col("tok_right"))
          .group_by(["tok", "tok_right"]).len().filter(pl.col("len") >= 20))
def is_abbr(s, l):
    if len(s) >= len(l) or s[0] != l[0]: return False
    if l.startswith(s): return True
    it = iter(l); return all(c in it for c in s)
cand = []
for a, b, n in pairs.iter_rows():
    s, l = (a, b) if len(a) < len(b) else (b, a)
    if is_abbr(s, l): cand.append((s, l, n))
best = {}
for s, l, n in sorted(cand, key=lambda x: -x[2]):
    best.setdefault(s, (l, n))
out = sorted(best.items(), key=lambda kv: -kv[1][1])
print(f"[{C}/{SPLIT}/{FIELD}] records={len(docs):,} learned {len(out)} abbreviation rules in {time.time()-t0:.0f}s:")
print("   ", ", ".join(f"{s}->{l}({n})" for s, (l, n) in out[:45]))
