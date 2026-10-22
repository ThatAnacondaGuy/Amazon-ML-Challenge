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
