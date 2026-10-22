"""Resources learned from the provided TRAINING data only (no external data)."""
import logging
import re
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from . import text

log = logging.getLogger("ber")


def _toks(s):
    return re.sub(r"[^a-z0-9]+", " ", text.base_clean(s)).split()


def _align(r, l):
    if len(r) == len(l):
        return list(zip(r, l))
    cands = sorted(((0.5 * fuzz.ratio(text.skel(a), text.skel(b)) + 0.5 * fuzz.ratio(a, b), i, j)
                    for i, a in enumerate(r) for j, b in enumerate(l)), reverse=True)
    ui, uj, out = set(), set(), []
    for sc, i, j in cands:
        if sc < 50 or i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); out.append((r[i], l[j]))
    return out


def learn_translit_dict(s1, s23, gt_pairs, exclude_s1=(), min_support=3, min_precision=0.6):
    """Romanized-token -> Latin-token dictionary aligned on true pairs whose S2/S3 name is in an Indic script."""
    P = (s23.filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))
            .join(gt_pairs, left_on="entity_id", right_on="rid")
            .join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1")), on="s1")
            .filter(~pl.col("s1").is_in(list(exclude_s1))))
    cnt = defaultdict(Counter)
    for a, b in zip(P["business_name"].to_list(), P["n1"].to_list()):
        for x, y in _align(_toks(a), _toks(b)):
            cnt[x][y] += 1
    d = {}
    for a, c in cnt.items():
        tot = sum(c.values()); b, n = c.most_common(1)[0]
        if tot >= min_support and n / tot >= min_precision:
            d[a] = b
    log.info("transliteration dictionary: %d entries from %d non-Latin true pairs", len(d), P.height)
    return d
