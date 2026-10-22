import sys, time, unicodedata, collections
sys.path.insert(0, "/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad")
from load import *
import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(200)

def script_of(s):
    """Dominant script of letters in s (by unicode name prefix)."""
    if not s: return "EMPTY"
    c = collections.Counter()
    for ch in s:
        if ch.isalpha():
            try: c[unicodedata.name(ch).split()[0]] += 1
            except ValueError: c["UNNAMED"] += 1
    if not c: return "NOLETTERS"
    top = c.most_common(2)
    return top[0][0] if len(top) == 1 or top[0][1] > 3 * top[1][1] else top[0][0] + "+" + top[1][0]

for split, ks in [("train", [1, 2, 3]), ("test", [1, 2, 3])]:
    for k in ks:
        t = time.time()
        df = load_source(split, k)
        n = df.height
        print(f"\n=========== {split} source{k}: rows={n:,} unique_ids={df['entity_id'].n_unique():,} load={time.time()-t:.1f}s")
        print("  id prefix ok:", df.filter(~pl.col("entity_id").str.starts_with(f"S{k}-")).height == 0)
        for col in ["business_name", "business_address", "country"]:
            s = df[col]
            print(f"  {col:17s} empty={ (s.str.strip_chars()=='').sum():>9,} ({(s.str.strip_chars()=='').mean()*100:5.2f}%)  "
                  f"len chars p50/p90/p99/max={s.str.len_chars().quantile(.5):.0f}/{s.str.len_chars().quantile(.9):.0f}/{s.str.len_chars().quantile(.99):.0f}/{s.str.len_chars().max()}")
        print("  countries:", dict(df.group_by("country").len().sort("len", descending=True).iter_rows()))
        # scripts on a sample
        smp = df.sample(min(n, 200_000), seed=1)
        for col in ["business_name", "business_address"]:
            cnt = collections.Counter(zip(smp["country"].to_list(), map(script_of, smp[col].to_list())))
            tot = collections.Counter(smp["country"].to_list())
            out = {f"{c}:{s}": f"{v/tot[c]*100:.1f}%" for (c, s), v in cnt.most_common(14)}
            print(f"  {col} script (200k sample):", out)
        nm = df["business_name"]
        print("  name patterns: @handle=%.2f%% domain=%.2f%% all_upper=%.1f%% has_digit=%.1f%%" % (
            nm.str.starts_with("@").mean()*100,
            nm.str.contains(r"(?i)\.(com|in|net|org|co|fr|biz|info)\b|www\.").mean()*100,
            (nm == nm.str.to_uppercase()).mean()*100,
            nm.str.contains(r"\d").mean()*100))
        ad = df["business_address"]
        print("  address patterns: 6digit(PIN)=%.1f%% 5digit(ZIP)=%.1f%% all_upper=%.1f%% 'near'=%.1f%%" % (
            ad.str.contains(r"(?:^|\D)\d{6}(?:\D|$)").mean()*100,
            ad.str.contains(r"(?:^|\D)\d{5}(?:\D|$)").mean()*100,
            (ad == ad.str.to_uppercase()).mean()*100,
            ad.str.contains(r"(?i)\bnear\b|\bopp\b|\bopposite\b|\bbehind\b").mean()*100))
        del df
