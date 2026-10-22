"""Shared loader for the profiling probes: raw tab split, then RFC-4180 unquote per field."""
import polars as pl
BASE = "/Users/devopsdreamer/Desktop/Amazon ML Challenge/student_resource/dataset"

def unquote(col):
    c = pl.col(col)
    quoted = c.str.starts_with('"') & c.str.ends_with('"') & (c.str.len_chars() >= 2)
    return pl.when(quoted).then(c.str.slice(1, c.str.len_chars() - 2).str.replace_all('""', '"', literal=True)).otherwise(c).alias(col)

def load_source(split, k):
    df = pl.read_csv(f"{BASE}/{split}/{split}_source{k}.tsv", separator="\t", quote_char=None,
                     infer_schema=False, missing_utf8_is_empty_string=True)
    return df.with_columns([unquote(c) for c in ["business_name", "business_address", "country"]])

def load_gt():
    gt = pl.read_csv(f"{BASE}/train/train_ground_truth.tsv", separator="\t", quote_char=None,
                     infer_schema=False, missing_utf8_is_empty_string=True)
    return gt
