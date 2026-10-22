"""Ingest (raw tab split + RFC-4180 unquote) and submission writers."""
import logging
import os

import polars as pl

log = logging.getLogger("ber")

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TEXT_COLS = ["business_name", "business_address", "country"]


def _unquote(col):
    """Fields wrapped in double quotes use CSV escaping ("" -> "); tabs never occur inside fields."""
    c = pl.col(col)
    quoted = c.str.starts_with('"') & c.str.ends_with('"') & (c.str.len_chars() >= 2)
    inner = c.str.slice(1, c.str.len_chars() - 2).str.replace_all('""', '"', literal=True)
    return pl.when(quoted).then(inner).otherwise(c).alias(col)


def _read_raw(path):
    try:
        return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                           missing_utf8_is_empty_string=True)
    except pl.exceptions.ComputeError:
        # malformed rows (wrong field count): salvage by merging surplus fields into the address
        log.warning("%s: malformed rows, falling back to manual parse", path)
        rows, bad = [], 0
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    bad += 1
                    continue
                if len(parts) > 4:
                    parts = [parts[0], parts[1], " ".join(parts[2:-1]), parts[-1]]
                    bad += 1
                parts += [""] * (4 - len(parts))
                rows.append(parts)
        log.warning("%s: %d rows salvaged/skipped", path, bad)
        return pl.DataFrame(rows, schema=header, orient="row")


def load_source(data_dir, split, k):
    path = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
    df = _read_raw(path)
    if df.columns != COLUMNS:
        raise ValueError(f"{path}: unexpected header {df.columns}")
    df = df.with_columns([pl.col(c).fill_null("") for c in COLUMNS])
    df = df.with_columns([_unquote(c) for c in TEXT_COLS])
    df = df.with_columns(pl.col("entity_id").str.strip_chars(), pl.col("country").str.strip_chars())
    n = df.height
    df = df.unique(subset="entity_id", keep="first", maintain_order=True)
    if df.height != n:
        log.warning("%s: dropped %d duplicate ids", path, n - df.height)
    wrong = df.filter(~pl.col("entity_id").str.starts_with(f"S{k}-")).height
    if wrong:
        log.warning("%s: %d ids without S%d- prefix", path, wrong, k)
    return df


def load_split(data_dir, split):
    s1 = load_source(data_dir, split, 1)
    s23 = pl.concat([load_source(data_dir, split, 2), load_source(data_dir, split, 3)])
    return s1, s23


def load_gt_pairs(data_dir):
    gt = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator="\t", quote_char=None,
                     infer_schema=False, missing_utf8_is_empty_string=True)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("rid")))


def write_id_lists(path, header, s1_ids, pairs):
    """pairs: DataFrame(s1, rid). One row per S1 in s1_ids (empty list allowed), LF endings, no quoting."""
    lists = (pairs.unique().sort(["s1", "rid"]).group_by("s1", maintain_order=True)
             .agg(pl.col("rid").str.join(",").alias("ids")))
    out = (pl.DataFrame({"s1": s1_ids}).join(lists, on="s1", how="left")
           .with_columns(pl.col("ids").fill_null("")))
    assert out.height == len(s1_ids) and out["s1"].n_unique() == len(s1_ids)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s, ids in out.iter_rows():
            f.write(f"{s}\t{ids}\n")
    log.info("wrote %s (%d rows, %d non-empty)", path, out.height, (out["ids"] != "").sum())
