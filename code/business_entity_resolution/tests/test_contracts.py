"""Unit tests for the metric, ingest unquoting, normalization edge cases and the output writer contract.
Run: PYTHONPATH=src python -m pytest -q tests   (or: PYTHONPATH=src python tests/test_contracts.py)"""
import os
import sys
import tempfile

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber import decide, io, text  # noqa: E402


def test_metric_readme_example():
    assert round(decide.f05_single({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"}), 3) == 0.714


def test_metric_singletons_and_empty():
    assert decide.f05_single(set(), set()) == 1.0
    assert decide.f05_single(set(), {"S2-1"}) == 0.0
    assert decide.f05_single({"S2-1"}, set()) == 0.0
    # vectorised version agrees with the reference
    T, P, TP = [0, 0, 2, 3], [0, 1, 3, 0], [0, 0, 2, 0]
    ref = np.mean([1.0, 0.0, decide.f05_single({"a", "b"}, {"a", "b", "c"}), 0.0])
    assert abs(decide.f05_macro(T, P, TP) - ref) < 1e-12


def test_unquote():
    df = pl.DataFrame({"business_name": ['"""ehpad Club SAS"', '"Fédération du ""ehpad"', "Plain"]})
    out = df.with_columns(io._unquote("business_name"))["business_name"].to_list()
    assert out == ['"ehpad Club SAS', 'Fédération du "ehpad', "Plain"]


def test_argmax_exclusivity():
    q = np.array([0, 0, 1]); s = np.array([5, 6, 5]); p = np.array([0.8, 0.9, 0.95]); sc = np.array([0.5, 0.4, 0.3])
    keep = decide.argmax_assign(q, s, p, sc, 0.7)
    assert keep.tolist() == [False, True, True]


def test_normalization_cases():
    assert text.name_fields("भारत एनर्जी प्राइवेट लिमिटेड", {"praivet": "private"})[3] == ["bharat", "enarji"]
    assert text.name_fields("Laurel Trading Limited 5ervices", {})[3] == ["laurel", "trading", "services"]
    assert text.name_fields("3M India", {})[3] == ["3m", "india"]
    assert text.name_fields('"ehpad Club S.A.S', {})[3] == ["ehpad", "club"]
    assert text.addr_fields("D-904 Neel Padam")[3] == ["d904"]
    assert text.addr_fields("PUNE, SUPER MALL, <NULL>, Maharashtra")[1] == ["pune", "super", "mall", "maharashtra"]


def test_writer_contract():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.tsv")
        pairs = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-1"], "rid": ["S3-9", "S2-5", "S3-9"]})
        io.write_id_lists(path, ["source1_entity_id", "matched_entity_ids"], ["S1-1", "S1-2"], pairs)
        raw = open(path, "rb").read()
        assert b"\r" not in raw and b'"' not in raw
        assert raw.decode() == "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5,S3-9\nS1-2\t\n"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
