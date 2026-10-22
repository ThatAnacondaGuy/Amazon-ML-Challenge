"""Candidate generation: per-country sparse TF-IDF (name view + address view + conjunctive name x address keys),
df-capped inverted index over S1, numba top-k per S2/S3 record, ratio pruning."""
import logging
import time
import zlib

import numba
import numpy as np
import polars as pl
from scipy import sparse

from . import text

log = logging.getLogger("ber")
UNK = 1 << 20          # extra hashed columns for query-only features (they only affect the query norm)

NORM_SCHEMA = {"n_full": pl.Utf8, "n_core": pl.Utf8, "n_ns": pl.Utf8, "n_tok": pl.List(pl.Utf8), "n_sk": pl.List(pl.Utf8),
               "n_init": pl.Utf8, "n_domain": pl.Boolean, "n_nonlatin": pl.Boolean, "a_full": pl.Utf8,
               "a_tok": pl.List(pl.Utf8), "a_num": pl.List(pl.Utf8), "a_hn": pl.List(pl.Utf8), "a_key": pl.Utf8}


def _key(s):
    b = s.encode()
    return (zlib.crc32(b) << 32) | zlib.crc32(b, 0x9747B28C)


def _ngrams(s, n=4):
    s = f"#{s}#"
    return [s[i:i + n] for i in range(max(1, len(s) - n + 1))]


def process_chunk(args):
    """Worker: normalize records and emit hashed sparse features. Pure function (safe under spawn)."""
    names, addrs, td = args
    if td is not None:
        text.set_translit_dict(td)
    cols = {k: [] for k in NORM_SCHEMA}
    ln, kn, la, ka = [], [], [], []
    for name, addr in zip(names, addrs):
        full, core, ns, ctoks, sks, init, dom, nonlat = text.name_fields(name)
        afull, words, nums, hn, akey = text.addr_fields(addr)
        for k, v in zip(NORM_SCHEMA, (full, core, ns, ctoks, sks, init, dom, nonlat, afull, words, nums, hn, akey)):
            cols[k].append(v)
        nf = ["n4|" + g for g in _ngrams(ns)] + ["nt|" + t for t in ctoks] + ["nk|" + t for t in sks]
        sk = sorted({s for s, t in zip(sks, ctoks) if len(t) > 1})
        aw = sorted(set(words))[:12]
        an = sorted(set(nums))[:4]
        nf += [f"xn|{a}|{b}" for a in sk for b in an] + [f"xw|{a}|{b}" for a in sk for b in aw]
        af = ["at|" + t for t in words] + ["ak|" + text.skel(t) for t in words] + ["an|" + t for t in nums]
        af += [f"xa|{a}|{b}" for a in an for b in aw]
        ln.append(len(nf)); kn.extend(_key(x) for x in nf)
        la.append(len(af)); ka.extend(_key(x) for x in af)
    return (cols, np.array(ln, np.int64), np.array(kn, np.uint64), np.array(la, np.int64), np.array(ka, np.uint64))


def featurize(pool, names, addrs, td, chunk=20000):
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], td) for i in range(0, len(names), chunk)]
    res = pool.map(process_chunk, jobs)
    norm = pl.concat([pl.DataFrame(r[0], schema=NORM_SCHEMA) for r in res]) if res else pl.DataFrame(schema=NORM_SCHEMA)
    cat = lambda i: np.concatenate([r[i] for r in res]) if res else np.zeros(0, np.int64)
    return norm, cat(1), cat(2), cat(3), cat(4)


def _csr(lens, cols, n_rows, n_cols):
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    m = sparse.csr_matrix((np.ones(len(cols), np.float32), cols.astype(np.int32), indptr), shape=(n_rows, n_cols))
    m.sum_duplicates()
    return m


def _weight(m, idf):
    m = m.tocsr(copy=True)
    m.data = ((1 + np.log(m.data)) * idf[m.indices]).astype(np.float32)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).astype(np.float32)
    norms[norms == 0] = 1
    return sparse.diags(1 / norms).dot(m).tocsr()


@numba.njit(parallel=True, fastmath=True, cache=True)
def _search(qp, qi, qd, ip, ii, idd, n_index, k, nblocks, out_i, out_s):
    nq = len(qp) - 1
    bs = (nq + nblocks - 1) // nblocks
    for b in numba.prange(nblocks):
        acc = np.zeros(n_index, np.float32)
        touched = np.empty(n_index, np.int32)
        s0 = b * bs
        s1 = min(nq, s0 + bs)
        for q in range(s0, s1):
            nt = 0
            for p in range(qp[q], qp[q + 1]):
                f = qi[p]
                wq = qd[p]
                for pp in range(ip[f], ip[f + 1]):
                    d = ii[pp]
                    if acc[d] == 0.0:
                        touched[nt] = d
                        nt += 1
                    acc[d] += wq * idd[pp]
            if nt > 0:
                sc = np.empty(nt, np.float32)
                for t in range(nt):
                    sc[t] = acc[touched[t]]
                kk = min(k, nt)
                order = np.argsort(-sc)[:kk]
                for t in range(kk):
                    out_i[q, t] = touched[order[t]]
                    out_s[q, t] = sc[order[t]]
                for t in range(nt):
                    acc[touched[t]] = 0.0


class Index:
    def __init__(self, lens_n, keys_n, lens_a, keys_a, n, cap, wa):
        self.N, self.wa = n, wa
        self.vocab = np.unique(np.concatenate([keys_n, keys_a]))
        self.V = len(self.vocab)
        C = self.V + UNK
        In = _csr(lens_n, np.searchsorted(self.vocab, keys_n), n, C)
        Ia = _csr(lens_a, np.searchsorted(self.vocab, keys_a), n, C)
        df = np.bincount(In.indices, minlength=C) + np.bincount(Ia.indices, minlength=C)
        self.idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        self.Wn, self.Wa = _weight(In, self.idf), _weight(Ia, self.idf)
        W = (self.Wn * np.float32(np.sqrt(1 - wa)) + self.Wa * np.float32(np.sqrt(wa))).tocsc()
        W.sort_indices()
        keep = df <= cap
        lens = np.diff(W.indptr) * keep
        self.ip = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        mask = np.repeat(keep, np.diff(W.indptr))
        self.ii = W.indices[mask].astype(np.int32)
        self.idd = W.data[mask].astype(np.float32)
        self.pruned_share = 1 - mask.mean() if len(mask) else 0.0

    def map_cols(self, keys):
        ids = np.searchsorted(self.vocab, keys)
        idc = np.minimum(ids, self.V - 1)
        known = self.vocab[idc] == keys
        return np.where(known, idc, self.V + (keys % np.uint64(UNK)).astype(np.int64))

    def query(self, lens_n, keys_n, lens_a, keys_a, k, ratio):
        nq, C = len(lens_n), self.V + UNK
        Qn = _weight(_csr(lens_n, self.map_cols(keys_n), nq, C), self.idf)
        Qa = _weight(_csr(lens_a, self.map_cols(keys_a), nq, C), self.idf)
        Q = (Qn * np.float32(np.sqrt(1 - self.wa)) + Qa * np.float32(np.sqrt(self.wa))).tocsr()
        oi = -np.ones((nq, k), np.int32)
        os_ = np.zeros((nq, k), np.float32)
        _search(Q.indptr.astype(np.int64), Q.indices.astype(np.int64), Q.data.astype(np.float32),
                self.ip, self.ii, self.idd, self.N, k, 40, oi, os_)
        top1 = os_[:, :1]
        rank = np.broadcast_to(np.arange(k, dtype=np.int16), oi.shape)
        keep = (oi >= 0) & ((rank == 0) | (os_ >= ratio * top1))
        q_loc = np.nonzero(keep)[0].astype(np.int64)
        s_idx = oi[keep].astype(np.int64)
        cos_n = np.asarray(Qn[q_loc].multiply(self.Wn[s_idx]).sum(1)).ravel().astype(np.float32)
        cos_a = np.asarray(Qa[q_loc].multiply(self.Wa[s_idx]).sum(1)).ravel().astype(np.float32)
        return dict(q=q_loc, s=s_idx, score=os_[keep], rank=rank[keep].astype(np.float32),
                    top1=np.repeat(top1[:, 0], keep.sum(1)), cos_n=cos_n, cos_a=cos_a)


def block_partition(pool, s1, s23, td, k=10, ratio=0.5, cap=2000, wa=0.5, chunk=400_000):
    """Returns (s1_norm, q_norm, pairs DataFrame) for one country partition."""
    t0 = time.time()
    s1n, ln, kn, la, ka = featurize(pool, s1["business_name"].to_list(), s1["business_address"].to_list(), td)
    idx = Index(ln, kn, la, ka, s1.height, cap, wa)
    del kn, ka
    log.info("  index: %d S1, vocab %d, pruned-posting share %.3f (%.0fs)", s1.height, idx.V, idx.pruned_share, time.time() - t0)
    names, addrs = s23["business_name"].to_list(), s23["business_address"].to_list()
    qparts, pparts = [], []
    t1 = time.time()
    for c0 in range(0, len(names), chunk):
        qn, ln, kn, la, ka = featurize(pool, names[c0:c0 + chunk], addrs[c0:c0 + chunk], td)
        r = idx.query(ln, kn, la, ka, k, ratio)
        r["q"] = r["q"] + c0
        qparts.append(qn)
        pparts.append(pl.DataFrame(r))
    q_norm = pl.concat(qparts) if qparts else pl.DataFrame(schema=NORM_SCHEMA)
    pairs = pl.concat(pparts) if pparts else pl.DataFrame()
    nq = max(1, len(names))
    log.info("  retrieval: %d queries in %.0fs (%.0f q/s), %d pairs (%.2f/query)", len(names), time.time() - t1,
             len(names) / max(1e-9, time.time() - t1), pairs.height, pairs.height / nq)
    return s1n, q_norm, pairs


NORM_KEEP = ["n_full", "n_core", "n_ns", "a_full", "a_hn", "a_key"]


def norm_chunk(args):
    """Worker: normalized fields only (no hashing) — used to rebuild stage-2 inputs cheaply."""
    names, addrs, td = args
    text.set_translit_dict(td)
    rows = []
    for name, addr in zip(names, addrs):
        full, core, ns, *_ = text.name_fields(name)
        afull, _, _, hn, akey = text.addr_fields(addr)
        rows.append((full, core, ns, afull, hn, akey))
    return rows


def normalize_only(pool, names, addrs, td, chunk=20000):
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], td) for i in range(0, len(names), chunk)]
    rows = [r for part in pool.map(norm_chunk, jobs) for r in part]
    return pl.DataFrame(rows, schema={"n_full": pl.Utf8, "n_core": pl.Utf8, "n_ns": pl.Utf8, "a_full": pl.Utf8,
                                      "a_hn": pl.List(pl.Utf8), "a_key": pl.Utf8}, orient="row")
