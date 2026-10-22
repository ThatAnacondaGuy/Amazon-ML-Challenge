"""Design probe: blocking recall@k vs cost on REAL-scale index (one country), sampled queries.

Usage: probe_block.py <country> <n_matched_queries> <n_unmatched_queries>
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, polars as pl, numba
from multiprocessing import Pool
from scipy import sparse
from load import load_source, load_gt
from probe_lib import name_tokens, addr_tokens, skel, char_ngrams, h, NB

COUNTRY = sys.argv[1] if len(sys.argv) > 1 else None
try:
    NQM = int(sys.argv[2]); NQU = int(sys.argv[3])
except (IndexError, ValueError):
    NQM = NQU = 0


NG = int(os.environ.get("NG", "3")); CONJ = int(os.environ.get("CONJ", "0"))
def featurize(rows):
    """rows: list of (name, address) -> list of (name_feats, addr_feats) as hashed int lists."""
    out = []
    for name, addr in rows:
        toks, core = name_tokens(name)
        joined = "".join(core)
        nf = [h("n3|" + g) for g in char_ngrams(joined, NG)]
        nf += [h("nt|" + t) for t in core] + [h("nk|" + skel(t)) for t in core]
        words, nums = addr_tokens(addr)
        af = [h("at|" + t) for t in words] + [h("ak|" + skel(t)) for t in words] + [h("an|" + t) for t in nums]
        if CONJ:
            sk = sorted({skel(t) for t in core if len(t) > 1})
            aw = sorted(set(words))[:12]; an = sorted(set(nums))[:4]
            nf += [h("xn|" + a + "|" + b) for a in sk for b in an]          # name x house number
            nf += [h("xw|" + a + "|" + b) for a in sk for b in aw]          # name x locality/street word
            af += [h("xa|" + a + "|" + b) for a in an for b in aw]          # house number x street word (DBA path)
        out.append((nf, af))
    return out


def par_featurize(names, addrs, procs=10):
    rows = list(zip(names, addrs))
    chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    with Pool(procs) as pool:
        res = pool.map(featurize, chunks)
    return [x for r in res for x in r]


def to_csr(feats, part):
    indptr = [0]; indices = []
    for f in feats:
        indices.extend(f[part]); indptr.append(len(indices))
    m = sparse.csr_matrix((np.ones(len(indices), np.float32), np.array(indices, np.int64), np.array(indptr, np.int64)),
                          shape=(len(feats), NB))
    m.sum_duplicates()
    return m


def weight(m, idf):
    m = m.tocsr(copy=True)
    m.data = (1 + np.log(m.data)) * idf[m.indices]           # sublinear tf * idf
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()); norms[norms == 0] = 1
    return sparse.diags(1 / norms).dot(m).tocsr()


@numba.njit(parallel=True, fastmath=True)
def search(qp, qi, qd, ip, ii, idd, n_index, k, nblocks, out_i, out_s, work):
    nq = len(qp) - 1
    bs = (nq + nblocks - 1) // nblocks
    for b in numba.prange(nblocks):
        acc = np.zeros(n_index, np.float32)
        touched = np.empty(n_index, np.int32)
        s0 = b * bs; s1 = min(nq, s0 + bs)
        for q in range(s0, s1):
            nt = 0; w_ = 0
            for p in range(qp[q], qp[q + 1]):
                f = qi[p]; wq = qd[p]
                for pp in range(ip[f], ip[f + 1]):
                    d = ii[pp]
                    if acc[d] == 0.0:
                        touched[nt] = d; nt += 1
                    acc[d] += wq * idd[pp]
                w_ += ip[f + 1] - ip[f]
            work[q] = w_
            if nt > 0:
                sc = np.empty(nt, np.float32)
                for t in range(nt):
                    sc[t] = acc[touched[t]]
                kk = min(k, nt)
                order = np.argsort(-sc)[:kk]
                for t in range(kk):
                    out_i[q, t] = touched[order[t]]; out_s[q, t] = sc[order[t]]
                for t in range(nt):
                    acc[touched[t]] = 0.0


def main():
    t0 = time.time()
    s1 = load_source("train", 1).filter(pl.col("country") == COUNTRY)
    s23 = pl.concat([load_source("train", 2), load_source("train", 3)]).filter(pl.col("country") == COUNTRY)
    gt = load_gt()
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "rid"}))
    matched = s23.join(pairs, left_on="entity_id", right_on="rid", how="inner").sample(NQM, seed=1)
    unmatched = s23.join(pairs, left_on="entity_id", right_on="rid", how="anti").sample(NQU, seed=1).with_columns(pl.lit(None, pl.Utf8).alias("s1"))
    q = pl.concat([matched.select("entity_id", "business_name", "business_address", "s1"),
                   unmatched.select("entity_id", "business_name", "business_address", "s1")])
    print(f"[{COUNTRY}] index S1={s1.height:,} queries={q.height:,} load {time.time()-t0:.0f}s", flush=True)
    t = time.time()
    fi = par_featurize(s1["business_name"].to_list(), s1["business_address"].to_list())
    fq = par_featurize(q["business_name"].to_list(), q["business_address"].to_list())
    print(f"featurize {time.time()-t:.0f}s ({(s1.height+q.height)/(time.time()-t):,.0f} rec/s)", flush=True)
    s1_pos = {e: i for i, e in enumerate(s1["entity_id"].to_list())}
    truth = np.array([s1_pos.get(x, -1) if x is not None else -1 for x in q["s1"].to_list()])
    N = s1.height
    Mi_n, Mi_a = to_csr(fi, 0), to_csr(fi, 1)
    Mq_n, Mq_a = to_csr(fq, 0), to_csr(fq, 1)
    df_n = np.bincount(Mi_n.indices, minlength=NB); df_a = np.bincount(Mi_a.indices, minlength=NB)
    idf_n = np.log((N + 1) / (df_n + 1)).astype(np.float32) + 1; idf_a = np.log((N + 1) / (df_a + 1)).astype(np.float32) + 1
    Wi_n, Wi_a, Wq_n, Wq_a = weight(Mi_n, idf_n), weight(Mi_a, idf_a), weight(Mq_n, idf_n), weight(Mq_a, idf_a)
    K = 50
    for wa in [float(x) for x in os.environ.get('WA', '0.5').split(',')]:
        for cap in [int(x) for x in os.environ.get('CAPS', '2000,5000').split(',')]:
            Wi = sparse.hstack([Wi_n * np.sqrt(1 - wa), Wi_a * np.sqrt(wa)]).tocsr() if wa > 0 else Wi_n
            Wq = sparse.hstack([Wq_n * np.sqrt(1 - wa), Wq_a * np.sqrt(wa)]).tocsr() if wa > 0 else Wq_n
            df = np.bincount(Wi.indices, minlength=Wi.shape[1])
            inv = Wi.tocsc(); inv.sort_indices()
            keep = df <= cap
            # zero out capped posting lists
            ip = inv.indptr.astype(np.int64); lens = np.diff(ip) * keep
            ip2 = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            mask = np.repeat(keep, np.diff(ip))
            ii = inv.indices[mask].astype(np.int32); idd = inv.data[mask].astype(np.float32)
            out_i = -np.ones((Wq.shape[0], K), np.int32); out_s = np.zeros((Wq.shape[0], K), np.float32)
            work = np.zeros(Wq.shape[0], np.int64)
            t = time.time()
            search(Wq.indptr.astype(np.int64), Wq.indices.astype(np.int64), Wq.data.astype(np.float32), ip2, ii, idd, N, K, 40, out_i, out_s, work)
            dt = time.time() - t
            m = truth >= 0
            hit = out_i[m] == truth[m][:, None]
            rank = np.where(hit.any(1), hit.argmax(1) + 1, 999)
            rec = {k: f"{(rank <= k).mean()*100:.1f}" for k in [1, 2, 3, 5, 10, 20, 50]}
            print(f"wa={wa:.2f} cap={cap:>6} recall@k={rec} none={(out_i[m][:,0]<0).mean()*100:.2f}% | "
                  f"{Wq.shape[0]/dt:,.0f} q/s, postings/q mean={work.mean():,.0f} p99={np.percentile(work,99):,.0f} | "
                  f"top1 score matched p50={np.median(out_s[m][:,0]):.2f} unmatched p50={np.median(out_s[~m][:,0]):.2f}", flush=True)
    miss = np.where(m)[0][rank > 10]
    qn = q["business_name"].to_list(); qa = q["business_address"].to_list(); s1n = s1["business_name"].to_list(); s1a = s1["business_address"].to_list()
    import re as _re
    ind = sum(1 for i in miss if _re.search(r"[\u0900-\u0DFF]", qn[i])); emp = sum(1 for i in miss if not qa[i].strip())
    print(f"misses@10: {len(miss)} ({len(miss)/m.sum()*100:.1f}%) indic-name={ind/len(miss)*100:.0f}% empty-addr={emp/len(miss)*100:.0f}%")
    for i in miss[:14]:
        print(f"   Q: {qn[i][:45]} | {qa[i][:60]}\n   T: {s1n[truth[i]][:45]} | {s1a[truth[i]][:60]}")
    np.save(f"/private/tmp/claude-501/-Users-devopsdreamer-Desktop-Amazon-ML-Challenge/d340b0e5-e331-4ed8-b620-03e45d99111f/scratchpad/rank_{COUNTRY}.npy", rank)


if __name__ == "__main__":
    main()
