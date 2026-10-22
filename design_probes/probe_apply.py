"""Design probe: apply the pooled (US+India) probe model to UNLABELED test countries (incl. France) and report
unsupervised diagnostics vs the labelled train reference. No labels are used for test (there are none)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from probe_core import load_country, run_blocking, make_features, log
SP = os.path.dirname(os.path.abspath(__file__)); K = 5; NS = 30000; T = 0.7


def main():
    bst = lgb.Booster(model_file=f"{SP}/probe_lgb_both.txt")
    for split, country in [(s, c) for s, c in [x.split(":") for x in sys.argv[1:]]]:
        s1, s23, truth_q = load_country(split, country)
        rng = np.random.default_rng(0); role = np.zeros(s1.height, np.int8); role[rng.permutation(s1.height)[:NS]] = 2
        log(f"{split}/{country}: S1={s1.height:,} queries={s23.height:,} queries/S1={s23.height/s1.height:.2f}")
        B = run_blocking(s1, s23, role, K)
        top1 = B["top_s"][:, 0]
        X, names, P_q, P_s = make_features(s1, s23, B, K)
        p = bst.predict(X, num_threads=10)
        # per query best p (over its candidates); per sampled S1: predicted matches with rule B
        order = np.lexsort((-p, P_q)); Pq, Ps, pp = P_q[order], P_s[order], p[order]
        first = np.concatenate([[True], Pq[1:] != Pq[:-1]])
        pbest = pp[first]
        sel = first & (pp >= T)
        samp = np.where(role == 2)[0]
        npred = np.bincount(Ps[sel], minlength=s1.height)[samp]
        qs = np.quantile(top1[top1 > 0], [0.1, 0.25, 0.5, 0.75, 0.9])
        msg = (f"  top1 retrieval score quantiles(10/25/50/75/90)={np.round(qs, 2).tolist()} none={np.mean(top1 == 0)*100:.2f}%\n"
               f"  relevant-query best-p: >=0.5 {np.mean(pbest >= 0.5)*100:.1f}%  >=0.7 {np.mean(pbest >= 0.7)*100:.1f}%  >=0.9 {np.mean(pbest >= 0.9)*100:.1f}%\n"
               f"  sampled S1 predicted matches @t={T}: mean={npred.mean():.2f}  predicted-singleton={np.mean(npred == 0)*100:.1f}%  "
               f"dist={np.bincount(np.minimum(npred, 9)).tolist()}")
        if split == "train":
            tc = np.bincount(truth_q[truth_q >= 0], minlength=s1.height)[samp]
            msg += f"\n  (labelled reference) true matches: mean={tc.mean():.2f} singleton={np.mean(tc == 0)*100:.1f}%"
        log(msg)
        del B, X, p


if __name__ == "__main__":
    main()
