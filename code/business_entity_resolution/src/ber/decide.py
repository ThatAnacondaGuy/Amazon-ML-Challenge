"""Decision layer and the exact challenge metric."""
import numpy as np


def argmax_assign(q, s, p, score, t):
    """Exclusivity: each S2/S3 record keeps only its best S1 (ties: retrieval score, then S1 index); keep if p >= t.
    Returns boolean mask over pairs."""
    order = np.lexsort((s, -score, -p, q))
    qo = q[order]
    first = np.ones(len(order), bool)
    first[1:] = qo[1:] != qo[:-1]
    keep = np.zeros(len(order), bool)
    keep[order[first]] = True
    return keep & (p >= t)


def best_per_query(q, s, p, score):
    order = np.lexsort((s, -score, -p, q))
    qo = q[order]
    first = np.ones(len(order), bool)
    first[1:] = qo[1:] != qo[:-1]
    idx = order[first]
    return idx  # pair indices of each query's best candidate


def cardinality(best_idx, p, s_n, t):
    """Mean predicted matches per S1 in a partition at threshold t (label-free)."""
    return float((p[best_idx] >= t).sum()) / max(1, s_n)


def card_matched_threshold(best_idx, p, s_n, ref_card, t_star, lo=-0.15, hi=0.25):
    grid = np.round(np.arange(t_star + lo, t_star + hi + 1e-9, 0.01), 2)
    pb = np.sort(p[best_idx])
    cards = (len(pb) - np.searchsorted(pb, grid, side="left")) / max(1, s_n)
    return float(grid[np.argmin(np.abs(cards - ref_card))]), cards


def f05_macro(true_count, pred_count, tp_count):
    """Exact README metric per S1 entity, macro-averaged. Arrays aligned over the evaluated S1 entities."""
    T = np.asarray(true_count, float); P = np.asarray(pred_count, float); TP = np.asarray(tp_count, float)
    f = np.zeros(len(T))
    f[(T == 0) & (P == 0)] = 1.0
    ok = (T > 0) & (P > 0) & (TP > 0)
    pr, rc = TP[ok] / P[ok], TP[ok] / T[ok]
    f[ok] = 1.25 * pr * rc / (0.25 * pr + rc)
    return float(f.mean()) if len(f) else 0.0


def f05_single(true_set, pred_set):
    """Reference implementation for unit tests (README example: 0.714)."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(set(true_set) & set(pred_set))
    if tp == 0:
        return 0.0
    pr, rc = tp / len(set(pred_set)), tp / len(set(true_set))
    return 1.25 * pr * rc / (0.25 * pr + rc)
