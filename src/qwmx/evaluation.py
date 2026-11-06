"""Evaluation protocols.

Seed/test splits (ten random 50/50 splits per disease)
    Ranking, tie breaking and splits follow Saarinen et al. (2024), so the
    metrics are comparable with theirs.
Leave-one-out (heterogeneous networks)
    Rank of the held-out gene among all non-seed genes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

RECALL_KS = (25, 100, 300)


def tie_break_key(n: int) -> np.ndarray:
    """Fixed random key used to break score ties (higher key ranks first)."""
    return np.random.default_rng(0).permutation(np.arange(n))


def split(genes, run: int, base_seed: int = 1000, train_size: float = 0.5):
    """Split the genes of a disease into seed and test genes (seed ``base_seed + run``)."""
    train, test = train_test_split(list(genes), train_size=train_size,
                                   random_state=base_seed + run)
    return train, test


def _candidates(n: int, train) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(train, dtype=np.int64)] = True
    return np.nonzero(~mask)[0]


def _finite(s: np.ndarray) -> np.ndarray:
    """Replace -inf (genes absent from a single graph) by a value below every real score."""
    finite = np.isfinite(s)
    if finite.all():
        return s
    s = s.copy()
    s[~finite] = (s[finite].min() if finite.any() else 0.0) - 1.0
    return s


def rank_metrics(scores: np.ndarray, train, test, key: np.ndarray,
                 ks=RECALL_KS) -> dict[str, float]:
    """AP, AUROC and Recall@k over all non-seed genes.

    Parameters
    ----------
    scores : ndarray, shape (n,)
        Gene scores; ``-inf`` ranks last.
    train, test : sequences of int
        Seed and test genes.
    key : ndarray, shape (n,)
        Tie-break key (:func:`tie_break_key`).
    ks : tuple of int
        Cut-offs for Recall@k.

    Returns
    -------
    dict
        ``recall@k`` for each ``k``, ``auroc`` and ``ap``.
    """
    n = len(scores)
    cand = _candidates(n, train)
    y = np.zeros(n, dtype=bool)
    y[np.asarray(test, dtype=np.int64)] = True
    s = np.asarray(scores, dtype=np.float64)[cand]
    yc = y[cand]
    order = np.lexsort((-key[cand], -s))
    hits = np.cumsum(yc[order])
    out = {f"recall@{k}": float(hits[min(k, len(hits)) - 1] / len(test)) for k in ks}
    out["auroc"], out["ap"] = auroc_ap(_finite(s), yc)
    return out


def auroc_ap(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """AUROC and average precision with scikit-learn's tie semantics, from one sort."""
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order].astype(np.float64)
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(yy)[last]
    fp = (last + 1) - tp
    P = tp[-1]
    Nn = len(ss) - P
    if P == 0 or Nn == 0:
        return float("nan"), float("nan")
    precision = tp / (tp + fp)
    recall = tp / P
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / Nn]
    auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return auroc, ap


def test_ranks(scores: np.ndarray, train, test, key: np.ndarray) -> np.ndarray:
    """1-based rank of each test gene among the non-seed genes."""
    n = len(scores)
    cand = _candidates(n, train)
    order = cand[np.lexsort((-key[cand], -np.asarray(scores, dtype=np.float64)[cand]))]
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(1, len(order) + 1)
    return pos[np.asarray(test, dtype=np.int64)]


def degree_tertiles(deg: np.ndarray) -> np.ndarray:
    """Degree tertile of each node: 0 low, 1 middle, 2 high."""
    lo, hi = np.quantile(deg, [1 / 3, 2 / 3])
    return np.where(deg <= lo, 0, np.where(deg <= hi, 1, 2))


def degree_metrics(scores: np.ndarray, train, test, key: np.ndarray, deg: np.ndarray,
                   tertile: np.ndarray, k: int = 300) -> dict[str, float]:
    """Score-degree dependence over the non-seed genes.

    Returns
    -------
    dict
        ``spearman_degree``: Spearman correlation between score and degree.
        ``recall@k_t{q}``: Recall@k restricted to test genes of degree tertile
        ``q`` (NaN if the tertile has no test gene).
    """
    n = len(scores)
    cand = _candidates(n, train)
    s = _finite(np.asarray(scores, dtype=np.float64)[cand])
    out = {"spearman_degree": float(spearmanr(s, deg[cand])[0])}
    order = cand[np.lexsort((-key[cand], -s))]
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(len(order))
    test = np.asarray(test, dtype=np.int64)
    for q in range(3):
        tq = test[tertile[test] == q]
        out[f"recall@{k}_t{q}"] = float((pos[tq] < k).mean()) if len(tq) else float("nan")
    return out


def loocv_rank(scores: np.ndarray, gene: int, seeds, key: np.ndarray) -> int:
    """1-based rank of the held-out gene among all non-seed genes."""
    cand = np.ones(len(scores), dtype=bool)
    cand[np.asarray(seeds, dtype=np.int64)] = False
    s, sg = scores[cand], scores[gene]
    k = key[cand]
    better = (s > sg) | ((s == sg) & (k > key[gene]))
    return int(better.sum()) + 1
