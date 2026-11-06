"""Metrics against scikit-learn and the qdgp ranking; statistics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from qwmx.evaluation import auroc_ap, degree_metrics, degree_tertiles, loocv_rank, rank_metrics
from qwmx.stats import holm, paired_tests


@pytest.mark.parametrize("seed", range(5))
def test_auroc_ap_match_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    s = np.round(rng.normal(size=3000), 1)
    y = rng.random(3000) < 0.02
    a, p = auroc_ap(s, y)
    assert abs(a - roc_auc_score(y, s)) < 1e-12
    assert abs(p - average_precision_score(y, s)) < 1e-12


def test_recall_matches_qdgp_sort():
    rng = np.random.default_rng(7)
    n = 500
    scores = np.round(rng.random(n), 2)
    key = rng.permutation(n)
    train, test = list(range(0, 40, 2)), list(range(1, 40, 2))
    m = rank_metrics(scores, train, test, key, ks=(25, 100))
    mask = np.ones(n, bool)
    mask[train] = False
    y = np.zeros(n, int)
    y[test] = 1
    labels = [yy for _, _, yy in sorted(zip(scores[mask], key[mask], y[mask]), reverse=True)]
    hits = np.cumsum(labels)
    assert m["recall@25"] == hits[24] / len(test)
    assert m["recall@100"] == hits[99] / len(test)


def test_loocv_rank_matches_rank_metrics_order():
    rng = np.random.default_rng(8)
    scores = np.round(rng.random(300), 2)
    key = rng.permutation(300)
    seeds = [0, 1, 2]
    order = [i for _, _, i in sorted(((scores[i], key[i], i) for i in range(3, 300)), reverse=True)]
    for g in (10, 50, 200):
        assert loocv_rank(scores, g, seeds, key) == order.index(g) + 1


def test_degree_metrics_of_degree_scorer():
    rng = np.random.default_rng(3)
    n = 600
    deg = rng.permutation(n).astype(float) + 1
    train, test = list(range(60)), list(range(60, 120))
    tert = degree_tertiles(deg)
    assert np.bincount(tert).tolist() == [200, 200, 200]
    m = degree_metrics(deg.copy(), train, test, np.arange(n), deg, tert)
    assert m["spearman_degree"] == pytest.approx(1.0)
    assert m["recall@300_t2"] == 1.0 and m["recall@300_t0"] == 0.0


def test_holm_and_paired_tests():
    adj = holm(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03}))
    assert np.allclose(adj[["a", "c", "b"]].to_numpy(), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"x": rng.random(40) + 0.2, "y": rng.random(40), "z": rng.random(40)})
    res = paired_tests(t, [("x", "y"), ("x", "z")])
    assert (res.median_diff > 0).all() and (res.p_holm >= res.p).all()
