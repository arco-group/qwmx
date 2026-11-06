"""Paired comparisons on per-disease metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values."""
    p = pvalues.sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.to_numpy())])
    return pd.Series(np.minimum(adj, 1.0), index=p.index).reindex(pvalues.index)


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation; zero differences are dropped."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_tests(table: pd.DataFrame, pairs) -> pd.DataFrame:
    """Wilcoxon signed-rank tests with Holm correction over the given pairs.

    Parameters
    ----------
    table : DataFrame
        One row per disease, one column per method.
    pairs : sequence of (str, str)
        Comparisons ``(a, b)``; differences are ``a - b``. Holm correction is
        applied over all pairs together.

    Returns
    -------
    DataFrame
        One row per pair: number of diseases, median and mean difference,
        rank-biserial correlation, p-value and Holm-adjusted p-value.
    """
    rows = []
    for a, b in pairs:
        d = (table[a] - table[b]).dropna().to_numpy()
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append({"a": a, "b": b, "n": len(d), "median_diff": float(np.median(d)),
                     "mean_diff": float(d.mean()), "rank_biserial": rank_biserial(d),
                     "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p).to_numpy()
    return out
