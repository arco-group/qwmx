"""Summaries and paired tests of the outputs of prioritize.py and loocv.py.

Commands
--------
summary
    Mean and standard deviation over diseases of every metric, per method and
    configuration (metrics are first averaged over the ten splits of each
    disease). For leave-one-out outputs: MRR, Top-k accuracy and median rank.
test
    Paired Wilcoxon signed-rank tests on per-disease values, Holm-corrected
    over the listed pairs.
union
    Recall@k of the union of two methods' top-k lists and the share of hits
    found by only one of them (from the ``--ranks`` output of prioritize.py).

Examples
--------
    python scripts/analyze.py summary results/opentargets.csv.gz
    python scripts/analyze.py test results/opentargets.csv.gz --metric ap \
        --pairs QW-M-TA:QW-M-ST QW-M-TA:DK-M QW-M-TA:RWR-M
    python scripts/analyze.py union results/opentargets_ranks.csv.gz --a QW-M-TA --b RWR-M
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from qwmx.stats import paired_tests

TOP_KS = (1, 10, 50, 100)


def read(paths) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def is_loocv(df: pd.DataFrame) -> bool:
    return "rank" in df.columns and "metric" not in df.columns


def per_disease(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Diseases by (label, config): the metric averaged over splits, or the MRR."""
    if is_loocv(df):
        v = df.assign(rr=1.0 / df["rank"]).groupby(["label", "config", "disease"]).rr.mean()
    else:
        v = df[df.metric == metric].groupby(["label", "config", "disease"]).value.mean()
    return v.unstack(["label", "config"])


def summary(df: pd.DataFrame, metrics) -> pd.DataFrame:
    if is_loocv(df):
        g = df.groupby(["label", "config"])
        out = pd.DataFrame({"mrr": per_disease(df, "rr").mean()})
        for k in TOP_KS:
            out[f"top{k}"] = g["rank"].apply(lambda r: (r <= k).mean())
        out["median_rank"] = g["rank"].median()
        out["tasks"] = g.size()
        out["diseases"] = g.disease.nunique()
        return out.sort_values("mrr", ascending=False)
    cols = {}
    for m in metrics:
        t = per_disease(df, m)
        cols[m] = t.mean()
        cols[f"{m}_sd"] = t.std(ddof=1)
    out = pd.DataFrame(cols)
    out["diseases"] = per_disease(df, metrics[0]).notna().sum()
    return out.sort_values(metrics[0], ascending=False)


def test(df: pd.DataFrame, metric: str, pairs) -> pd.DataFrame:
    t = per_disease(df, metric)
    by_label = {}
    for label, config in t.columns:
        if label in by_label:
            raise SystemExit(f"{label!r} has several configurations; keep one per label")
        by_label[label] = t[(label, config)]
    table = pd.DataFrame(by_label)
    return paired_tests(table, [tuple(p.split(":")) for p in pairs])


def union(ranks: pd.DataFrame, a: str, b: str, k: int) -> pd.Series:
    ra = ranks[ranks.label == a].set_index(["disease", "run", "gene"])["rank"]
    rb = ranks[ranks.label == b].set_index(["disease", "run", "gene"])["rank"]
    j = pd.concat([ra.rename("a"), rb.rename("b")], axis=1).reset_index()
    ha, hb = j.a <= k, j.b <= k
    j["hit_a"], j["hit_b"], j["hit_union"] = ha, hb, ha | hb
    recall = j.groupby(["disease", "run"])[["hit_a", "hit_b", "hit_union"]].mean()
    recall = recall.groupby("disease").mean().mean()
    found = j[j.hit_union]
    return pd.Series({f"recall@{k} {a}": recall.hit_a, f"recall@{k} {b}": recall.hit_b,
                      f"recall@{k} union": recall.hit_union,
                      f"share found by both": (found.hit_a & found.hit_b).mean(),
                      f"share found only by {a}": (found.hit_a & ~found.hit_b).mean(),
                      f"share found only by {b}": (~found.hit_a & found.hit_b).mean()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.add_argument("--metrics", nargs="+",
                   default=["ap", "recall@25", "recall@100", "recall@300"])
    s.add_argument("--top", type=int, default=None, help="show only the best configurations")
    t = sub.add_parser("test")
    t.add_argument("files", nargs="+")
    t.add_argument("--metric", default="ap", help="metric name, or 'rr' for leave-one-out MRR")
    t.add_argument("--pairs", nargs="+", required=True, help="comparisons as A:B (A minus B)")
    u = sub.add_parser("union")
    u.add_argument("ranks")
    u.add_argument("--a", required=True)
    u.add_argument("--b", required=True)
    u.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    if args.command == "summary":
        out = summary(read(args.files), args.metrics)
        if args.top:
            out = out.groupby(level="label", sort=False).head(args.top)
        print(out.round(4).to_string())
    elif args.command == "test":
        print(test(read(args.files), args.metric, args.pairs).to_string(index=False))
    else:
        print(union(pd.read_csv(args.ranks), args.a, args.b, args.k).round(4).to_string())


if __name__ == "__main__":
    main()
