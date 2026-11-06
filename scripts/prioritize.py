"""Disease gene prioritization with ten random 50/50 seed/test splits per disease.

Runs every method of a configuration file and writes one row per task and
metric: dataset, split, label, config, disease, run, metric, value.

Example
-------
    python scripts/prioritize.py configs/opentargets.yaml --backend cupy \
        --out results/opentargets.csv.gz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

from qwmx.config import config_id, expand, load_config
from qwmx.data import LAYERS, disease_split, load_diseases, load_multiplex
from qwmx.evaluation import (degree_metrics, degree_tertiles, rank_metrics, split, test_ranks,
                             tie_break_key)
from qwmx.graphs import degrees, laplacian, merged, sym_normalize
from qwmx.rwr import degree_scores, nei_scores, rwr_m_scores, rwr_m_transition, rwr_scores
from qwmx.walks import iter_scores, multiplex_operator, window_times

RUNS = 10
RWR_M_CHUNK = 256
SINGLE_CHUNK = 512


class Recorder:
    """Evaluates score columns and collects metric rows (and optionally test-gene ranks)."""

    def __init__(self, tasks, key, deg=None, ranks=False):
        self.tasks, self.key = tasks, key
        self.deg = deg
        self.tertile = None if deg is None else degree_tertiles(deg)
        self.rows = []
        self.ranks = [] if ranks else None

    def add(self, label, cid, task_ids, scores):
        for k, i in enumerate(task_ids):
            d, run, train, test = self.tasks[i]
            m = rank_metrics(scores[:, k], train, test, self.key)
            if self.deg is not None:
                m.update(degree_metrics(scores[:, k], train, test, self.key, self.deg,
                                        self.tertile))
            for metric, value in m.items():
                self.rows.append((label, cid, d, run, metric, value))
            if self.ranks is not None:
                for g, r in zip(test, test_ranks(scores[:, k], train, test, self.key)):
                    self.ranks.append((label, cid, d, run, g, int(r)))


def to_host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def run_walk(label, p, layers, seed_sets, rec, xp, dtype):
    """QW-M (unitary) or DK-M (diffusion) on the supra-graph, single-time or time-averaged."""
    L = len(layers)
    batch = int(p.get("batch", 512))
    if p["method"] == "QW-M":
        W = multiplex_operator(layers, p["delta"], p.get("norm", "spectral"),
                               p.get("scale_coupling", False))
        kw = dict(gamma=p["gamma"], init=p.get("init", "coherent"))
    else:
        kw = dict(kind="diffusion")
        W = laplacian(multiplex_operator(layers, p["delta"]))
    windows = None
    if "T" in p:
        times, windows = window_times(p["T"], int(p.get("m", 8)))
    else:
        times = p["t"]
    base = {k: v for k, v in p.items() if k not in ("t", "T")}
    for sets, sc in iter_scores(W, seed_sets, times, n_layers=L, batch_cols=batch, xp=xp,
                                dtype=dtype, **kw):
        sc = to_host(sc)
        if windows is not None:
            for T, idx in windows.items():
                rec.add(label, config_id({**base, "T": T}), sets, sc[idx].mean(axis=0))
        else:
            for i, t in enumerate(times):
                rec.add(label, config_id({**base, "t": t}), sets, sc[i])


def run_rwr_m(label, p, layers, seed_sets, rec):
    N, L = layers[0].shape[0], len(layers)
    P = rwr_m_transition(layers, p["delta"])
    for i in range(0, len(seed_sets), RWR_M_CHUNK):
        idx = list(range(i, min(i + RWR_M_CHUNK, len(seed_sets))))
        sc = rwr_m_scores(P, [seed_sets[j] for j in idx], N, L, r=p["r"])
        rec.add(label, config_id(p), idx, sc)


def run_single(label, p, graphs, seed_sets, rec, xp, dtype):
    """QA, DK, RWR, NEI or Degree on one graph.

    Only genes with at least one edge in the graph are scored; the others get
    ``-inf`` and rank last. Seeds outside the graph are dropped.
    """
    A = graphs[p["graph"]]
    N = A.shape[0]
    nodes = np.nonzero(degrees(A) > 0)[0]
    pos = -np.ones(N, dtype=np.int64)
    pos[nodes] = np.arange(len(nodes))
    sub = sps.csr_matrix(A[nodes][:, nodes])
    sets = [[int(pos[s]) for s in S if pos[s] >= 0] for S in seed_sets]
    nonempty = [j for j, S in enumerate(sets) if S]

    def expand_scores(sc):
        full = np.full((N,) + sc.shape[1:], -np.inf)
        full[nodes] = sc
        return full

    method = p["method"]
    if method in ("QA", "DK"):
        base = {k: v for k, v in p.items() if k != "t"}
        if method == "QA":
            H, kw = sub, dict(gamma=p["gamma"], init="incoherent")
        else:
            H, kw = laplacian(sub), dict(kind="diffusion")
        for sets_k, sc in iter_scores(H, [sets[j] for j in nonempty], p["t"], xp=xp,
                                      dtype=dtype, **kw):
            sc = to_host(sc)
            ids = [nonempty[k] for k in sets_k]
            for i, t in enumerate(p["t"]):
                rec.add(label, config_id({**base, "t": t}), ids, expand_scores(sc[i]))
        cids = [config_id({**base, "t": t}) for t in p["t"]]
    else:
        R = sym_normalize(sub) if method == "RWR" else None
        for i in range(0, len(nonempty), SINGLE_CHUNK):
            ids = nonempty[i:i + SINGLE_CHUNK]
            chunk = [sets[j] for j in ids]
            if method == "RWR":
                sc = rwr_scores(sub, chunk, p["r"], R=R)
            elif method == "NEI":
                sc = nei_scores(sub, chunk)
            else:
                sc = degree_scores(sub, chunk)
            rec.add(label, config_id(p), ids, expand_scores(sc))
        cids = [config_id(p)]
    empty = [j for j, S in enumerate(sets) if not S]
    if empty:
        blank = np.full((N, len(empty)), -np.inf)
        for cid in cids:
            rec.add(label, cid, empty, blank)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="configuration file (configs/*.yaml)")
    ap.add_argument("--out", required=True, help="output CSV (.csv or .csv.gz)")
    ap.add_argument("--data", default="data/processed", help="processed data directory")
    ap.add_argument("--only", nargs="+", default=None, help="run only these method labels")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    ap.add_argument("--precision", default="float64", choices=["float64", "float32"])
    ap.add_argument("--degree", action="store_true",
                    help="add score-degree metrics (merged-network degree)")
    ap.add_argument("--ranks", default=None, help="also write the rank of every test gene here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    xp = np
    if args.backend == "cupy":
        import cupy as xp
    dtype = np.dtype(args.precision)

    universe, layers = load_multiplex(args.data)
    N = len(universe)
    by_name = dict(zip(LAYERS, layers))
    graphs = {**by_name, "MERGED": merged(layers)}
    diseases = load_diseases(args.data, cfg["dataset"])
    assignment = disease_split(diseases)
    names = sorted(d for d in diseases if cfg["split"] == "all" or assignment[d] == cfg["split"])
    tasks = []
    for run in range(cfg.get("runs", RUNS)):
        for d in names:
            train, test = split(diseases[d], run)
            tasks.append((d, run, sorted(train), sorted(test)))
    seed_sets = [t[2] for t in tasks]
    print(f"{cfg['dataset']} {cfg['split']}: {len(names)} diseases, {len(tasks)} tasks, "
          f"{N} genes, backend {args.backend} {dtype}", flush=True)

    rec = Recorder(tasks, tie_break_key(N), deg=degrees(graphs["MERGED"]) if args.degree else None,
                   ranks=args.ranks is not None)
    for label, blocks in cfg["methods"].items():
        if args.only and label not in args.only:
            continue
        for block in blocks:
            for p in expand(block):
                t0 = time.perf_counter()
                sub_layers = [by_name[k] for k in p.get("layers", LAYERS)]
                if p["method"] in ("QW-M", "DK-M"):
                    run_walk(label, p, sub_layers, seed_sets, rec, xp, dtype)
                elif p["method"] == "RWR-M":
                    run_rwr_m(label, p, sub_layers, seed_sets, rec)
                else:
                    run_single(label, p, graphs, seed_sets, rec, xp, dtype)
                print(f"  {label} {config_id(p)}: {time.perf_counter() - t0:.1f} s", flush=True)

    out = pd.DataFrame(rec.rows, columns=["label", "config", "disease", "run", "metric", "value"])
    out.insert(0, "split", cfg["split"])
    out.insert(0, "dataset", cfg["dataset"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    if rec.ranks is not None:
        pd.DataFrame(rec.ranks, columns=["label", "config", "disease", "run", "gene", "rank"]
                     ).to_csv(args.ranks, index=False)

    per_disease = out.groupby(["label", "config", "metric", "disease"]).value.mean()
    summary = per_disease.groupby(["label", "config", "metric"]).mean().unstack("metric")
    cols = [c for c in ("ap", "recall@25", "recall@100", "recall@300") if c in summary]
    print(summary[cols].sort_values("ap", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
