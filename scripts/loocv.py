"""Leave-one-out gene prioritization on the multiplex-heterogeneous network (MX-V + OMIM).

For every OMIM disease with at least two genes, each gene is held out in turn;
the other genes (and, for the heterogeneous methods, the disease node) are the
seeds, and the links between the held-out gene and the disease are removed.
Writes one row per task: label, config, split, disease, gene, rank.

Example
-------
    python scripts/loocv.py configs/loocv.yaml --backend cupy --out results/loocv.csv.gz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qwmx.config import config_id, expand, load_config
from qwmx.data import disease_split, load_heterogeneous, load_multiplex
from qwmx.evaluation import loocv_rank, tie_break_key
from qwmx.heterogeneous import (edit_bounds, iter_qw_mh_scores, loocv_tasks, qw_mh_columns,
                                qw_mh_hamiltonian, rwr_mh_loocv_scores)
from qwmx.rwr import rwr_m_scores, rwr_m_transition, rwr_mh_transition
from qwmx.walks import window_times

BATCH = 512


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="configuration file (configs/loocv*.yaml)")
    ap.add_argument("--out", required=True, help="output CSV (.csv or .csv.gz)")
    ap.add_argument("--data", default="data/processed", help="processed data directory")
    ap.add_argument("--only", nargs="+", default=None, help="run only these method labels")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    xp = np
    if args.backend == "cupy":
        import cupy as xp

    universe, layers = load_multiplex(args.data)
    N, L = len(universe), len(layers)
    B, D, omim = load_heterogeneous(args.data)
    M = D.shape[0]
    all_tasks, genes = loocv_tasks(B, omim)
    assignment = disease_split(genes)
    tasks = [t for t in all_tasks if assignment[str(omim[t[0]])] == cfg["split"]]
    key = tie_break_key(N)
    print(f"omim {cfg['split']}: {len({t[0] for t in tasks})} diseases, {len(tasks)} tasks, "
          f"backend {args.backend}", flush=True)

    rows = []

    def record(label, cid, idx, scores):
        for k, i in enumerate(idx):
            d, g, S = tasks[i]
            rows.append((label, cid, str(omim[d]), g, loocv_rank(scores[:, k], g, S, key)))

    for label, blocks in cfg["methods"].items():
        if args.only and label not in args.only:
            continue
        for block in blocks:
            for p in expand(block):
                t0 = time.perf_counter()
                method = p["method"]
                if method in ("QW-MH", "QW-M"):
                    kappa = p.get("kappa", 0.0) if method == "QW-MH" else 0.0
                    eta = p.get("eta", 0.0) if method == "QW-MH" else 0.0
                    H = qw_mh_hamiltonian(layers, p["delta"], B, D, kappa)
                    cols = qw_mh_columns(tasks, N, L, B, kappa=kappa, eta=eta, gamma=p["gamma"])
                    bounds = edit_bounds(H, kappa=kappa, gamma=p["gamma"], L=L)
                    windows = None
                    if "T" in p:
                        times, windows = window_times(p["T"], int(p.get("m", 8)))
                    else:
                        times = p["t"]
                    base = {k: v for k, v in p.items() if k not in ("t", "T")}
                    batch = int(p.get("batch", BATCH))
                    for idx, sc in iter_qw_mh_scores(H, cols, times, N, L, bounds,
                                                     batch_cols=batch, xp=xp):
                        sc = sc.get() if hasattr(sc, "get") else sc
                        if windows is not None:
                            for T, w in windows.items():
                                record(label, config_id({**base, "T": T}), idx, sc[w].mean(axis=0))
                        else:
                            for i, t in enumerate(times):
                                record(label, config_id({**base, "t": t}), idx, sc[i])
                elif method == "RWR-MH":
                    T = rwr_mh_transition(layers, p["delta"], B, D, p["lam"])
                    for b0 in range(0, len(tasks), BATCH):
                        idx = list(range(b0, min(b0 + BATCH, len(tasks))))
                        sc = rwr_mh_loocv_scores(T, [tasks[i] for i in idx], N, L, M, r=p["r"],
                                                 eta=p["eta"], xp=xp)
                        record(label, config_id(p), idx, sc)
                elif method == "RWR-M":
                    P = rwr_m_transition(layers, p["delta"])
                    for b0 in range(0, len(tasks), BATCH):
                        idx = list(range(b0, min(b0 + BATCH, len(tasks))))
                        sc = rwr_m_scores(P, [tasks[i][2] for i in idx], N, L, r=p["r"])
                        record(label, config_id(p), idx, sc)
                else:
                    raise ValueError(f"unknown method {method!r}")
                print(f"  {label} {config_id(p)}: {time.perf_counter() - t0:.1f} s", flush=True)

    out = pd.DataFrame(rows, columns=["label", "config", "disease", "gene", "rank"])
    out.insert(0, "split", cfg["split"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    out["rr"] = 1.0 / out["rank"]
    mrr = out.groupby(["label", "config", "disease"]).rr.mean().groupby(["label", "config"]).mean()
    print(mrr.sort_values(ascending=False).rename("mrr").round(4).to_string())


if __name__ == "__main__":
    main()
