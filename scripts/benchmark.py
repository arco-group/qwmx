"""Numerical accuracy and runtime of the batched multi-time Chebyshev propagator.

Commands
--------
accuracy
    Chebyshev against dense ``expm`` on random graphs and against
    ``expm_multiply`` on the MX-V supra-graph, in float64 and float32.
throughput
    Propagation time of QW-M on MX-V for the GMB diseases (one seed/test
    split): Chebyshev on CPU or GPU for several batch widths and numbers of
    evolution times, and optionally the reference (complex ``expm_multiply``
    per disease and per time). ``--workers`` splits the diseases over CPU
    processes. Chebyshev times are medians of ``--repeats`` runs after a warm-up.
dense
    GPU time to propagate one block of columns with a dense eigendecomposition
    (decomposition excluded) and with Chebyshev, on supra-graphs of growing size.

Examples
--------
    python scripts/benchmark.py accuracy
    python scripts/benchmark.py throughput --backend cupy --B 1 8 32 128 512 --n-t 20
    python scripts/benchmark.py throughput --max-diseases 10 --B 512 --reference
    python scripts/benchmark.py dense --n-t 20
"""

from __future__ import annotations

import argparse
import statistics
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from qwmx.data import LAYERS, load_diseases, load_multiplex
from qwmx.evaluation import split
from qwmx.graphs import laplacian, normalize_layer, supra_adjacency
from qwmx.propagate import propagate, seed_loop_operator, spectral_bounds
from qwmx.walks import iter_scores, layer_sum

DELTA, GAMMA = 0.5, 1.0


def times_for(n_t: int) -> list[float]:
    """``t = 4`` for one time point, otherwise ``n_t`` log-spaced times in [0.25, 32]."""
    return [4.0] if n_t == 1 else list(np.geomspace(0.25, 32, n_t))


def mxv_operator(processed, layers=LAYERS, delta=DELTA):
    universe, mats = load_multiplex(processed, layers)
    return len(universe), supra_adjacency([normalize_layer(A, "spectral") for A in mats], delta)


def gmb_seed_sets(processed, max_diseases=None):
    diseases = load_diseases(processed, "gmb")
    names = sorted(diseases)[:max_diseases]
    return [sorted(split(diseases[d], 0)[0]) for d in names]


def top_k(scores, exclude, k=300):
    s = np.asarray(scores, dtype=np.float64).copy()
    s[np.asarray(exclude, dtype=np.int64)] = -np.inf
    return np.argsort(-s, kind="stable")[:k]


# ----------------------------------------------------------------------------- accuracy

def accuracy(args) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for N, gen in ((500, "ER"), (2000, "ER"), (2000, "BA")):
        G = (nx.gnp_random_graph(N, 8 / N, seed=1) if gen == "ER"
             else nx.barabasi_albert_graph(N, 4, seed=1))
        A = nx.adjacency_matrix(G, nodelist=range(N)).astype(np.float64).tocsr()
        Ad = A.toarray()
        seeds = rng.choice(N, 6, replace=False)
        times = [0.45, 4.0, 16.0]
        for gamma in (0.0, 2.0):
            ref = np.empty((len(times), N, len(seeds)))
            for j, s in enumerate(seeds):
                H = Ad.copy()
                H[s, s] += gamma
                for i, t in enumerate(times):
                    ref[i, :, j] = np.abs(expm(-1j * t * H)[:, s]) ** 2
            bounds = spectral_bounds(A, gamma=gamma)
            for precision in ("float64", "float32"):
                dt = np.dtype(precision)
                psi0 = np.zeros((N, len(seeds)), dtype=dt)
                psi0[seeds, np.arange(len(seeds))] = 1
                op = seed_loop_operator(sps.csr_matrix(A, dtype=dt), gamma, seeds,
                                        np.arange(len(seeds)))
                out = propagate(op, psi0, times, bounds)
                rows.append({"case": f"{gen} N={N}", "reference": "expm", "kind": "unitary",
                             "gamma": gamma, "precision": precision,
                             "max_abs_err": float(np.abs(out - ref).max())})
        Lap = laplacian(A)
        bounds = spectral_bounds(Lap, psd=True)
        for precision in ("float64", "float32"):
            dt = np.dtype(precision)
            psi0 = np.zeros((N, len(seeds)), dtype=dt)
            psi0[seeds, np.arange(len(seeds))] = 1
            L_dt = sps.csr_matrix(Lap, dtype=dt)
            out = propagate(lambda P: L_dt @ P, psi0, [0.3, 1.0], bounds, kind="diffusion")
            err = max(float(np.abs(out[i] - expm(-t * Lap.toarray())[:, seeds]).max())
                      for i, t in enumerate([0.3, 1.0]))
            rows.append({"case": f"{gen} N={N}", "reference": "expm", "kind": "diffusion",
                         "gamma": 0.0, "precision": precision, "max_abs_err": err})
        print(f"  random {gen} N={N} done", flush=True)

    N, W = mxv_operator(args.data)
    L = len(LAYERS)
    seed_sets = gmb_seed_sets(args.data, 3)
    times = [4.0, 16.0]
    refs = {}
    for j, S in enumerate(seed_sets):
        R = (np.arange(L)[:, None] * N + np.asarray(S)[None, :]).ravel()
        H = W + sps.csr_matrix((np.full(len(R), GAMMA), (R, R)), shape=W.shape)
        Z = np.zeros((W.shape[0], len(R)), dtype=complex)
        Z[R, np.arange(len(R))] = 1
        for i, t in enumerate(times):
            p = (np.abs(expm_multiply(-1j * t * H, Z)) ** 2).sum(axis=1)
            refs[j, i] = layer_sum(p[None, :, None], L)[0, :, 0]
    for precision in ("float64", "float32"):
        cheb = {}
        for sets, sc in iter_scores(W, seed_sets, times, gamma=GAMMA, n_layers=L,
                                    dtype=np.dtype(precision)):
            for k, j in enumerate(sets):
                for i in range(len(times)):
                    cheb[j, i] = sc[i, :, k]
        err = max(float(np.abs(cheb[key] - refs[key]).max()) for key in refs)
        same = sum(set(top_k(cheb[j, i], S).tolist()) == set(top_k(refs[j, i], S).tolist())
                   for j, S in enumerate(seed_sets) for i in range(len(times)))
        rows.append({"case": f"MX-V N*L={W.shape[0]}", "reference": "expm_multiply",
                     "kind": "unitary", "gamma": GAMMA, "precision": precision,
                     "max_abs_err": err, "top300_sets_identical": f"{same}/{len(refs)}"})
    print("  MX-V done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- throughput

_SHARED = {}


def chebyshev_run(W, seed_sets, times, B, dtype, xp, bounds):
    t0 = time.perf_counter()
    for _ in iter_scores(W, seed_sets, times, gamma=GAMMA, n_layers=len(LAYERS), init="incoherent",
                         batch_cols=B, dtype=dtype, xp=xp, bounds=bounds):
        pass
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0


def reference_run(W, seed_sets, times, N, L=len(LAYERS)):
    t0 = time.perf_counter()
    for S in seed_sets:
        R = (np.arange(L)[:, None] * N + np.asarray(S)[None, :]).ravel()
        H = W + sps.csr_matrix((np.full(len(R), GAMMA), (R, R)), shape=W.shape)
        Z = np.zeros((W.shape[0], len(R)), dtype=complex)
        Z[R, np.arange(len(R))] = 1.0
        for t in times:
            np.abs(expm_multiply(-1j * t * H, Z)) ** 2
    return time.perf_counter() - t0


def _groups(seed_sets, workers):
    """Split the diseases over workers, balancing the number of seed columns."""
    groups, load = [[] for _ in range(workers)], [0] * workers
    for S in sorted(seed_sets, key=len, reverse=True):
        i = load.index(min(load))
        groups[i].append(S)
        load[i] += len(S)
    return [g for g in groups if g]


def _cheb_worker(job):
    sets, times, B, precision = job
    return chebyshev_run(_SHARED["W"], sets, times, B, np.dtype(precision), np, _SHARED["bounds"])


def _ref_worker(job):
    sets, times = job
    return reference_run(_SHARED["W"], sets, times, _SHARED["N"])


def throughput(args) -> pd.DataFrame:
    xp = np
    if args.backend == "cupy":
        import cupy as xp
    N, W = mxv_operator(args.data)
    seed_sets = gmb_seed_sets(args.data, args.max_diseases)
    n_cols = sum(len(LAYERS) * len(S) for S in seed_sets)
    bounds = spectral_bounds(W, gamma=GAMMA)
    print(f"N*L={W.shape[0]} nnz={W.nnz} diseases={len(seed_sets)} columns={n_cols}", flush=True)
    pool = None
    if args.workers > 1:
        import multiprocessing as mp
        _SHARED.update(W=W, bounds=bounds, N=N)
        pool = mp.get_context("fork").Pool(args.workers)
        groups = _groups(seed_sets, args.workers)

    rows = []

    def record(impl, precision, B, n_t, walls):
        rows.append({"implementation": impl, "precision": precision, "B": B, "n_t": n_t,
                     "workers": args.workers, "diseases": len(seed_sets), "columns": n_cols,
                     "median_s": statistics.median(walls), "repeats": len(walls)})
        print(f"  {impl} {precision} B={B} n_t={n_t}: {statistics.median(walls):.2f} s", flush=True)

    for precision in args.precision:
        for n_t in args.n_t:
            times = times_for(n_t)
            for B in args.B:
                if pool is not None:
                    jobs = [(g, times, B, precision) for g in groups]
                    pool.map(_cheb_worker, [(g[:1], times, B, precision) for g in groups])
                    walls = []
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        pool.map(_cheb_worker, jobs)
                        walls.append(time.perf_counter() - t0)
                else:
                    dtype = np.dtype(precision)
                    chebyshev_run(W, seed_sets[:2], times, B, dtype, xp, bounds)
                    walls = [chebyshev_run(W, seed_sets, times, B, dtype, xp, bounds)
                             for _ in range(args.repeats)]
                record(f"chebyshev-{args.backend}", precision, B, n_t, walls)

    if args.reference:
        for n_t in args.n_t:
            times = times_for(n_t)
            if pool is not None:
                t0 = time.perf_counter()
                pool.map(_ref_worker, [(g, times) for g in groups])
                walls = [time.perf_counter() - t0]
            else:
                walls = [reference_run(W, seed_sets, times, N)]
            record("reference-expm_multiply", "complex128", 1, n_t, walls)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- dense

def dense(args) -> pd.DataFrame:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    def timed(fn):
        fn()
        cp.cuda.Stream.null.synchronize()
        walls = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            fn()
            cp.cuda.Stream.null.synchronize()
            walls.append(time.perf_counter() - t0)
        return statistics.median(walls)

    rows = []
    rng = np.random.default_rng(0)
    for subset in args.subsets:
        _, W = mxv_operator(args.data, subset.split(","))
        n = W.shape[0]
        bounds = spectral_bounds(W)
        Wd = cpsp.csr_matrix(W)
        lam = None
        for precision in args.dense_precision:
            cp.get_default_memory_pool().free_all_blocks()
            try:
                Hd = Wd.astype(precision).toarray()
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                lam, V = cp.linalg.eigh(Hd)
                cp.cuda.Stream.null.synchronize()
                t_eig = time.perf_counter() - t0
                del Hd
                break
            except cp.cuda.memory.OutOfMemoryError:
                Hd = None
                print(f"  {subset}: eigendecomposition in {precision} does not fit", flush=True)
        if lam is None:
            continue
        Wd = Wd.astype(precision)
        cp.get_default_memory_pool().free_all_blocks()
        idx = rng.choice(n, args.B, replace=False)
        psi0 = cp.zeros((n, args.B), dtype=precision)
        psi0[cp.asarray(idx), cp.arange(args.B)] = 1.0
        for n_t in args.n_t:
            times = times_for(n_t)

            def dense_block():
                c = V.T @ psi0
                out = cp.empty((len(times), n, args.B), dtype=precision)
                for i, t in enumerate(times):
                    re = V @ (cp.cos(lam * t)[:, None] * c)
                    im = V @ (cp.sin(lam * t)[:, None] * c)
                    out[i] = re * re + im * im
                return out

            def chebyshev_block():
                return propagate(lambda P: Wd @ P, psi0, times, bounds, xp=cp)

            t_dense, t_cheb = timed(dense_block), timed(chebyshev_block)
            rows.append({"layers": subset, "nodes": n, "precision": precision, "B": args.B,
                         "n_t": n_t, "eigendecomposition_s": t_eig, "dense_s": t_dense,
                         "chebyshev_s": t_cheb})
            print(f"  {subset} n={n} {precision} n_t={n_t}: dense {t_dense:.3f} s, "
                  f"Chebyshev {t_cheb:.3f} s", flush=True)
        del V, lam, psi0
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("accuracy")
    t = sub.add_parser("throughput")
    t.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    t.add_argument("--B", type=int, nargs="+", default=[512])
    t.add_argument("--n-t", type=int, nargs="+", default=[1, 5, 20])
    t.add_argument("--precision", nargs="+", default=["float64"])
    t.add_argument("--repeats", type=int, default=5)
    t.add_argument("--max-diseases", type=int, default=None)
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--reference", action="store_true")
    d = sub.add_parser("dense")
    d.add_argument("--subsets", nargs="+", default=["PPI", "PPI,PATH", "PPI,PATH,COEX"])
    d.add_argument("--n-t", type=int, nargs="+", default=[20])
    d.add_argument("--B", type=int, default=512)
    d.add_argument("--dense-precision", nargs="+", default=["float64", "float32"])
    d.add_argument("--repeats", type=int, default=3)
    for p in (a, t, d):
        p.add_argument("--data", default="data/processed")
        p.add_argument("--out", default=None, help="write the results to this CSV")
    args = ap.parse_args()

    out = {"accuracy": accuracy, "throughput": throughput, "dense": dense}[args.command](args)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
