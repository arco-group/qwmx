"""Quantum walks (QA, QW-M) and heat-kernel diffusion (DK, DK-M) on one code path.

A single graph is the ``L = 1`` case of a supra-graph. Seeds are gene indices
in ``[0, N)``; every replica ``l * N + s`` of a seed ``s`` is seeded. Many seed
sets are scored in batches of at most ``batch_cols`` propagated columns.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import scipy.sparse as sps

from qwmx.graphs import degrees, normalize_layer, supra_adjacency
from qwmx.propagate import Bounds, propagate, seed_loop_operator, spectral_bounds


def multiplex_operator(layers, delta: float, norm: str = "spectral",
                       scale_coupling: bool = False) -> sps.csr_matrix:
    """Supra-adjacency ``W_delta`` of normalized layers.

    Parameters
    ----------
    layers : list of sparse matrices
        Layer adjacency matrices on the same gene set.
    delta : float
        Inter-layer coupling.
    norm : {"spectral", "raw"}
        Layer normalization.
    scale_coupling : bool
        If True, the inter-layer blocks are multiplied by the mean node degree
        (averaged over layers), so that ``delta`` weighs inter-layer against
        intra-layer links of unnormalized layers (QW-M-RC).

    Returns
    -------
    csr_matrix
    """
    normed = [normalize_layer(A, norm) for A in layers]
    scale = float(np.mean([degrees(A).mean() for A in layers])) if scale_coupling else 1.0
    return supra_adjacency(normed, delta, coupling_scale=scale)


def window_times(windows, m: int = 8):
    """Sample times of the time-averaged readout.

    Window ``T`` is sampled at ``j * T / m`` for ``j = 1, ..., m``.

    Parameters
    ----------
    windows : sequence of float
        Averaging windows ``T``.
    m : int
        Number of times per window.

    Returns
    -------
    times : list of float
        Sorted union of the sample times of all windows.
    index : dict
        Window ``T`` to the positions of its ``m`` times in ``times``.
    """
    times = sorted({float(T) * j / m for T in windows for j in range(1, m + 1)})
    pos = {t: i for i, t in enumerate(times)}
    return times, {float(T): [pos[float(T) * j / m] for j in range(1, m + 1)] for T in windows}


def to_device(H, xp, dtype):
    """Copy a SciPy sparse matrix to the array module ``xp`` with the given dtype."""
    H = sps.csr_matrix(H, dtype=dtype)
    if xp is np:
        return H
    import cupyx.scipy.sparse as cpsp
    return cpsp.csr_matrix(H)


def layer_sum(p, n_layers: int):
    """Sum an ``(n_t, N * L, B)`` array over the ``L`` layer replicas."""
    if n_layers == 1:
        return p
    n_t, NL, B = p.shape
    return p.reshape(n_t, n_layers, NL // n_layers, B).sum(axis=1)


def _replicas(S: Sequence[int], N: int, n_layers: int) -> np.ndarray:
    S = np.unique(np.asarray(S, dtype=np.int64))
    return (np.arange(n_layers)[:, None] * N + S[None, :]).ravel()


def _columns(R: np.ndarray, kind: str, init: str):
    """Initial columns of one seed set as (rows, values) pairs."""
    if kind == "diffusion" or init == "coherent":
        val = 1.0 if kind == "diffusion" else 1.0 / np.sqrt(len(R))
        return [(R, np.full(len(R), val))]
    if init == "incoherent":
        return [(np.array([r]), np.ones(1)) for r in R]
    raise ValueError(f"unknown initial state {init!r}")


def iter_scores(H, seed_sets: Sequence[Sequence[int]], times: Sequence[float], *,
                kind: str = "unitary", gamma: float = 0.0, n_layers: int = 1,
                init: str = "incoherent", batch_cols: int = 512, bounds: Bounds | None = None,
                dtype=np.float64, xp=np, stats: dict | None = None
                ) -> Iterator[tuple[list[int], object]]:
    """Score many seed sets with batched multi-time propagation.

    Parameters
    ----------
    H : sparse matrix, shape (N * L, N * L)
        Hamiltonian (``kind="unitary"``) or Laplacian (``kind="diffusion"``).
    seed_sets : sequence of sequences of int
        Seed genes of each task, as indices in ``[0, N)``.
    times : sequence of float
        Evolution times.
    kind : {"unitary", "diffusion"}
        Quantum walk or heat-kernel diffusion.
    gamma : float
        Self-loop weight on every seed replica (unitary only).
    n_layers : int
        Number of layers ``L``.
    init : {"coherent", "incoherent"}
        Coherent: one normalized superposition of all seed replicas. Incoherent:
        one column per seed replica, probabilities summed (as QA).
    batch_cols : int
        Maximum number of columns per propagation.
    bounds : Bounds, optional
        Spectral bounds; computed from ``H`` and ``gamma`` if omitted.
    dtype : dtype
        Working precision.
    xp : module
        NumPy or CuPy.
    stats : dict, optional
        Receives the expansion order ``K`` and the number of propagated columns.

    Yields
    ------
    sets : list of int
        Positions in ``seed_sets`` of the tasks in this batch.
    scores : array, shape (n_t, N, len(sets))
        Gene scores, summed over layer replicas.
    """
    if kind == "diffusion" and gamma != 0.0:
        raise ValueError("seed self-loops are defined for the unitary walk only")
    NL = H.shape[0]
    N = NL // n_layers
    times = list(times)
    if bounds is None:
        bounds = spectral_bounds(H, gamma=gamma, psd=(kind == "diffusion"))
    Hd = to_device(H, xp, dtype)
    dtype = np.dtype(dtype)

    def run(block):
        B = len(block)
        psi0 = np.zeros((NL, B), dtype=dtype)
        lr, lc = [], []
        for c, (_, rows, vals, R) in enumerate(block):
            psi0[rows, c] = vals
            if gamma != 0.0:
                lr.append(R)
                lc.append(np.full(len(R), c))
        op = seed_loop_operator(Hd, gamma,
                                xp.asarray(np.concatenate(lr)) if lr else None,
                                xp.asarray(np.concatenate(lc)) if lc else None)
        out, st = propagate(op, xp.asarray(psi0), times, bounds, kind=kind, xp=xp,
                            return_stats=True)
        if stats is not None:
            stats["K"] = st["K"]
            stats["spmm_cols"] = stats.get("spmm_cols", 0) + B * st["K"]
        return out

    def flush(chunk, sets):
        acc = xp.zeros((len(times), NL, len(sets)), dtype=dtype)
        for b0 in range(0, len(chunk), batch_cols):
            block = chunk[b0:b0 + batch_cols]
            out = run(block)
            owner = np.array([c[0] for c in block])
            for k in np.unique(owner):
                acc[:, :, k] += out[:, :, xp.asarray(np.nonzero(owner == k)[0])].sum(axis=2)
        return sets, layer_sum(acc, n_layers)

    # A chunk holds whole seed sets and at most batch_cols columns; a single set
    # larger than batch_cols is split over several propagations and summed.
    chunk, sets = [], []
    for j, S in enumerate(seed_sets):
        R = _replicas(S, N, n_layers)
        cols = _columns(R, kind, init)
        if chunk and len(chunk) + len(cols) > batch_cols:
            yield flush(chunk, sets)
            chunk, sets = [], []
        k = len(sets)
        sets.append(j)
        chunk.extend((k, rows, vals, R) for rows, vals in cols)
    if chunk:
        yield flush(chunk, sets)


def qw_m_ta_scores(layers, seed_sets, *, delta: float = 0.0, gamma: float = 0.0,
                   window: float = 256.0, m: int = 8, batch_cols: int = 128,
                   dtype=np.float64, xp=np) -> np.ndarray:
    """Time-averaged multiplex quantum walk (QW-M-TA) scores.

    Spectrally normalized layers, coherent initial state over all seed
    replicas, probabilities summed over layers and averaged over the times
    ``j * window / m``, ``j = 1, ..., m``.

    Parameters
    ----------
    layers : list of sparse matrices
        Layer adjacency matrices on the same ``N`` genes.
    seed_sets : sequence of sequences of int
        Seed genes of each disease.
    delta : float
        Inter-layer coupling.
    gamma : float
        Seed self-loop weight.
    window : float
        Averaging window ``T``.
    m : int
        Number of sample times.
    batch_cols : int
        Maximum number of columns per propagation.
    dtype : dtype
        Working precision.
    xp : module
        NumPy or CuPy.

    Returns
    -------
    ndarray, shape (N, len(seed_sets))
        Scores on the host. Seeds are not masked.
    """
    W = multiplex_operator(layers, delta)
    times, _ = window_times([window], m)
    out = np.zeros((layers[0].shape[0], len(seed_sets)))
    for sets, sc in iter_scores(W, seed_sets, times, gamma=gamma, n_layers=len(layers),
                                init="coherent", batch_cols=batch_cols, dtype=dtype, xp=xp):
        sc = sc.get() if hasattr(sc, "get") else sc
        out[:, sets] = sc.mean(axis=0)
    return out
