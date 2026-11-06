"""Classical baselines: single-graph RWR, NEI and Degree; RWR-M and RWR-MH.

The single-graph methods follow Saarinen et al. (2024). RWR-M and RWR-MH are
reimplementations of the R code of Valdeolivas et al. (2019) and reproduce its
example outputs (see ``tests/test_rwr_reference.py``). Multiplex node order is
layer-major: gene ``i`` of layer ``l`` is row ``l * N + i``; in RWR-MH the
``M`` disease nodes follow the ``N * L`` gene replicas.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from qwmx.graphs import degrees, supra_adjacency, sym_normalize

SINGLE_RWR_TOL = 1e-6
MULTIPLEX_RWR_TOL = 1e-10


def seed_mask(seed_sets, n: int) -> np.ndarray:
    """``(n, B)`` 0/1 matrix with the seeds of each set."""
    M = np.zeros((n, len(seed_sets)))
    for j, S in enumerate(seed_sets):
        M[np.asarray(S, dtype=np.int64), j] = 1.0
    return M


def rwr_scores(A, seed_sets, r: float = 0.4, tol: float = SINGLE_RWR_TOL, R=None) -> np.ndarray:
    """Random walk with restart on one graph.

    Iterates ``p <- (1 - r) D^-1/2 A D^-1/2 p + r p0``; each column stops at its
    own convergence. Seeds get score 0.

    Parameters
    ----------
    A : sparse matrix
        Adjacency matrix.
    seed_sets : sequence of sequences of int
        Seeds of each task.
    r : float
        Restart probability.
    tol : float
        Convergence threshold on the L2 change of a column.
    R : sparse matrix, optional
        Precomputed ``D^-1/2 A D^-1/2``.

    Returns
    -------
    ndarray, shape (n, len(seed_sets))
    """
    R = sym_normalize(A) if R is None else R
    M = seed_mask(seed_sets, A.shape[0])
    p0 = M / M.sum(axis=0, keepdims=True)
    p = p0.copy()
    active = np.arange(p.shape[1])
    while len(active):
        new = (1 - r) * (R @ p[:, active]) + r * p0[:, active]
        res = np.linalg.norm(new - p[:, active], axis=0)
        p[:, active] = new
        active = active[res > tol]
    return p * (1 - M)


def nei_scores(A, seed_sets) -> np.ndarray:
    """Fraction of each node's neighbours that are seeds; seeds get 0."""
    M = seed_mask(seed_sets, A.shape[0])
    d = degrees(A)[:, None]
    return (A @ M) / (d + 1e-50) * (1 - M)


def degree_scores(A, seed_sets) -> np.ndarray:
    """Node degree, repeated for every task."""
    d = degrees(A)
    return np.repeat(d[:, None], len(seed_sets), axis=1)


def column_normalize(W) -> sps.csr_matrix:
    """Divide each column by its sum; zero columns stay zero."""
    s = np.asarray(W.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(s > 0, 1.0 / s, 0.0)
    return sps.csr_matrix(W @ sps.diags(inv))


def random_walk_restart(P, p0, r: float, tol: float = MULTIPLEX_RWR_TOL):
    """Iterate ``p <- (1 - r) P p + r p0`` until every column changes by less than ``tol``.

    Parameters
    ----------
    P : sparse matrix or callable
        Column-stochastic transition matrix, or a function ``p -> P p``.
    p0 : array, shape (n, B)
        Restart vectors, columns summing to 1 (NumPy or CuPy).
    r : float
        Restart probability.
    tol : float
        Convergence threshold on the L2 change of every column.

    Returns
    -------
    array, shape (n, B)
    """
    apply_p = P if callable(P) else (lambda x: P @ x)
    p = p0.copy()
    while True:
        new = (1 - r) * apply_p(p) + r * p0
        residue = (((new - p) ** 2).sum(axis=0)) ** 0.5
        p = new
        if bool((residue < tol).all()):
            return p


def geometric_mean_layers(scores, L: int, N: int):
    """Geometric mean over the ``L`` layer replicas of an ``(N * L, B)`` score array."""
    stacked = scores.reshape((L, N) + scores.shape[1:])
    return np.prod(stacked, axis=0) ** (1.0 / L)


def multiplex_seed_vector(seed_sets, N: int, L: int) -> np.ndarray:
    """RWR-M restart vectors: uniform over all replicas of the seeds."""
    p0 = np.zeros((N * L, len(seed_sets)))
    for j, S in enumerate(seed_sets):
        S = np.asarray(S, dtype=np.int64)
        for l in range(L):
            p0[l * N + S, j] = 1.0 / L / len(S)
    return p0 / p0.sum(axis=0, keepdims=True)


def rwr_m_transition(layers, delta: float) -> sps.csr_matrix:
    """Column-normalized supra-adjacency of the raw layers."""
    return column_normalize(supra_adjacency(layers, delta))


def rwr_m_scores(P, seed_sets, N: int, L: int, r: float = 0.7) -> np.ndarray:
    """RWR-M gene scores.

    Parameters
    ----------
    P : sparse matrix
        Output of :func:`rwr_m_transition`.
    seed_sets : sequence of sequences of int
        Seed genes of each task.
    N, L : int
        Number of genes and layers.
    r : float
        Restart probability.

    Returns
    -------
    ndarray, shape (N, len(seed_sets))
        Geometric mean over layers of the stationary probabilities.
    """
    p0 = multiplex_seed_vector(seed_sets, N, L)
    p = random_walk_restart(P, p0, r)
    return geometric_mean_layers(p, L, N)


def rwr_mh_transition(layers, delta: float, B, D, lam: float = 0.5) -> sps.csr_matrix:
    """Column-stochastic transition matrix of the multiplex-heterogeneous network.

    Parameters
    ----------
    layers : list of sparse matrices
        Gene layers (raw adjacency).
    delta : float
        Inter-layer coupling.
    B : sparse matrix, shape (N, M)
        Gene-disease incidence, replicated to every layer.
    D : sparse matrix, shape (M, M)
        Disease-similarity adjacency.
    lam : float
        Probability of jumping between the gene and disease networks where a
        bipartite link exists.

    Returns
    -------
    csr_matrix, shape (N * L + M, N * L + M)
    """
    L = len(layers)
    W = supra_adjacency(layers, delta)
    B_sup = sps.csr_matrix(sps.vstack([B] * L), dtype=np.float64)
    D = sps.csr_matrix(D, dtype=np.float64)

    col_w = np.asarray(W.sum(axis=0)).ravel()
    row_b = np.asarray(B_sup.sum(axis=1)).ravel()
    col_b = np.asarray(B_sup.sum(axis=0)).ravel()
    col_d = np.asarray(D.sum(axis=0)).ravel()

    def inv(x):
        with np.errstate(divide="ignore"):
            return np.where(x > 0, 1.0 / x, 0.0)

    T_gg = _scale_columns(W, np.where(row_b > 0, 1 - lam, 1.0) * inv(col_w))
    T_dd = _scale_columns(D, np.where(col_b > 0, 1 - lam, 1.0) * inv(col_d))
    T_gd = _scale_columns(B_sup, lam * inv(col_b))
    T_dg = _scale_columns(sps.csr_matrix(B_sup.T), lam * inv(row_b))
    T = sps.bmat([[T_gg, T_gd], [T_dg, T_dd]], format="csr")
    T.sort_indices()
    return T


def _scale_columns(X, s: np.ndarray) -> sps.csr_matrix:
    return sps.csr_matrix(X @ sps.diags(s))


def mh_seed_vector(gene_seeds, disease_seeds, N: int, L: int, M: int,
                   eta: float = 0.5) -> np.ndarray:
    """RWR-MH restart vectors.

    Each gene-seed replica gets ``(1 - eta) / (L |G|)`` and each seed disease
    ``eta / |D|``; if one of the two seed sets is empty the other gets all mass.
    """
    p0 = np.zeros((N * L + M, len(gene_seeds)))
    for j, (G, Ds) in enumerate(zip(gene_seeds, disease_seeds)):
        e = eta if (len(G) and len(Ds)) else (1.0 if len(Ds) else 0.0)
        if len(G):
            G = np.asarray(G, dtype=np.int64)
            for l in range(L):
                p0[l * N + G, j] = (1 - e) / L / len(G)
        if len(Ds):
            p0[N * L + np.asarray(Ds, dtype=np.int64), j] = e / len(Ds)
    return p0 / p0.sum(axis=0, keepdims=True)


def rwr_mh_scores(T, gene_seeds, disease_seeds, N: int, L: int, M: int,
                  r: float = 0.7, eta: float = 0.5):
    """RWR-MH scores.

    Returns
    -------
    genes : ndarray, shape (N, B)
        Geometric mean over layers.
    diseases : ndarray, shape (M, B)
    """
    p0 = mh_seed_vector(gene_seeds, disease_seeds, N, L, M, eta)
    p = random_walk_restart(T, p0, r)
    return geometric_mean_layers(p[: N * L], L, N), p[N * L:]
