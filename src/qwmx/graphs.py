"""Sparse graph construction: layers on a shared gene set, supra-graphs, normalizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh


def adjacency(edges: pd.DataFrame, index: dict, n: int) -> sps.csr_matrix:
    """Symmetric 0/1 adjacency matrix from an edge list.

    Parameters
    ----------
    edges : DataFrame
        Columns ``a`` and ``b`` with node identifiers.
    index : dict
        Node identifier to row position.
    n : int
        Number of nodes.

    Returns
    -------
    csr_matrix, shape (n, n)
        No self-loops; duplicate edges are collapsed.
    """
    a = edges.a.map(index).to_numpy()
    b = edges.b.map(index).to_numpy()
    if np.isnan(a.astype(float)).any() or np.isnan(b.astype(float)).any():
        raise ValueError("edge endpoint outside the node set")
    a, b = a.astype(np.int64), b.astype(np.int64)
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    A = sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    return A


def degrees(A) -> np.ndarray:
    """Row sums of ``A``."""
    return np.asarray(A.sum(axis=1)).ravel()


def laplacian(A) -> sps.csr_matrix:
    """Combinatorial Laplacian ``D - A``."""
    return sps.csr_matrix(sps.diags(degrees(A)) - A)


def sym_normalize(A) -> sps.csr_matrix:
    """``D^(-1/2) A D^(-1/2)``; isolated nodes get zero rows."""
    d = degrees(A)
    with np.errstate(divide="ignore"):
        inv = np.where(d > 0, d ** -0.5, 0.0)
    D = sps.diags(inv)
    return sps.csr_matrix(D @ A @ D)


def spectral_radius(A) -> float:
    """Largest absolute eigenvalue of a symmetric matrix."""
    if A.nnz == 0:
        return 0.0
    return abs(float(eigsh(sps.csr_matrix(A, dtype=np.float64), k=1, which="LM",
                           return_eigenvectors=False, tol=1e-8)[0]))


def normalize_layer(A, how: str = "spectral") -> sps.csr_matrix:
    """Normalize a layer: ``"spectral"`` divides by the spectral radius, ``"raw"`` leaves it."""
    if how == "raw":
        return sps.csr_matrix(A, dtype=np.float64)
    if how == "spectral":
        rho = spectral_radius(A)
        return sps.csr_matrix(A / rho) if rho > 0 else sps.csr_matrix(A, dtype=np.float64)
    raise ValueError(f"unknown normalization {how!r}")


def supra_adjacency(layers, delta: float, coupling_scale: float = 1.0) -> sps.csr_matrix:
    """Multiplex supra-adjacency matrix.

    Diagonal blocks are ``(1 - delta) A_l``; off-diagonal blocks are
    ``coupling_scale * delta / (L - 1) * I``. Node ``i`` of layer ``l`` is row
    ``l * N + i``, as in RWR-M.

    Parameters
    ----------
    layers : list of sparse matrices
        ``L`` adjacency matrices on the same ``N`` nodes.
    delta : float
        Inter-layer coupling in ``[0, 1]``.
    coupling_scale : float
        Multiplier of the inter-layer blocks.

    Returns
    -------
    csr_matrix, shape (N * L, N * L)
    """
    L = len(layers)
    N = layers[0].shape[0]
    blocks = [[None] * L for _ in range(L)]
    eye = sps.identity(N, format="csr")
    for i in range(L):
        for j in range(L):
            if i == j:
                blocks[i][j] = (1 - delta) * sps.csr_matrix(layers[i], dtype=np.float64)
            elif delta > 0:
                blocks[i][j] = (coupling_scale * delta / (L - 1)) * eye
    S = sps.bmat(blocks, format="csr")
    S.eliminate_zeros()
    S.sort_indices()
    return S


def merged(layers) -> sps.csr_matrix:
    """Union of the edges of several layers, as a 0/1 matrix."""
    M = sps.csr_matrix(sum(sps.csr_matrix(A) for A in layers))
    M.data[:] = 1.0
    return M
