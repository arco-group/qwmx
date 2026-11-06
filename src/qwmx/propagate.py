"""Batched multi-time Chebyshev propagation.

Two kernels share one recurrence:

* unitary: ``|exp(-i t H) psi0|^2``, computed in real arithmetic;
* diffusion: ``exp(-t H) psi0`` for a positive semidefinite ``H``.

``H`` enters only through a function ``apply_h(Psi) -> H @ Psi`` acting on a
block of columns, so the same code handles single graphs, supra-graphs and
batches whose columns carry different seed self-loops or edge edits. The
array module ``xp`` (NumPy or CuPy) is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from scipy.special import ive, jv

ApplyH = Callable[[object], object]

COEF_TOL = 1e-12
BOUND_MARGIN = 0.01


@dataclass(frozen=True)
class Bounds:
    """Interval ``[lmin, lmax]`` containing the spectrum of ``H``."""

    lmin: float
    lmax: float

    @property
    def a(self) -> float:
        """Half-width of the interval."""
        return (self.lmax - self.lmin) / 2

    @property
    def b(self) -> float:
        """Centre of the interval."""
        return (self.lmax + self.lmin) / 2


def spectral_bounds(A, *, gamma: float = 0.0, margin: float = BOUND_MARGIN,
                    psd: bool = False) -> Bounds:
    """Spectral bounds of ``H = A + gamma * P``, with ``P`` a diagonal 0/1 projector.

    Parameters
    ----------
    A : sparse matrix
        Real symmetric matrix.
    gamma : float
        Weight of the diagonal term; it shifts the spectrum by at most ``gamma``.
    margin : float
        Fraction of the interval width added on each side.
    psd : bool
        If True, ``A`` is positive semidefinite and the lower bound is 0.

    Returns
    -------
    Bounds
    """
    A = sps.csr_matrix(A, dtype=np.float64)
    if A.shape[0] <= 2:
        w = np.linalg.eigvalsh(A.toarray())
        lmin, lmax = float(w[0]), float(w[-1])
    else:
        lmax = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0])
        lmin = 0.0 if psd else float(
            eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0])
    if gamma > 0:
        lmax += gamma
    else:
        lmin += gamma
    width = max(lmax - lmin, 1e-12)
    lo = 0.0 if psd else lmin - margin * width
    return Bounds(lo, lmax + margin * width)


def seed_loop_operator(A, gamma: float = 0.0, loop_rows=None, loop_cols=None) -> ApplyH:
    """Batched action of ``A`` plus per-column seed self-loops.

    Column ``j`` of the block evolves under ``A + gamma * sum_s |s><s|``, where
    ``s`` runs over the rows ``loop_rows[i]`` with ``loop_cols[i] == j``.

    Parameters
    ----------
    A : sparse matrix
        Shared operator (SciPy or CuPy sparse).
    gamma : float
        Self-loop weight.
    loop_rows, loop_cols : array_like of int, optional
        Coordinates of the self-loops in the block.

    Returns
    -------
    callable
        ``Psi -> (A + gamma * M) Psi``.
    """
    if gamma == 0.0 or loop_rows is None or len(loop_rows) == 0:
        return lambda Psi: A @ Psi

    def apply_h(Psi):
        out = A @ Psi
        out[loop_rows, loop_cols] += gamma * Psi[loop_rows, loop_cols]
        return out

    return apply_h


def column_edit_operator(A, rows, srcs, cols, vals, xp=np) -> ApplyH:
    """Batched action of ``A`` plus sparse per-column edits.

    Edit ``i`` adds ``vals[i] * Psi[srcs[i], cols[i]]`` to ``out[rows[i], cols[i]]``,
    so column ``j`` evolves under ``A + sum_{i: cols[i] = j} vals[i] |rows[i]><srcs[i]|``.
    A self-loop is an edit with ``rows == srcs``; removing a symmetric edge
    ``(u, v)`` of weight ``w`` is the pair of edits ``(u, v, -w)`` and ``(v, u, -w)``.

    Parameters
    ----------
    A : sparse matrix
        Shared operator (SciPy or CuPy sparse).
    rows, srcs, cols : array_like of int
        Target row, source row and column of each edit.
    vals : array_like of float
        Edit weights.
    xp : module
        Array module of the block.

    Returns
    -------
    callable
    """
    rows, srcs, cols = (xp.asarray(x) for x in (rows, srcs, cols))
    vals = xp.asarray(vals)
    if xp is np:
        scatter = np.add.at
    else:
        import cupyx
        scatter = cupyx.scatter_add

    def apply_h(Psi):
        out = A @ Psi
        scatter(out, (rows, cols), vals.astype(Psi.dtype) * Psi[srcs, cols])
        return out

    return apply_h


def unitary_coefficients(tau: float, tol: float = COEF_TOL) -> np.ndarray:
    """Bessel coefficients ``J_k(tau)``, truncated where the tail falls below ``tol``."""
    tau = abs(float(tau))
    k_cap = int(np.ceil(tau + 10 * tau ** (1 / 3) + 20))
    c = jv(np.arange(k_cap + 1), tau)
    return _truncate(c, tau, tol)


def diffusion_coefficients(t: float, bounds: Bounds, tol: float = COEF_TOL) -> np.ndarray:
    """Coefficients ``c_k`` with ``exp(-tH) = sum_k c_k T_k((H - b) / a)``.

    From ``exp(-z x) = I_0(z) + 2 sum_k (-1)^k I_k(z) T_k(x)`` with ``z = t a``.
    """
    z = float(t) * bounds.a
    k_cap = int(np.ceil(z + 10 * z ** (1 / 3) + 20))
    k = np.arange(k_cap + 1)
    c = ive(k, z) * np.exp(z - float(t) * bounds.b)
    c[1:] *= 2 * (-1.0) ** k[1:]
    return _truncate(c, z, tol)


def _truncate(c: np.ndarray, tau: float, tol: float) -> np.ndarray:
    big = np.nonzero(np.abs(c) >= tol)[0]
    last = int(big[-1]) if len(big) else 0
    return c[: max(last, min(int(np.floor(tau)), len(c) - 1)) + 1]


def propagate(apply_h: ApplyH, psi0, times: Sequence[float], bounds: Bounds, *,
              kind: str = "unitary", tol: float = COEF_TOL, xp=np, return_stats: bool = False):
    """Propagate a block of states to several times with one Chebyshev recurrence.

    The Chebyshev vectors ``T_k(H_hat) psi0`` do not depend on time, so they are
    computed once and accumulated with the coefficients of every requested time.
    For the unitary kernel, even and odd orders go to separate real accumulators
    (real and imaginary parts of the evolved state).

    Parameters
    ----------
    apply_h : callable
        ``Psi -> H @ Psi`` on an ``(n, B)`` block; ``H`` must be real symmetric.
    psi0 : array, shape (n, B)
        Real initial states; the dtype sets the working precision.
    times : sequence of float
        Evolution times.
    bounds : Bounds
        Spectral interval valid for every column's Hamiltonian.
    kind : {"unitary", "diffusion"}
        ``"unitary"`` returns ``|exp(-itH) psi0|^2``; ``"diffusion"`` returns ``exp(-tH) psi0``.
    tol : float
        Coefficient truncation threshold.
    xp : module
        Array module holding ``psi0`` (NumPy or CuPy).
    return_stats : bool
        If True, also return a dict with the expansion order ``K``.

    Returns
    -------
    out : array, shape (n_t, n, B)
    stats : dict, optional
    """
    if kind not in ("unitary", "diffusion"):
        raise ValueError(f"unknown kind {kind!r}")
    times = [float(t) for t in times]
    a, b = bounds.a, bounds.b
    if kind == "unitary":
        coefs = [unitary_coefficients(a * t, tol) for t in times]
    else:
        coefs = [diffusion_coefficients(t, bounds, tol) for t in times]
    K = max(len(c) for c in coefs) - 1

    dtype = psi0.dtype
    n_t = len(times)
    re = xp.zeros((n_t,) + psi0.shape, dtype=dtype)
    im = xp.zeros((n_t,) + psi0.shape, dtype=dtype) if kind == "unitary" else None

    def scaled(Psi):
        out = apply_h(Psi)
        out -= b * Psi
        out *= 1.0 / a
        return out

    def accumulate(k, Tk):
        for i, c in enumerate(coefs):
            if k >= len(c) or c[k] == 0.0:
                continue
            if kind == "diffusion":
                re[i] += dtype.type(c[k]) * Tk
            elif k == 0:
                re[i] += dtype.type(c[0]) * Tk
            elif k % 2 == 0:
                re[i] += dtype.type(2 * (-1) ** (k // 2) * c[k]) * Tk
            else:
                im[i] += dtype.type(2 * (-1) ** ((k + 1) // 2) * c[k]) * Tk

    T_prev = psi0.copy()
    accumulate(0, T_prev)
    if K >= 1:
        T_curr = scaled(T_prev)
        accumulate(1, T_curr)
        for k in range(2, K + 1):
            T_next = scaled(T_curr)
            T_next *= 2
            T_next -= T_prev
            T_prev, T_curr = T_curr, T_next
            accumulate(k, T_curr)

    if kind == "unitary":
        out = re
        out *= re
        out += im * im
    else:
        out = re
    if return_stats:
        return out, {"K": K, "a": a, "b": b}
    return out
