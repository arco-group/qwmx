"""Quantum walk on multiplex-heterogeneous networks (QW-MH) and leave-one-out tasks.

Node order: ``N * L`` gene replicas (layer-major) followed by ``M`` diseases::

    H = [[ W_delta,             kappa * B_sup ],
         [ kappa * B_sup^T,     A_D / rho(A_D) ]]

``W_delta`` is the QW-M supra-adjacency, ``B_sup`` the gene-disease incidence
replicated to every layer. With ``kappa = 0`` the disease part is dropped and
the walk is QW-M.

A leave-one-out task ``(d, g, S)`` holds out gene ``g`` of disease ``d`` and
seeds the other genes ``S`` (and ``d``). The links ``g_l -- d`` are removed for
that task only, through per-column edits, so all tasks share one batched
propagation.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import scipy.sparse as sps

from qwmx.graphs import normalize_layer, supra_adjacency
from qwmx.propagate import Bounds, column_edit_operator, propagate, spectral_bounds
from qwmx.rwr import geometric_mean_layers, mh_seed_vector, random_walk_restart
from qwmx.walks import to_device


def loocv_tasks(B, names):
    """Leave-one-out tasks for every disease with at least two genes.

    Parameters
    ----------
    B : sparse matrix, shape (N, M)
        Gene-disease incidence.
    names : sequence
        Disease identifiers, in column order.

    Returns
    -------
    tasks : list of (int, int, list of int)
        ``(disease column, held-out gene, other genes)``.
    genes : dict
        Disease identifier (as str) to its genes.
    """
    Bc = sps.csc_matrix(B)
    tasks, genes = [], {}
    for d in range(Bc.shape[1]):
        G = np.sort(Bc.indices[Bc.indptr[d]:Bc.indptr[d + 1]])
        if len(G) < 2:
            continue
        genes[str(names[d])] = G.tolist()
        for g in G:
            tasks.append((d, int(g), [int(x) for x in G if x != g]))
    return tasks, genes


def qw_mh_hamiltonian(layers, delta: float, B=None, D=None, kappa: float = 0.0) -> sps.csr_matrix:
    """Hamiltonian of QW-MH (or of QW-M if ``kappa = 0``), without seed terms.

    Parameters
    ----------
    layers : list of sparse matrices
        Gene layers; each is divided by its spectral radius.
    delta : float
        Inter-layer coupling.
    B : sparse matrix, shape (N, M), optional
        Gene-disease incidence.
    D : sparse matrix, shape (M, M), optional
        Disease-similarity adjacency.
    kappa : float
        Gene-disease coupling.

    Returns
    -------
    csr_matrix
    """
    W = supra_adjacency([normalize_layer(A, "spectral") for A in layers], delta)
    if kappa == 0.0:
        return W
    L = len(layers)
    B_sup = sps.csr_matrix(sps.vstack([B] * L), dtype=np.float64)
    H = sps.bmat([[W, kappa * B_sup], [kappa * B_sup.T, normalize_layer(D, "spectral")]],
                 format="csr")
    H.sort_indices()
    return H


def qw_mh_columns(tasks, N: int, L: int, B, *, kappa: float, eta: float, gamma: float):
    """Initial state and per-task edits of each leave-one-out task.

    The initial state has amplitude ``sqrt((1 - eta) / (L |S|))`` on every seed
    replica and ``sqrt(eta)`` on the disease node (genes only if ``kappa = 0``
    or ``eta = 0``). ``gamma`` adds a self-loop on every seeded node. The links
    between the held-out gene and its disease are removed in every layer.

    Returns
    -------
    list of tuples
        ``(rows, values, edit_rows, edit_sources, edit_values)`` per task.
    """
    with_d = kappa != 0.0 and eta > 0.0
    Bc = None if B is None else sps.csc_matrix(B)
    out = []
    for d, g, S in tasks:
        R = (np.arange(L)[:, None] * N + np.unique(np.asarray(S, dtype=np.int64))[None, :]).ravel()
        rows, vals = [R], [np.full(len(R), np.sqrt((1 - eta if with_d else 1.0) / len(R)))]
        dn = N * L + d
        if with_d:
            rows.append(np.array([dn]))
            vals.append(np.array([np.sqrt(eta)]))
        rows = np.concatenate(rows)
        er, es, ev = [], [], []
        if gamma != 0.0:
            er.append(rows)
            es.append(rows)
            ev.append(np.full(len(rows), gamma))
        if kappa != 0.0 and Bc[g, d] != 0:
            gl = np.arange(L) * N + g
            er += [gl, np.full(L, dn)]
            es += [np.full(L, dn), gl]
            ev += [np.full(2 * L, -kappa)]

        def cat(x, dt):
            return np.concatenate(x).astype(dt) if x else np.zeros(0, dt)

        out.append((rows, np.concatenate(vals), cat(er, np.int64), cat(es, np.int64),
                    cat(ev, np.float64)))
    return out


def edit_bounds(H, *, kappa: float, gamma: float, L: int) -> Bounds:
    """Spectral bounds valid for ``H`` plus the edits of any task.

    The removed links form a star of ``L`` edges of weight ``kappa`` (norm at
    most ``kappa * sqrt(L)``); the self-loops shift the spectrum by at most ``gamma``.
    """
    b = spectral_bounds(H)
    pert = kappa * np.sqrt(L)
    return Bounds(b.lmin - pert - (gamma if gamma < 0 else 0.0),
                  b.lmax + pert + (gamma if gamma > 0 else 0.0))


def iter_qw_mh_scores(H, columns, times: Sequence[float], N: int, L: int, bounds: Bounds, *,
                      batch_cols: int = 512, xp=np, dtype=np.float64
                      ) -> Iterator[tuple[list[int], object]]:
    """Propagate the leave-one-out tasks in batches.

    Parameters
    ----------
    H : sparse matrix
        Output of :func:`qw_mh_hamiltonian`.
    columns : list
        Output of :func:`qw_mh_columns`.
    times : sequence of float
        Evolution times.
    N, L : int
        Number of genes and layers.
    bounds : Bounds
        Output of :func:`edit_bounds`.
    batch_cols : int
        Tasks per propagation.
    xp : module
        NumPy or CuPy.
    dtype : dtype
        Working precision.

    Yields
    ------
    tasks : list of int
        Task positions in ``columns``.
    scores : array, shape (n_t, N, len(tasks))
        Gene probabilities summed over layer replicas.
    """
    n = H.shape[0]
    Hd = to_device(H, xp, dtype)
    dtype = np.dtype(dtype)
    for b0 in range(0, len(columns), batch_cols):
        block = columns[b0:b0 + batch_cols]
        psi0 = np.zeros((n, len(block)), dtype=dtype)
        er, es, ec, ev = [], [], [], []
        for c, (rows, vals, r_, s_, v_) in enumerate(block):
            psi0[rows, c] = vals
            er.append(r_)
            es.append(s_)
            ev.append(v_)
            ec.append(np.full(len(r_), c))
        er, es, ec, ev = (np.concatenate(x) for x in (er, es, ec, ev))
        op = column_edit_operator(Hd, er, es, ec, ev, xp) if len(er) else (lambda P: Hd @ P)
        out = propagate(op, xp.asarray(psi0), times, bounds, xp=xp)
        genes = out[:, : N * L].reshape(len(times), L, N, len(block)).sum(axis=1)
        yield list(range(b0, b0 + len(block))), genes


def rwr_mh_loocv_scores(T, tasks, N: int, L: int, M: int, *, r: float = 0.7,
                        eta: float = 0.5, xp=np) -> np.ndarray:
    """RWR-MH scores of leave-one-out tasks.

    As in the original R code, the transition entries between ``g_l`` and ``d``
    are set to zero without renormalization; the seeds are the other genes and
    disease ``d``.

    Returns
    -------
    ndarray, shape (N, len(tasks))
    """
    T = sps.csr_matrix(T)
    er, es, ec, ev = [], [], [], []
    for c, (d, g, _) in enumerate(tasks):
        dn = N * L + d
        for l in range(L):
            gl = l * N + g
            for row, src in ((gl, dn), (dn, gl)):
                w = T[row, src]
                if w != 0:
                    er.append(row)
                    es.append(src)
                    ec.append(c)
                    ev.append(-w)
    Td = to_device(T, xp, np.float64)
    op = column_edit_operator(Td, np.array(er, dtype=np.int64), np.array(es, dtype=np.int64),
                              np.array(ec, dtype=np.int64), np.array(ev), xp)
    p0 = mh_seed_vector([S for _, _, S in tasks], [[d] for d, _, _ in tasks], N, L, M, eta)
    p = random_walk_restart(op, xp.asarray(p0), r)
    p = p.get() if hasattr(p, "get") else p
    return geometric_mean_layers(p[: N * L], L, N)
