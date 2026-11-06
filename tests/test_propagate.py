"""Chebyshev propagation against dense expm and expm_multiply."""

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from qwmx.propagate import propagate, seed_loop_operator, spectral_bounds

TIMES = [0.0, 0.3, 0.45, 1.0, 4.0, 16.0]


def random_graph(n=80, p=0.08, seed=0):
    rng = np.random.default_rng(seed)
    upper = sps.random(n, n, density=p, random_state=rng, data_rvs=lambda k: np.ones(k))
    A = sps.triu(upper, k=1)
    A = ((A + A.T) > 0).astype(np.float64)
    return sps.csr_matrix(A)


def seed_columns(n, seed_sets):
    """One indicator column per seed, with self-loop coordinates for its seed set."""
    cols, rows_l, cols_l = [], [], []
    for S in seed_sets:
        for s in S:
            j = len(cols)
            e = np.zeros(n)
            e[s] = 1.0
            cols.append(e)
            rows_l.extend(S)
            cols_l.extend([j] * len(S))
    return np.stack(cols, axis=1), np.array(rows_l), np.array(cols_l)


@pytest.mark.parametrize("gamma", [0.0, 5.0])
def test_unitary_matches_dense_expm_mixed_batch(gamma):
    A = random_graph()
    n = A.shape[0]
    seed_sets = [[0, 5, 9], [3, 40], [12, 13, 70, 71]]
    Psi0, lr, lc = seed_columns(n, seed_sets)
    probs = propagate(seed_loop_operator(A, gamma, lr, lc), Psi0, TIMES,
                      spectral_bounds(A, gamma=gamma))
    j = 0
    for S in seed_sets:
        cols = slice(j, j + len(S))
        H = A.toarray().copy()
        H[S, S] += gamma
        for i, t in enumerate(TIMES):
            ref = np.abs(expm(-1j * t * H) @ Psi0[:, cols]) ** 2
            assert np.max(np.abs(probs[i][:, cols] - ref)) < 1e-10
        j += len(S)


def test_unitary_preserves_norm():
    A = random_graph(seed=1)
    Psi0, lr, lc = seed_columns(A.shape[0], [[1, 2, 3]])
    probs = propagate(seed_loop_operator(A, 2.0, lr, lc), Psi0, TIMES,
                      spectral_bounds(A, gamma=2.0))
    assert np.max(np.abs(probs.sum(axis=1) - 1.0)) < 1e-10


def test_unitary_matches_expm_multiply():
    A = random_graph(n=200, p=0.03, seed=2)
    S = [4, 8, 15, 16, 23, 42]
    Psi0, lr, lc = seed_columns(A.shape[0], [S])
    H = A.copy().tolil()
    for s in S:
        H[s, s] += 5.0
    probs = propagate(seed_loop_operator(A, 5.0, lr, lc), Psi0, [0.45],
                      spectral_bounds(A, gamma=5.0))
    ref = np.abs(expm_multiply(-1j * 0.45 * H.tocsr(), Psi0.astype(complex))) ** 2
    assert np.max(np.abs(probs[0] - ref)) < 1e-10


def test_multi_time_equals_single_time():
    A = random_graph(seed=3)
    Psi0, lr, lc = seed_columns(A.shape[0], [[0, 1], [7]])
    op = seed_loop_operator(A, 1.0, lr, lc)
    bounds = spectral_bounds(A, gamma=1.0)
    multi = propagate(op, Psi0, TIMES, bounds)
    for i, t in enumerate(TIMES):
        assert np.array_equal(multi[i], propagate(op, Psi0, [t], bounds)[0])


def test_diffusion_matches_dense_expm():
    A = random_graph(seed=4)
    L = sps.csr_matrix(sps.diags(np.asarray(A.sum(axis=1)).ravel()) - A)
    psi0 = np.zeros((A.shape[0], 2))
    psi0[[0, 3, 9], 0] = 1.0
    psi0[[20, 21], 1] = 1.0
    out = propagate(lambda X: L @ X, psi0, TIMES, spectral_bounds(L, psd=True), kind="diffusion")
    for i, t in enumerate(TIMES):
        assert np.max(np.abs(out[i] - expm(-t * L.toarray()) @ psi0)) < 1e-10


def test_float32_close_to_float64():
    A = random_graph(seed=5)
    Psi0, lr, lc = seed_columns(A.shape[0], [[0, 1, 2]])
    bounds = spectral_bounds(A, gamma=1.0)
    p64 = propagate(seed_loop_operator(A, 1.0, lr, lc), Psi0, [1.0, 4.0], bounds)
    p32 = propagate(seed_loop_operator(A.astype(np.float32), 1.0, lr, lc),
                    Psi0.astype(np.float32), [1.0, 4.0], bounds)
    assert p32.dtype == np.float32
    assert np.max(np.abs(p32 - p64)) < 1e-5
