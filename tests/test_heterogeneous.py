"""Per-column edits and leave-one-out tasks against explicit matrices."""

import numpy as np
import scipy.sparse as sps
from scipy.linalg import expm

from qwmx.heterogeneous import (edit_bounds, iter_qw_mh_scores, qw_mh_columns, qw_mh_hamiltonian,
                                rwr_mh_loocv_scores)
from qwmx.propagate import column_edit_operator
from qwmx.rwr import geometric_mean_layers, mh_seed_vector, random_walk_restart, rwr_mh_transition
from qwmx.walks import window_times


def random_graph(n, p, rng):
    A = sps.random(n, n, density=p, random_state=rng, data_rvs=np.ones)
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    return sps.csr_matrix(A)


def network(rng, N=30, L=2, M=8):
    layers = [random_graph(N, 0.2, rng) for _ in range(L)]
    D = random_graph(M, 0.4, rng)
    B = sps.lil_matrix((rng.random((N, M)) < 0.15).astype(float))
    B[3, 2] = B[5, 2] = B[7, 2] = 1.0
    return layers, D, sps.csr_matrix(B), N, L, M


def test_column_edit_operator():
    rng = np.random.default_rng(0)
    A = random_graph(12, 0.3, rng)
    X = rng.standard_normal((12, 3))
    rows, srcs, cols, vals = [1, 1, 4, 2], [5, 6, 4, 0], [0, 0, 2, 1], [0.5, -2.0, 3.0, 1.5]
    out = column_edit_operator(A, rows, srcs, cols, vals)(X)
    for j in range(3):
        E = A.toarray()
        for r, s, c, v in zip(rows, srcs, cols, vals):
            if c == j:
                E[r, s] += v
        np.testing.assert_allclose(out[:, j], E @ X[:, j], atol=1e-12)


def test_qw_mh_loocv_matches_dense():
    rng = np.random.default_rng(1)
    layers, D, B, N, L, M = network(rng)
    kappa, eta, gamma, delta, t = 0.4, 0.5, 1.0, 0.3, 2.0
    H = qw_mh_hamiltonian(layers, delta, B, D, kappa)
    tasks = [(2, 3, [5, 7]), (2, 5, [3, 7])]
    cols = qw_mh_columns(tasks, N, L, B, kappa=kappa, eta=eta, gamma=gamma)
    _, sc = next(iter_qw_mh_scores(H, cols, [t], N, L, edit_bounds(H, kappa=kappa, gamma=gamma, L=L)))
    for j, (d, g, S) in enumerate(tasks):
        Hd = H.toarray()
        dn = N * L + d
        for l in range(L):
            Hd[l * N + g, dn] = Hd[dn, l * N + g] = 0.0
        psi0 = np.zeros(Hd.shape[0])
        R = [l * N + s for l in range(L) for s in S]
        psi0[R] = np.sqrt((1 - eta) / len(R))
        psi0[dn] = np.sqrt(eta)
        for r in R + [dn]:
            Hd[r, r] += gamma
        p = np.abs(expm(-1j * t * Hd) @ psi0) ** 2
        np.testing.assert_allclose(sc[0, :, j], p[: N * L].reshape(L, N).sum(axis=0), atol=1e-10)


def test_qw_mh_time_average_is_mean_of_single_times():
    rng = np.random.default_rng(2)
    layers, D, B, N, L, M = network(rng)
    H = qw_mh_hamiltonian(layers, 0.3, B, D, 0.4)
    cols = qw_mh_columns([(2, 3, [5, 7])], N, L, B, kappa=0.4, eta=0.5, gamma=1.0)
    bounds = edit_bounds(H, kappa=0.4, gamma=1.0, L=L)
    times, windows = window_times([2.0, 4.0], 8)
    assert times[:3] == [0.25, 0.5, 0.75] and len(times) == 12
    _, sc = next(iter_qw_mh_scores(H, cols, times, N, L, bounds))
    for T, w in windows.items():
        _, ref = next(iter_qw_mh_scores(H, cols, [T * j / 8 for j in range(1, 9)], N, L, bounds))
        np.testing.assert_allclose(sc[w].mean(axis=0), ref.mean(axis=0), atol=1e-10)


def test_rwr_mh_loocv_matches_zeroed_transition():
    rng = np.random.default_rng(2)
    layers, D, B, N, L, M = network(rng)
    T = rwr_mh_transition(layers, 0.5, B, D, 0.5)
    tasks = [(2, 3, [5, 7]), (2, 7, [3, 5])]
    got = rwr_mh_loocv_scores(T, tasks, N, L, M, r=0.7, eta=0.5)
    for j, (d, g, S) in enumerate(tasks):
        Tz = T.tolil()
        for l in range(L):
            Tz[l * N + g, N * L + d] = 0.0
            Tz[N * L + d, l * N + g] = 0.0
        p = random_walk_restart(sps.csr_matrix(Tz), mh_seed_vector([S], [[d]], N, L, M, 0.5), 0.7)
        np.testing.assert_allclose(got[:, j], geometric_mean_layers(p[: N * L], L, N)[:, 0],
                                   atol=1e-12)
