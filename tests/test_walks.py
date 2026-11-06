"""Batched multiplex walks against dense computations."""

import numpy as np
import pytest
from scipy.linalg import expm

from qwmx.graphs import laplacian, supra_adjacency
from qwmx.walks import iter_scores, multiplex_operator, qw_m_ta_scores, window_times
from test_propagate import random_graph

SEED_SETS = [[0, 5, 9], [3, 40], [12, 13, 70, 71, 2], [7]]


def collect(gen, n_sets):
    out = {}
    for sets, sc in gen:
        for k, j in enumerate(sets):
            out[j] = sc[:, :, k]
    return [out[j] for j in range(n_sets)]


@pytest.mark.parametrize("batch_cols", [1, 3, 4, 512])
def test_batch_width_does_not_change_scores(batch_cols):
    A = random_graph()
    ref = collect(iter_scores(A, SEED_SETS, [0.45, 2.0], gamma=5.0, batch_cols=512), 4)
    got = collect(iter_scores(A, SEED_SETS, [0.45, 2.0], gamma=5.0, batch_cols=batch_cols), 4)
    for r, g in zip(ref, got):
        assert np.max(np.abs(r - g)) < 1e-12


@pytest.mark.parametrize("init", ["incoherent", "coherent"])
def test_qw_m_matches_dense(init):
    layers = [random_graph(n=40, seed=s) for s in (10, 11, 12)]
    W = supra_adjacency(layers, 0.5)
    sets = [[1, 2, 3], [20, 30]]
    got = collect(iter_scores(W, sets, [1.0, 4.0], gamma=1.0, n_layers=3, init=init,
                              batch_cols=5), 2)
    for j, S in enumerate(sets):
        H = W.toarray()
        R = (np.arange(3)[:, None] * 40 + np.asarray(S)[None, :]).ravel()
        H[R, R] += 1.0
        for i, t in enumerate([1.0, 4.0]):
            U = expm(-1j * t * H)
            if init == "incoherent":
                p = (np.abs(U[:, R]) ** 2).sum(axis=1)
            else:
                p = np.abs(U[:, R].sum(axis=1) / np.sqrt(len(R))) ** 2
            assert np.max(np.abs(got[j][i] - p.reshape(3, 40).sum(axis=0))) < 1e-10


def test_dk_m_matches_dense():
    layers = [random_graph(n=40, seed=s) for s in (20, 21)]
    Lsup = laplacian(supra_adjacency(layers, 0.25))
    got = collect(iter_scores(Lsup, [[4, 5, 6]], [0.5, 2.0], kind="diffusion", n_layers=2), 1)[0]
    mask = np.zeros(80)
    mask[[4, 5, 6, 44, 45, 46]] = 1
    for i, t in enumerate([0.5, 2.0]):
        ref = (expm(-t * Lsup.toarray()) @ mask).reshape(2, 40).sum(axis=0)
        assert np.max(np.abs(got[i] - ref)) < 1e-10


def test_time_average_is_mean_of_single_times():
    layers = [random_graph(n=40, seed=s) for s in (30, 31, 32)]
    sets = [[1, 2, 3], [20, 30]]
    got = qw_m_ta_scores(layers, sets, delta=0.1, gamma=0.5, window=16.0, m=8)
    times, index = window_times([16.0], 8)
    assert times == [2.0 * j for j in range(1, 9)] and index[16.0] == list(range(8))
    W = multiplex_operator(layers, 0.1)
    ref = collect(iter_scores(W, sets, times, gamma=0.5, n_layers=3, init="coherent"), 2)
    for j in range(2):
        np.testing.assert_allclose(got[:, j], ref[j].mean(axis=0), atol=1e-12)
