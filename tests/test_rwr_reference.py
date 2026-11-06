"""RWR-M and RWR-MH reproduce the example outputs of the original R code.

Runs on the original symbol networks, without identifier mapping. Requires the
raw data (scripts/download_data.sh).
"""

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from scipy.stats import spearmanr

from qwmx.data import LAYERS, RawPaths, read_gene_phenotype, read_symbol_network
from qwmx.graphs import adjacency
from qwmx.rwr import rwr_m_scores, rwr_m_transition, rwr_mh_scores, rwr_mh_transition

RAW = RawPaths(os.environ.get("QWMX_RAW", "data/raw"))
pytestmark = pytest.mark.skipif(not RAW.rwrmh.exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def multiplex():
    edges = [read_symbol_network(RAW.layers[k]) for k in LAYERS]
    pool = sorted(set().union(*[set(e.a) | set(e.b) for e in edges]))
    index = {g: i for i, g in enumerate(pool)}
    return pool, index, [adjacency(e, index, len(pool)) for e in edges]


def compare(ours: pd.Series, path, key):
    ref = pd.read_csv(path, sep="\t", dtype={key: str})
    assert len(ref) == len(ours)
    mine = ours.loc[ref[key]].to_numpy()
    rho = spearmanr(mine, ref.Score.to_numpy()).statistic
    top = list(ours.sort_values(ascending=False, kind="stable").index[:100])
    return rho, top == list(ref[key][:100])


def test_rwr_m_reference(multiplex):
    pool, index, layers = multiplex
    seeds = ["TP53", "LMNB1"]
    s = rwr_m_scores(rwr_m_transition(layers, 0.5), [[index[g] for g in seeds]], len(pool), 3,
                     r=0.7)[:, 0]
    rho, same_top = compare(pd.Series(s, index=pool).drop(seeds),
                            RAW.rwrm / "Output_Files" / "Results_Example.txt", "GeneNames")
    assert rho > 0.999999 and same_top


def test_rwr_mh_reference(multiplex):
    pool, index, layers = multiplex
    d_edges = read_symbol_network(RAW.disease_similarity)
    diseases = sorted(set(d_edges.a) | set(d_edges.b))
    d_index = {d: i for i, d in enumerate(diseases)}
    D = adjacency(d_edges, d_index, len(diseases))
    gp = read_gene_phenotype(RAW)
    gp = gp[gp.hgnc_symbol.isin(index) & gp.mim_morbid.isin(d_index)]
    B = sps.csr_matrix((np.ones(len(gp)), (gp.hgnc_symbol.map(index), gp.mim_morbid.map(d_index))),
                       shape=(len(pool), len(diseases)))
    B.data[:] = 1.0
    T = rwr_mh_transition(layers, 0.5, B, D, lam=0.5)
    g, d = rwr_mh_scores(T, [[]], [[d_index["264090"]]], len(pool), 3, len(diseases),
                         r=0.7, eta=0.5)
    out = RAW.rwrmh / "Output_Files"
    rho, same_top = compare(pd.Series(g[:, 0], index=pool), out / "Results_Example_Genes.txt",
                            "GeneNames")
    assert rho > 0.999999 and same_top
    rho, same_top = compare(pd.Series(d[:, 0], index=diseases).drop("264090"),
                            out / "Results_Example_Diseases.txt", "DiseaseID")
    assert rho > 0.999999 and same_top
