"""Build the processed datasets from the raw downloads and print their statistics (Table I).

Outputs (in ``--out``):

* ``mxv_layers.npz``: PPI, PATH and COEX layers on the Entrez gene universe;
* ``diseases_{gmb,dgn,ot}.csv``: disease-gene associations restricted to the
  universe, diseases with at least 15 genes;
* ``omim_similarity.npz``, ``omim_bipartite.npz``, ``omim_diseases.csv``:
  OMIM disease network and gene-disease incidence for the heterogeneous network.

Gene symbols are mapped to Entrez IDs with ``data/hgnc_symbol_map.csv``. Pass
``--hgnc`` to rebuild that table from an HGNC release instead; a release other
than 2026-10-13 may map a few symbols differently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

from qwmx.data import (DATASETS, LAYERS, RawPaths, map_symbols, read_disease_set,
                       read_gene_phenotype, read_symbol_network, simple_edges)
from qwmx.graphs import adjacency, degrees

MIN_GENES = 15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--symbol-map", default="data/hgnc_symbol_map.csv")
    ap.add_argument("--hgnc", default=None, help="rebuild the symbol map from this HGNC file")
    args = ap.parse_args()
    raw = RawPaths(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sym_edges = {k: read_symbol_network(raw.layers[k]) for k in LAYERS}
    gp = read_gene_phenotype(raw)
    if args.hgnc:
        pool = set().union(*[set(e.a) | set(e.b) for e in sym_edges.values()])
        table = map_symbols(pool | set(gp.hgnc_symbol), args.hgnc, raw.ppi202207)
        table.to_csv(args.symbol_map, index=False)
    table = pd.read_csv(args.symbol_map, dtype={"symbol": str}, keep_default_na=False,
                        na_values={"entrez": [""]})
    table = table.dropna(subset=["entrez"])
    mapping = dict(zip(table.symbol, table.entrez.astype(int)))

    layer_edges = {}
    for name, e in sym_edges.items():
        a, b = e.a.map(mapping), e.b.map(mapping)
        keep = a.notna() & b.notna()
        layer_edges[name] = simple_edges(pd.DataFrame({"a": a[keep].astype(int),
                                                       "b": b[keep].astype(int)}))
    universe = np.array(sorted(set().union(*[set(e.a) | set(e.b) for e in layer_edges.values()])),
                        dtype=np.int64)
    index = {int(g): i for i, g in enumerate(universe)}
    N = len(universe)
    layers = {k: adjacency(e, index, N) for k, e in layer_edges.items()}
    np.savez_compressed(out / "mxv_layers.npz", universe=universe,
                        **{f"{k}_{f}": getattr(A, f) for k, A in layers.items()
                           for f in ("data", "indices", "indptr")})

    print(f"MX-V: {N} genes, {N * len(LAYERS)} gene-layer nodes")
    print(f"{'Layer':8s} {'Active genes':>12s} {'Edges':>10s} {'Mean degree':>12s}")
    for k, A in layers.items():
        d = degrees(A)
        print(f"{k:8s} {int((d > 0).sum()):12,d} {A.nnz // 2:10,d} {d[d > 0].mean():12.1f}")

    print(f"\n{'Dataset':8s} {'Diseases':>9s} {'Assoc.':>8s} {'Median genes':>13s}")
    for name in DATASETS:
        df = read_disease_set(raw, name)
        df = df[df.gene.isin(index)].drop_duplicates()
        df = df[df.groupby("disease").gene.transform("size") >= MIN_GENES].copy()
        df["gene_idx"] = df.gene.map(index)
        df.rename(columns={"gene": "entrez"}).to_csv(out / f"diseases_{name}.csv", index=False)
        sizes = df.groupby("disease").size()
        print(f"{name:8s} {len(sizes):9,d} {len(df):8,d} {sizes.median():13.0f}")

    d_edges = read_symbol_network(raw.disease_similarity)
    omim = sorted(set(d_edges.a) | set(d_edges.b))
    d_index = {d: i for i, d in enumerate(omim)}
    D = adjacency(d_edges, d_index, len(omim))
    gp = gp.assign(entrez=gp.hgnc_symbol.map(mapping)).dropna(subset=["entrez"])
    gp = gp[gp.entrez.astype(int).isin(index) & gp.mim_morbid.isin(d_index)]
    rows = gp.entrez.astype(int).map(index).to_numpy()
    cols = gp.mim_morbid.map(d_index).to_numpy()
    B = sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, len(omim)))
    B.data[:] = 1.0
    sps.save_npz(out / "omim_similarity.npz", D)
    sps.save_npz(out / "omim_bipartite.npz", B)
    pd.DataFrame({"position": np.arange(len(omim)), "omim": omim}).to_csv(
        out / "omim_diseases.csv", index=False)
    n_genes = np.asarray((B > 0).sum(axis=0)).ravel()
    print(f"\nOMIM: {len(omim):,d} diseases, {D.nnz // 2:,d} similarity edges, "
          f"{B.nnz:,d} gene-disease associations, {(n_genes >= 2).sum()} diseases with >= 2 genes")


if __name__ == "__main__":
    main()
