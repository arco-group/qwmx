"""Data sources, gene identifier mapping and processed datasets.

Raw sources (downloaded by ``scripts/download_data.sh`` into ``data/raw``):

* ``RWR-MH``: the MX-V multiplex (PPI, PATH, COEX; HGNC symbols, 2016), the
  OMIM disease-similarity network and gene-phenotype associations
  (Valdeolivas et al., 2019);
* ``qdgp``: the GMB, DisGeNET and Open Targets disease-gene sets (Entrez IDs)
  and the PPI202207 symbol table (Saarinen et al., 2024).

``scripts/prepare_data.py`` writes the processed files read by
:func:`load_multiplex`, :func:`load_diseases` and :func:`load_heterogeneous`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

LAYERS = ("PPI", "PATH", "COEX")
DATASETS = ("gmb", "dgn", "ot")
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 0


class RawPaths:
    """Locations of the raw files under ``root`` (usually ``data/raw``)."""

    def __init__(self, root):
        self.root = Path(root)
        self.qdgp = self.root / "qdgp" / "data"
        self.rwrmh = self.root / "rwrmh" / "RWR-MH"
        self.rwrm = self.root / "rwrm" / "RWR-M"
        networks = self.rwrmh / "Networks"
        self.layers = {
            "PPI": networks / "PPI_2016-11-23.gr",
            "PATH": networks / "AllPathways_2016-11-24.gr",
            "COEX": networks / "Co-Expression_2016-11-23.gr",
        }
        self.disease_similarity = networks / "DiseaseSimilarity_2016-12-06.gr"
        self.gene_phenotype = self.rwrmh / "Input_Files" / "Gene_Phenotype_relation.txt"
        self.ppi202207 = self.qdgp / "PPI202207.txt"


# ----------------------------------------------------------------------------- raw readers

def simple_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Undirected simple edge list: no self-loops, one row ``(a, b)`` with ``a < b`` per pair."""
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    keep = a != b
    a, b = a[keep], b[keep]
    lo = np.where(a < b, a, b)
    hi = np.where(a < b, b, a)
    return pd.DataFrame({"a": lo, "b": hi}).drop_duplicates(ignore_index=True)


def read_symbol_network(path) -> pd.DataFrame:
    """Space-separated symbol edge list (``.gr`` files of RWR-MH)."""
    df = pd.read_csv(path, sep=" ", header=None, dtype=str, keep_default_na=False)
    return simple_edges(df)


def read_gene_phenotype(raw: RawPaths) -> pd.DataFrame:
    """Gene symbol to OMIM phenotype associations."""
    df = pd.read_csv(raw.gene_phenotype, sep="\t", dtype=str)
    return df.dropna().drop_duplicates(ignore_index=True)


def read_ppi202207(path) -> pd.DataFrame:
    """qdgp ``PPI202207.txt``: Entrez and symbol of both interactors."""
    return pd.read_csv(path, sep=r"\s+", header=None, skiprows=[237433 - 1],
                       dtype={0: int, 1: str, 2: int, 3: str})


def read_disease_set(raw: RawPaths, name: str) -> pd.DataFrame:
    """Disease-gene associations of ``gmb``, ``dgn`` or ``ot`` as (disease, gene) rows."""
    if name == "gmb":
        rows = []
        with (raw.qdgp / "gmb" / "pcbi.1004120.s004.tsv").open(newline="") as fh:
            reader = csv.reader(fh, delimiter=" ")
            next(reader, None)
            for row in reader:
                disease = " ".join(row[:-1]).rstrip()
                rows.extend((disease, int(g)) for g in row[-1].split("/"))
        return pd.DataFrame(rows, columns=["disease", "gene"])
    if name == "dgn":
        df = pd.read_csv(raw.qdgp / "DisGenet_filtered.csv",
                         usecols=["score", "geneId", "diseaseName", "type", "EL", "EI",
                                  "source", "DSI"]).drop_duplicates()
        df = df[df.type == "disease"][["diseaseName", "geneId"]].dropna()
    elif name == "ot":
        df = pd.read_csv(raw.qdgp / "OpenTargetsIDs_filtered.csv").drop_duplicates()
        df = df[["Disease_Name", "id"]].dropna()
    else:
        raise ValueError(f"unknown disease set {name!r}")
    df.columns = ["disease", "gene"]
    df["gene"] = df.gene.astype(int)
    return df.drop_duplicates(ignore_index=True)


# ----------------------------------------------------------------------------- symbol mapping

MAPPING_STEPS = ("symbol", "prev_symbol", "alias_symbol", "ppi202207")


def map_symbols(symbols, hgnc_path, ppi202207_path=None) -> pd.DataFrame:
    """Map HGNC symbols to Entrez Gene IDs.

    A symbol is resolved by the first rule that matches it: current approved
    symbol, previous symbol, alias, then the symbol columns of qdgp
    ``PPI202207.txt``. A rule that matches several Entrez IDs leaves the symbol
    ambiguous; ambiguous and unresolved symbols are dropped.

    Parameters
    ----------
    symbols : iterable of str
    hgnc_path : path
        HGNC complete set (``hgnc_complete_set.txt``).
    ppi202207_path : path, optional

    Returns
    -------
    DataFrame
        Columns ``symbol``, ``entrez`` (nullable) and ``step`` (the rule used,
        ``unresolved`` or ``ambiguous:<rule>``).
    """
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str,
                       usecols=["symbol", "status", "prev_symbol", "alias_symbol", "entrez_id"])
    hgnc = hgnc[(hgnc.status == "Approved") & hgnc.entrez_id.notna()].copy()
    hgnc["entrez_id"] = hgnc.entrez_id.astype(float).astype(int)

    def cells(cell):
        if not isinstance(cell, str) or not cell:
            return []
        return [s.strip().strip('"') for s in cell.strip('"').split("|") if s.strip()]

    tables = {k: defaultdict(set) for k in MAPPING_STEPS}
    for sym, prev, alias, eid in hgnc[["symbol", "prev_symbol", "alias_symbol",
                                       "entrez_id"]].itertuples(index=False):
        tables["symbol"][sym].add(eid)
        for p in cells(prev):
            tables["prev_symbol"][p].add(eid)
        for a in cells(alias):
            tables["alias_symbol"][a].add(eid)
    if ppi202207_path is not None:
        wl = read_ppi202207(ppi202207_path)
        for e_col, s_col in ((0, 1), (2, 3)):
            for eid, sym in wl[[e_col, s_col]].drop_duplicates().itertuples(index=False):
                tables["ppi202207"][sym].add(int(eid))

    rows = []
    for sym in sorted(set(symbols)):
        step_used, eid = "unresolved", None
        for step in MAPPING_STEPS:
            hits = tables[step].get(sym)
            if not hits:
                continue
            if len(hits) == 1:
                step_used, eid = step, next(iter(hits))
            else:
                step_used = f"ambiguous:{step}"
            break
        rows.append((sym, eid, step_used))
    return pd.DataFrame(rows, columns=["symbol", "entrez", "step"]).astype({"entrez": "Int64"})


# ----------------------------------------------------------------------------- processed data

def load_multiplex(processed, layers=LAYERS) -> tuple[np.ndarray, list[sps.csr_matrix]]:
    """Gene universe (Entrez IDs) and the requested MX-V layers."""
    z = np.load(Path(processed) / "mxv_layers.npz")
    universe = z["universe"]
    n = len(universe)
    mats = [sps.csr_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]), shape=(n, n))
            for k in layers]
    return universe, mats


def load_diseases(processed, name: str) -> dict[str, list[int]]:
    """Disease name to sorted gene positions in the universe."""
    df = pd.read_csv(Path(processed) / f"diseases_{name}.csv")
    return {d: sorted(g.gene_idx.unique().tolist()) for d, g in df.groupby("disease")}


def load_heterogeneous(processed):
    """Gene-disease incidence ``B`` (N, M), disease adjacency ``D`` (M, M) and OMIM IDs."""
    processed = Path(processed)
    B = sps.load_npz(processed / "omim_bipartite.npz").tocsr()
    D = sps.load_npz(processed / "omim_similarity.npz").tocsr()
    omim = pd.read_csv(processed / "omim_diseases.csv").omim.to_numpy()
    return B, D, omim


def disease_split(diseases: dict) -> dict[str, str]:
    """Assign 20% of the diseases to validation, stratified by gene-set size.

    Diseases are sorted by size and cut into five strata; each stratum is
    shuffled with a fixed seed and its first 20% go to validation.

    Parameters
    ----------
    diseases : dict
        Disease name to genes.

    Returns
    -------
    dict
        Disease name to ``"validation"`` or ``"test"``.
    """
    names = sorted(diseases)
    sizes = np.array([len(diseases[d]) for d in names])
    rng = np.random.default_rng(SPLIT_SEED)
    order = np.argsort(sizes, kind="stable")
    split = {}
    for stratum in np.array_split(order, 5):
        stratum = rng.permutation(stratum)
        n_val = int(round(VALIDATION_FRACTION * len(stratum)))
        for k, i in enumerate(stratum):
            split[names[i]] = "validation" if k < n_val else "test"
    return split
