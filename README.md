# Scalable Time-Averaged Quantum Walks for Disease Gene Prioritization on Multiplex Biological Networks

Code for the paper

> M. Tortora, F. Di Feola, I. Iele, F. Caprio, P. Soda. *Scalable Time-Averaged Quantum Walks for Disease Gene Prioritization on Multiplex Biological Networks.* 17th International Workshop on High Performance Bioinformatics and Biomedicine (HiBB), IEEE BIBM 2026.

The repository contains:

- **QW-M-TA**, a continuous-time quantum walk on a multiplex gene network whose
  gene scores are occupation probabilities summed over layers and averaged over
  `m` evolution times in a window `T`, and **QW-MH-TA**, its extension to
  heterogeneous gene-disease networks;
- a **batched multi-time Chebyshev propagator** that computes all evolution
  times from one recurrence in real arithmetic, propagates many diseases at once
  (each with its own seed self-loops and edge edits), and runs on CPU (NumPy/SciPy)
  or GPU (CuPy) with the same code;
- the baselines (DK-M, RWR-M, RWR-MH, QA, DK, RWR, NEI, Degree), the evaluation
  protocols, and the configurations and scripts that reproduce the tables and
  figures of the paper.

All quantum dynamics are simulated classically.

## Installation

```bash
git clone https://github.com/arco-group/qwmx.git
cd qwmx
pip install -e .            # Python >= 3.10
pip install cupy-cuda12x    # optional, GPU backend; pick the package matching your CUDA
```

The results in the paper were obtained with Python 3.10, NumPy 2.2, SciPy 1.15
and CuPy 14.2 on an NVIDIA GH200 node.

## Usage

```python
import numpy as np
from qwmx.walks import qw_m_ta_scores

# layers: list of N x N symmetric scipy.sparse adjacency matrices on the same genes
# seed_sets: list of lists of gene indices, one per disease
scores = qw_m_ta_scores(layers, seed_sets, delta=0.0, gamma=0.0, window=256.0, m=8)
ranking = np.argsort(-scores[:, 0])   # candidate ranking for the first disease (seeds included)
```

Pass `xp=cupy` to run on the GPU. The building blocks are
`qwmx.propagate.propagate` (Chebyshev propagation of a block of states to several
times), `qwmx.walks.iter_scores` (batched scoring of many seed sets) and
`qwmx.heterogeneous` (QW-MH and leave-one-out tasks).

## Reproducing the paper

### Data

```bash
./scripts/download_data.sh          # about 100 MB into data/raw
python scripts/prepare_data.py      # data/processed; prints Table I
```

The networks and disease-gene sets come from the repositories of
[Valdeolivas et al. (2019)](https://github.com/alberto-valdeolivas/RWR-MH) and
[Saarinen et al. (2024)](https://github.com/markgolds/qdgp), at fixed commits.
Gene symbols are mapped to Entrez IDs with `data/hgnc_symbol_map.csv`, built
from the HGNC release of 13 October 2026 (`prepare_data.py --hgnc` rebuilds it
from another release).

### Experiments

Every experiment is a configuration file in `configs/` run by one script.
Outputs are long-format CSV files (one row per task and metric) that
`scripts/analyze.py` summarizes. Add `--backend cupy` to run the quantum walks
and diffusion kernels on the GPU. GPU runs are not bitwise
deterministic, so results can differ slightly (up to the third decimal).

| Paper | Configuration | Script |
|---|---|---|
| Table III, Section V-A | `opentargets.yaml` | `prioritize.py` |
| Fig. 2, Section V-B | `temporal_gmb.yaml`, `temporal_dgn.yaml` | `prioritize.py` |
| Layer subsets, Section V-B | `layers_gmb.yaml`, `layers_dgn.yaml` | `prioritize.py` |
| Table IV, Section V-C | `loocv.yaml` | `loocv.py` |
| Table V, Fig. 3, Section V-D | | `benchmark.py` |

**Table III and RQ1.**

```bash
python scripts/prioritize.py configs/opentargets.yaml --backend cupy --degree \
    --ranks results/opentargets_ranks.csv.gz --out results/opentargets.csv.gz
python scripts/analyze.py summary results/opentargets.csv.gz
python scripts/analyze.py test results/opentargets.csv.gz --metric ap \
    --pairs QW-M-TA:QW-M-ST QW-M-TA:DK-M QW-M-TA:RWR-M
python scripts/analyze.py union results/opentargets_ranks.csv.gz --a QW-M-TA --b RWR-M --k 100
```

Repeat `test` with `--metric recall@25`, `recall@100` or `recall@300` for the
other comparisons in Section V-A.

**Fig. 2 and layer subsets.**

```bash
for s in gmb dgn; do
    python scripts/prioritize.py configs/temporal_$s.yaml --backend cupy --degree \
        --out results/temporal_$s.csv.gz
    python scripts/analyze.py summary results/temporal_$s.csv.gz \
        --metrics ap spearman_degree recall@300_t0
    python scripts/prioritize.py configs/layers_$s.yaml --backend cupy --out results/layers_$s.csv.gz
done
```

The temporal summary gives, for every method and time, the mean AP, the
Spearman correlation between scores and merged-network degree, and Recall@300
on low-degree genes.

**Table IV and RQ3.**

```bash
python scripts/loocv.py configs/loocv.yaml --backend cupy --out results/loocv.csv.gz
python scripts/analyze.py summary results/loocv.csv.gz
python scripts/analyze.py test results/loocv.csv.gz --metric rr --pairs \
    "QW-MH-TA:RWR-MH (ext.)" "QW-MH-TA:RWR-MH (defaults)" "QW-MH-TA:RWR-MH (3x3)" \
    QW-MH-TA:QW-MH-ST QW-MH-TA:QW-M-TA QW-M-TA:RWR-M
```

**Table V and Fig. 3.** Runtimes depend on the hardware; the paper used one
GH200 GPU and 16 Grace CPU cores.

```bash
# CPU timings use one thread per process; 16 processes at B = 512 and n_t = 20 need about 65 GB
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Table V, 10 diseases: single-core reference and Chebyshev on CPU and GPU
python scripts/benchmark.py throughput --max-diseases 10 --B 512 --reference
python scripts/benchmark.py throughput --max-diseases 10 --B 512 --backend cupy
# Table V, 62 diseases: 16 CPU processes and GPU
python scripts/benchmark.py throughput --workers 16 --B 32 512 --reference
python scripts/benchmark.py throughput --B 32 --backend cupy
# Fig. 3: GPU batch width (a) and dense versus Chebyshev (b)
python scripts/benchmark.py throughput --B 1 8 32 128 512 --n-t 20 --backend cupy
python scripts/benchmark.py dense --n-t 20
# Section V-D: accuracy against expm and expm_multiply
python scripts/benchmark.py accuracy
```

### Hyperparameter selection

The configurations in `configs/opentargets.yaml`, `temporal_*.yaml`,
`layers_*.yaml` and `loocv.yaml` were selected on validation diseases.
`configs/validation_{dgn,gmb}.yaml` and `configs/loocv_validation.yaml` list the
values that were evaluated. The selection can be rerun with:

```bash
python scripts/prioritize.py configs/validation_dgn.yaml --backend cupy --out results/validation_dgn.csv.gz
python scripts/analyze.py summary results/validation_dgn.csv.gz --top 1
python scripts/loocv.py configs/loocv_validation.yaml --backend cupy --out results/loocv_validation.csv.gz
python scripts/analyze.py summary results/loocv_validation.csv.gz
```

For the single-graph baselines (QA, DK, RWR) the paper reports the graph with
the highest validation AP.

## Repository layout

```
src/qwmx/
  propagate.py       batched multi-time Chebyshev propagation
  walks.py           QW-M (single-time and time-averaged), QA, DK-M, DK
  heterogeneous.py   QW-MH and leave-one-out tasks
  rwr.py             RWR, RWR-M, RWR-MH, NEI, Degree
  evaluation.py      splits, ranking metrics, score-degree metrics
  stats.py           Wilcoxon signed-rank tests with Holm correction
  data.py            data sources, symbol mapping, processed datasets
  config.py          experiment configuration files
scripts/             download_data.sh, prepare_data.py, prioritize.py, loocv.py,
                     analyze.py, benchmark.py
configs/             one file per experiment
tests/               correctness tests (pytest)
```

`pytest` checks the propagator against dense `expm` and `expm_multiply`, the
batched walks against dense computations, the metrics against scikit-learn,
and, once the data is downloaded, RWR-M and RWR-MH against the outputs of the
original R implementation.

## Citation

```bibtex
@inproceedings{tortora2026scalable,
  title     = {Scalable Time-Averaged Quantum Walks for Disease Gene Prioritization on Multiplex Biological Networks},
  author    = {Tortora, Matteo and Di Feola, Francesco and Iele, Irene and Caprio, Floriano and Soda, Paolo},
  booktitle = {2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

## License

The code is released under the MIT License. The data are not redistributed:
`download_data.sh` fetches them from their original repositories, whose terms
apply. `data/hgnc_symbol_map.csv` is derived from HGNC data (CC0).
