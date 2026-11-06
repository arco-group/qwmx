"""Experiment configuration files.

A configuration file names a dataset, a split and a set of methods::

    dataset: ot
    split: all
    methods:
      QW-M-TA: {method: QW-M, init: coherent, delta: 0.0, gamma: 0.0, T: 256}

Each method label maps to one block of parameters or to a list of blocks. A
list-valued parameter is a grid: a block is expanded into the product of its
lists. Evolution times ``t`` and averaging windows ``T`` are not expanded,
because all of them are computed from a single propagation, and ``layers`` is
one set of layers rather than a grid.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

TIME_KEYS = ("t", "T")
WHOLE_KEYS = ("layers",)
NOT_IN_ID = ("method", "batch")


def load_config(path) -> dict:
    """Read a configuration file and normalize every method to a list of blocks."""
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["methods"] = {label: blocks if isinstance(blocks, list) else [blocks]
                      for label, blocks in cfg["methods"].items()}
    return cfg


def as_list(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def expand(block: dict) -> list[dict]:
    """Expand a block into configurations; time lists are kept whole.

    Returns
    -------
    list of dict
        One dict per combination of the non-time parameters. ``t`` or ``T``,
        if present, is a list of floats.
    """
    keys = [k for k in block if k not in TIME_KEYS + WHOLE_KEYS]
    out = []
    for values in itertools.product(*[as_list(block[k]) for k in keys]):
        p = dict(zip(keys, values))
        for k in WHOLE_KEYS:
            if k in block:
                p[k] = as_list(block[k])
        for k in TIME_KEYS:
            if k in block:
                p[k] = [float(x) for x in as_list(block[k])]
        out.append(p)
    return out


def config_id(params: dict) -> str:
    """Readable identifier of one configuration, e.g. ``delta=0.1|gamma=0|t=16``."""
    parts = []
    for k, v in params.items():
        if k in NOT_IN_ID:
            continue
        if isinstance(v, list):
            v = "+".join(str(x) for x in v)
        elif isinstance(v, float):
            v = f"{v:g}"
        parts.append(f"{k}={v}")
    return "|".join(parts)
