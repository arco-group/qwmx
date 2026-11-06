"""Expansion of configuration blocks."""

from qwmx.config import config_id, expand


def test_lists_are_grids_except_times_and_layers():
    block = {"method": "QW-M", "delta": [0.0, 0.1], "gamma": 1.0, "t": [1, 2], "layers": ["PATH"]}
    configs = expand(block)
    assert [p["delta"] for p in configs] == [0.0, 0.1]
    assert all(p["t"] == [1.0, 2.0] and p["layers"] == ["PATH"] for p in configs)


def test_config_id_is_readable():
    assert config_id({"method": "QW-M", "batch": 128, "delta": 0.0, "layers": ["PATH"], "t": 16.0}) \
        == "delta=0|layers=PATH|t=16"
