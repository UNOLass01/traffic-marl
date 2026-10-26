import copy

import pytest

from trafficmarl.scenarios.spec import load_scenario_config


@pytest.fixture
def tiny_single_cfg(tmp_path):
    """The `single` scenario shrunk to 5 simulated minutes, built under tmp_path."""
    cfg = copy.deepcopy(load_scenario_config("single"))
    cfg["output_dir"] = str(tmp_path / "single")
    cfg["demand"]["duration"] = 300
    cfg["demand"]["profiles"] = {"medium": [1.0], "peak": [0.5, 1.5, 0.5]}
    cfg["demand"]["splits"] = {"train": [1], "eval": [2]}
    cfg["validation"]["end"] = 900
    return cfg
