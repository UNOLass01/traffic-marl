"""Job order of the sweep runner: a deadline must only cost the lowest priorities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sweep_module():
    sys.path.insert(0, str(ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location("sweep", ROOT / "experiments" / "sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flat_sweep_runs_seed_by_seed():
    sweep = {"seeds": [0, 1], "runs": [{"config": "a"}, {"config": "b"}]}
    jobs = _sweep_module().expand_jobs(sweep)
    assert [(item["config"], seed) for item, seed in jobs] == [
        ("a", 0),
        ("b", 0),
        ("a", 1),
        ("b", 1),
    ]


def test_tiers_keep_priority_order():
    sweep = {
        "tiers": [
            {"seeds": [0, 1], "runs": [{"config": "first"}]},
            {"seeds": [3], "runs": [{"config": "second"}]},
        ]
    }
    jobs = _sweep_module().expand_jobs(sweep)
    assert [(item["config"], seed) for item, seed in jobs] == [
        ("first", 0),
        ("first", 1),
        ("second", 3),
    ]


def test_stop_time_starts_no_job(tmp_path, monkeypatch):
    from datetime import datetime

    module = _sweep_module()
    monkeypatch.setattr(module.paths, "RESULTS_DIR", tmp_path)
    marker = tmp_path / "ran"
    job = [sys.executable, "-c", f"open({str(marker)!r}, 'w')"]
    assert module.run_parallel([job], 1, datetime(2000, 1, 1), "test") == 0
    assert not marker.exists()
    assert module.run_parallel([job], 1, None, "test") == 0
    assert marker.exists()
