"""Metrics: parsing of SUMO outputs, delay definitions, signal timing, statistics, evaluation."""

import gzip
import multiprocessing
import shutil

import numpy as np
import pytest

from trafficmarl.metrics import episode as ep
from trafficmarl.metrics import stats

TRIPINFO = """<tripinfos>
  <tripinfo id="a" depart="0.00" departDelay="1.00" arrival="60.00" duration="60.00"
      waitingTime="10.00" timeLoss="20.00"><emissions CO2_abs="100000.0"/></tripinfo>
  <tripinfo id="b" depart="10.00" departDelay="3.00" arrival="90.00" duration="80.00"
      waitingTime="30.00" timeLoss="40.00"><emissions CO2_abs="300000.0"/></tripinfo>
  <tripinfo id="c" depart="50.00" departDelay="0.00" arrival="-1.00" duration="50.00"
      waitingTime="45.00" timeLoss="48.00"><emissions CO2_abs="200000.0"/></tripinfo>
</tripinfos>"""

ROUTES = """<routes>
  <vehicle id="a" depart="0.00"/><vehicle id="b" depart="7.00"/>
  <vehicle id="c" depart="50.00"/><vehicle id="d" depart="80.00"/>
  <vehicle id="e" depart="150.00"/>
</routes>"""


@pytest.fixture
def outputs(tmp_path):
    (tmp_path / "tripinfo.xml").write_text(TRIPINFO)
    with gzip.open(tmp_path / "r.rou.xml.gz", "wt") as fh:
        fh.write(ROUTES)
    (tmp_path / "summary.xml").write_text(
        '<summary><step time="0" halting="2"/><step time="1" halting="4"/></summary>'
    )
    return tmp_path


def test_delay_metrics_include_unfinished_and_never_inserted(outputs):
    records = ep.trip_records(outputs / "tripinfo.xml")
    departures = ep.intended_departures(outputs / "r.rou.xml.gz")
    m = ep.delay_metrics(records, departures, end=100.0)
    # e departs after the end: not due. d is due but never inserted.
    assert m["vehicles_due"] == 4 and m["inserted"] == 3
    assert (m["arrived"], m["unfinished_in_network"], m["never_inserted"]) == (2, 1, 1)
    assert m["unfinished_total"] == 2 and m["throughput"] == 2
    assert m["mean_waiting_time"] == pytest.approx(20.0)  # arrived vehicles only
    # Insertion delay: 1, 3, 0 and 100 - 80 = 20 for the vehicle never inserted.
    assert m["mean_insertion_delay"] == pytest.approx((1 + 3 + 0 + 20) / 4)
    # Total delay: departDelay + timeLoss (21, 43, 48) and 20 for d.
    assert m["mean_total_delay"] == pytest.approx((21 + 43 + 48 + 20) / 4)
    assert m["co2_total_kg"] == pytest.approx(0.6)
    assert m["max_waiting_time"] == 45.0  # the unfinished vehicle counts


def test_network_queue_is_time_average(outputs):
    assert ep.network_queue(outputs / "summary.xml") == pytest.approx(3.0)


def test_signal_timing_longest_red_and_changes():
    greens = {"t": ["GGrr", "rrGG"]}
    intervals = {
        "t": [
            ("GGrr", 0.0, 20, True), ("rrGG", 23.0, 30, True),
            ("GGrr", 56.0, 10, True), ("rrGG", 69.0, 31, False),
        ]
    }  # fmt: skip
    timing = ep.signal_timing(intervals, greens, end=100.0)
    assert (timing["green_min_s"], timing["green_max_s"]) == (10.0, 30.0)
    # GGrr red from 20 to 56 (36 s) and from 66 to the end (34 s); rrGG red 0-23 and 53-69.
    assert timing["longest_red_per_signal"]["t"] == 36.0
    assert timing["phase_changes_per_signal"] == 3.0


def test_signal_timing_phase_never_green():
    timing = ep.signal_timing({"t": [("GGrr", 0.0, 100, False)]}, {"t": ["GGrr", "rrGG"]}, 100.0)
    assert timing["longest_red_max_s"] == 100.0


# ---------------------------------------------------------------- statistics
def test_summary_confidence_interval():
    s = stats.summarize([10.0, 12.0, 14.0])
    assert s.mean == 12.0 and s.std == pytest.approx(2.0)
    # t(0.975, 2) = 4.303; half width = 4.303 * 2 / sqrt(3)
    assert s.ci_half_width == pytest.approx(4.303 * 2 / np.sqrt(3), rel=1e-3)
    assert np.isnan(stats.summarize([1.0]).std)


def test_acceptance_rule():
    base_wait = [30.0, 31.0, 29.5, 30.5, 30.2]
    better = [20.0, 21.0, 19.5, 20.4, 20.8]
    zeros = [0, 0, 0, 0, 0]
    verdict = stats.compare(better, base_wait, zeros, zeros, zeros, zeros)
    assert verdict.beats and verdict.waiting_lower and verdict.p_waiting < 0.05
    # Same waiting but more teleports in every seed: rejected.
    worse_tele = stats.compare(better, base_wait, [5, 6, 5, 7, 6], zeros, zeros, zeros)
    assert not worse_tele.beats and worse_tele.teleports_increased
    # A not-significant difference in waiting is not enough.
    noisy = stats.compare([29.0, 31.5, 28.0, 32.0, 29.9], base_wait, zeros, zeros, zeros, zeros)
    assert not noisy.beats and not noisy.waiting_lower


def test_increase_ignores_seed_noise():
    assert not stats.increased([1, 0, 2, 0, 1], [0, 1, 1, 0, 1])  # 0.8 vs 0.6: noise
    assert stats.increased([1, 1, 1], [0, 0, 0])  # zero variance: any increase is real
    assert not stats.increased([0, 0, 0], [1, 2, 3])


# ---------------------------------------------------------------- evaluation
def _evaluate(method: str) -> dict:
    from trafficmarl.envs.factory import experiment_config
    from trafficmarl.metrics.evaluate import controller_spec, evaluate_episode

    config = experiment_config("single", {"env": {"profile": "low"}})
    return evaluate_episode(config, controller_spec(method), seed=0, route_index=0)


@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")
def test_same_metrics_for_program_and_policy_controllers():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2, maxtasksperchild=1) as pool:
        fixed, random = pool.map(_evaluate, ["fixed_time", "random"])
    assert fixed.keys() == random.keys()
    for row in (fixed, random):
        assert row["vehicles_due"] == row["arrived"] + row["unfinished_total"]
        assert row["mean_total_delay"] >= row["mean_time_loss"] - 1e-6
        assert row["green_min_s"] >= 5 and row["green_max_s"] <= 50
    assert fixed["mean_waiting_time"] < random["mean_waiting_time"]
