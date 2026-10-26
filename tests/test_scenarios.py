import filecmp
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import sumolib

from trafficmarl.scenarios import demand, synthetic
from trafficmarl.scenarios.lust import scaled_window
from trafficmarl.scenarios.spec import Manifest, available_scenarios, load_scenario_config
from trafficmarl.scenarios.tls import ACTUATED_PROGRAM_ID, split_tls_programs, to_actuated
from trafficmarl.scenarios.validate import (
    Job,
    drain_without_teleport,
    parse_statistics,
    profile_rules,
    run_job,
)
from trafficmarl.utils import paths

PROFILES = {"low", "medium", "high", "peak", "saturated"}
requires_sumo = pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")

STATS_XML = """<statistics>
    <performance begin="0.00" end="900.00"/>
    <vehicles loaded="120" inserted="118" running="3" waiting="2"/>
    <teleports total="4" jam="4" yield="0" wrongLane="0"/>
    <vehicleTripStatistics count="115" routeLength="400" speed="8" duration="61.5"
        waitingTime="12.25" timeLoss="20.5" departDelay="1"/>
</statistics>"""


@pytest.mark.parametrize("name", available_scenarios())
def test_scenario_configs_complete(name):
    cfg = load_scenario_config(name)
    assert cfg["name"] == name
    assert set(cfg["demand"]["profiles"]) >= PROFILES
    extra = set(cfg["demand"]["profiles"]) - PROFILES
    assert extra <= set(cfg["demand"].get("robustness_only", [])), (
        "extra levels are robustness-only"
    )
    assert cfg["validation"]["max_teleport_ratio"] <= 0.01
    if cfg["kind"] == "synthetic":
        train, test = cfg["demand"]["splits"]["train"], cfg["demand"]["splits"]["eval"]
    else:
        train, test = cfg["demand"]["windows"]["train"], cfg["demand"]["windows"]["eval"]
    assert train and test and not set(train) & set(test), "train/eval must not share traffic"


def test_expected_scenarios_present():
    assert {"single", "corridor", "grid3x3", "grid5x5", "lust"} <= set(available_scenarios())


def test_parse_statistics():
    stats = parse_statistics(STATS_XML, "r.rou.xml")
    assert (stats.loaded, stats.inserted, stats.arrived) == (120, 118, 115)
    assert (stats.running_at_end, stats.waiting_to_insert, stats.teleports) == (3, 2, 4)
    assert stats.mean_waiting_time == 12.25
    assert stats.teleport_ratio == pytest.approx(4 / 118)


def _vehicle(vid: str, depart: float) -> ET.Element:
    return ET.Element("vehicle", id=vid, depart=f"{depart:.2f}")


def test_scaled_window_shifts_filters_and_scales():
    vehicles = [_vehicle(f"v{i}", 1000 + i) for i in range(4000)]  # 1 veh/s from t=1000
    rng = np.random.default_rng(0)
    same = scaled_window(vehicles, 2000, 1000, [1.0], rng)
    assert len(same) == 1000
    assert float(same[0].get("depart")) == 0.0
    assert max(float(v.get("depart")) for v in same) < 1000

    thinned = scaled_window(vehicles, 2000, 1000, [0.5], rng)
    assert len(thinned) == pytest.approx(500, abs=60)

    doubled = scaled_window(vehicles, 2000, 1000, [2.0], rng)
    ids = [v.get("id") for v in doubled]
    assert len(doubled) == 2000 and len(set(ids)) == 2000
    departs = [float(v.get("depart")) for v in doubled]
    assert departs == sorted(departs)
    assert vehicles[1000].get("depart") == "2000.00", "input vehicles must not be modified"


def test_split_tls_programs(tmp_path):
    net = tmp_path / "net.xml"
    net.write_text(
        "<net><tlLogic id='a' type='actuated' programID='0'><phase duration='30' state='Gr'/>"
        "</tlLogic><tlLogic id='a' type='static' programID='1'/><edge id='e'/></net>"
    )
    moved = split_tls_programs(net, tmp_path / "act.add.xml")
    assert moved == 1
    kept = ET.parse(net).getroot().findall("tlLogic")
    assert [tl.get("type") for tl in kept] == ["static"]
    assert ET.parse(net).getroot().find("edge") is not None
    actuated = ET.parse(tmp_path / "act.add.xml").getroot().find("tlLogic")
    assert (actuated.get("type"), actuated.get("programID")) == ("actuated", ACTUATED_PROGRAM_ID)


def test_to_actuated_adds_duration_range_to_green_phases_only():
    tl = ET.fromstring(
        "<tlLogic id='a' type='static' programID='0'>"
        "<phase duration='30' state='GGrr'/><phase duration='3' state='yyrr'/>"
        "<phase duration='30' state='rrGG' minDur='10' maxDur='40'/></tlLogic>"
    )
    (act,) = to_actuated([tl])
    green, yellow, preset = act.findall("phase")
    assert (green.get("minDur"), green.get("maxDur")) == ("5", "50")
    assert yellow.get("minDur") is None
    assert (preset.get("minDur"), preset.get("maxDur")) == ("10", "40")
    assert tl.get("type") == "static", "input must not be modified"


@requires_sumo
def test_synthetic_build_routes_are_valid(tiny_single_cfg):
    manifest = synthetic.build(tiny_single_cfg)
    assert manifest.tls_ids == ["A0"]
    net = sumolib.net.readNet(str(manifest.path(manifest.net_file)), withPrograms=True)
    greens = [p for p in net.getTLS("A0").getPrograms()["0"].getPhases() if "y" not in p.state]
    assert len(greens) == 4, "NS/EW through + protected left phases"
    entries, exits = demand.entry_and_exit_edges(net)
    with paths.open_input(manifest.route_files("medium", "train")[0]) as fh:
        root = ET.parse(fh).getroot()
    for veh in root.iter("vehicle"):
        edges = veh.find("route").get("edges").split()
        assert edges[0] in entries and edges[-1] in exits
        for a, b in zip(edges, edges[1:], strict=False):
            assert net.getEdge(b) in net.getEdge(a).getOutgoing()
    assert Manifest.load(manifest.root).vehicles == manifest.vehicles
    assert manifest.sumo_version and manifest.path(manifest.actuated_file).is_file()


@requires_sumo
def test_synthetic_build_is_deterministic(tiny_single_cfg, tmp_path):
    first = synthetic.build(tiny_single_cfg)
    tiny_single_cfg["output_dir"] = str(tmp_path / "again")
    second = synthetic.build(tiny_single_cfg)
    for rel in first.vehicles:
        assert filecmp.cmp(first.path(rel), second.path(rel), shallow=False), rel


def test_profile_rules_layering():
    cfg = {
        "end": 5000,
        "profiles": {"saturated": {"controller": "actuated", "gridlock_check": True}},
    }
    assert profile_rules(cfg, "medium")["controller"] == "static"
    assert profile_rules(cfg, "medium")["end"] == 5000
    sat = profile_rules(cfg, "saturated")
    assert sat["controller"] == "actuated" and sat["gridlock_check"] and sat["end"] == 5000


def _job(manifest, profile, rules):
    rel = manifest.route_files(profile, "eval")[0].relative_to(manifest.root)
    return Job(manifest.name, str(manifest.root), profile, "eval", str(rel), json.dumps(rules), "k")


@requires_sumo
def test_tiny_scenario_validates_in_sumo(tiny_single_cfg):
    manifest = synthetic.build(tiny_single_cfg)
    rules = profile_rules(tiny_single_cfg["validation"], "medium")
    row = run_job(_job(manifest, "medium", rules))
    assert row["passed"] and row["arrived"] == row["inserted"] > 0
    rules.update(controller="actuated", gridlock_check=True)
    row = run_job(_job(manifest, "medium", rules))
    assert row["passed"] and row["controller"] == "actuated" and row["drained_without_teleport"]


@requires_sumo
def test_gridlock_is_detected(tiny_single_cfg):
    """A tiny window with no arrivals right after the start must be flagged as gridlock."""
    manifest = synthetic.build(tiny_single_cfg)
    rel = str(manifest.route_files("medium", "eval")[0].relative_to(manifest.root))
    result = drain_without_teleport(manifest, rel, end=900, controller="static", window=5)
    assert not result.drained and result.gridlock_at is not None
