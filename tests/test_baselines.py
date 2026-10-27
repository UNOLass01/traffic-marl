"""Baselines: Webster plans, max-pressure and the centralized wrapper."""

import multiprocessing
import shutil

import pytest
import sumolib
import yaml

from trafficmarl.baselines import webster
from trafficmarl.envs.factory import experiment_config
from trafficmarl.scenarios.spec import Manifest, output_dir

CFG = {
    "saturation_flow": 1800, "max_cycle": 150, "lost_time_per_phase": 3, "peak_window": 900,
    "min_green": 5, "max_green": 50,
}  # fmt: skip
requires_sumo = pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")


@pytest.fixture(scope="module")
def single():
    manifest = Manifest.load(output_dir(experiment_config("single")["scenario"]))
    net_file = manifest.path(manifest.net_file)
    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    return net.getTLS("A0"), webster.program_phases(net_file)["A0"]


def _flows(tl, through, left):
    """Same flow on every through/right lane and on every left-turn lane."""
    flows = {}
    for in_lane, _, _ in tl.getConnections():
        lane = in_lane.getID()
        flows[lane] = left if lane.endswith("_2") else through
    return flows


def test_webster_cycle_formula(single):
    tl, phases = single
    plan = webster.plan_signal(tl, phases, _flows(tl, 360.0, 90.0), CFG)
    # y = 360/1800 = 0.2 on both through phases, 90/1800 = 0.05 on both left phases.
    assert plan.critical_ratios == pytest.approx([0.2, 0.05, 0.2, 0.05])
    assert plan.flow_ratio_sum == pytest.approx(0.5)
    c0 = (1.5 * 4 * 3 + 5) / (1 - 0.5)  # 46 s
    non_green = sum(d for s, d in phases if "y" in s)
    assert not plan.capped
    assert sum(plan.greens) + non_green == pytest.approx(plan.cycle)
    assert abs(plan.cycle - c0) <= 4  # rounding and min-green clipping
    assert plan.greens[0] > plan.greens[1]  # proportional to the critical ratios


def test_webster_caps_cycle_when_oversaturated(single):
    tl, phases = single
    plan = webster.plan_signal(tl, phases, _flows(tl, 900.0, 300.0), CFG)  # Y = 1.33
    assert plan.flow_ratio_sum >= 1 and plan.capped
    assert plan.cycle <= CFG["max_cycle"]
    assert all(CFG["min_green"] <= g <= CFG["max_green"] for g in plan.greens)


def test_webster_peak_uses_busiest_window():
    counts = {0: {("a", "b"): 10.0}, 1: {("a", "b"): 40.0}, 2: {("a", "b"): 10.0}}
    lanes = {("a", "b"): {"a_0"}}
    mean = webster.hourly_lane_flows(
        counts, lanes, duration=2700, bin_seconds=900, busiest_window=False
    )
    peak = webster.hourly_lane_flows(
        counts, lanes, duration=2700, bin_seconds=900, busiest_window=True
    )
    assert mean["a_0"] == pytest.approx(60 / 0.75)  # 60 vehicles in 45 min
    assert peak["a_0"] == pytest.approx(40 * 4)  # busiest 15 min scaled to one hour


def test_webster_plans_exist_for_every_profile():
    for name in ("single", "corridor", "grid3x3", "grid5x5", "lust"):
        manifest = Manifest.load(output_dir(experiment_config(name)["scenario"]))
        plans = yaml.safe_load((manifest.root / "webster" / "plans.yaml").read_text())["plans"]
        assert set(plans) == set(manifest.routes)
        for per_tls in plans.values():
            for plan in per_tls.values():
                assert all(5 <= g <= 50 for g in plan["greens_s"])


def _episode(method: str) -> dict:
    from trafficmarl.metrics.evaluate import controller_spec, evaluate_episode

    return evaluate_episode(experiment_config("single"), controller_spec(method), 0, 0)


@requires_sumo
def test_baselines_control_the_signals():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(3, maxtasksperchild=1) as pool:
        web, act, mp = pool.map(_episode, ["webster", "actuated", "max_pressure"])
    manifest = Manifest.load(output_dir(experiment_config("single")["scenario"]))
    plan = yaml.safe_load((manifest.root / "webster" / "plans.yaml").read_text())["plans"][
        "medium"
    ]["A0"]
    # The Webster program is the one SUMO runs: its longest green is the plan's.
    assert web["green_max_s"] == max(plan["greens_s"])
    for row in (web, act, mp):
        assert row["green_min_s"] >= 5 and row["green_max_s"] <= 50
        assert row["unfinished_total"] == 0
    # Max-pressure acts through the agents' mask (decision counters exist).
    assert mp["min_green_active_rate"] == mp["min_green_active_rate"]  # not NaN


def _centralized_steps() -> dict:
    import numpy as np

    from trafficmarl.baselines.centralized import CentralizedEnv

    env = CentralizedEnv(experiment_config("corridor"), split="train", seed=0)
    obs, _ = env.reset(seed=0)
    shapes = (obs.shape, env.action_space.nvec.tolist(), env.action_masks().shape)
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = env.action_masks().reshape(len(env.agents), -1)
        action = [int(rng.choice(np.flatnonzero(m))) for m in mask]
        obs, reward, term, trunc, _ = env.step(action)
    env.close()
    return {"shapes": shapes, "reward": reward}


@requires_sumo
def test_centralized_wrapper_shapes_and_masks():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        out = pool.apply(_centralized_steps)
    obs_shape, nvec, mask_shape = out["shapes"]
    assert obs_shape == (4 * 47,) and nvec == [4, 4, 4, 4] and mask_shape == (16,)
    assert isinstance(out["reward"], float)


# ------------------------------------------------- Webster sanity checks (D31)
SCENARIOS = ("single", "corridor", "grid3x3", "grid5x5", "lust")


def _manifest(name: str) -> Manifest:
    return Manifest.load(output_dir(experiment_config(name)["scenario"]))


def test_webster_programs_keep_phase_structure():
    """Webster only changes green durations; every movement with demand gets a green."""
    import xml.etree.ElementTree as ET

    from trafficmarl.scenarios.tls import is_green_phase

    for name in SCENARIOS:
        manifest = _manifest(name)
        net_file = manifest.path(manifest.net_file)
        original = webster.program_phases(net_file)
        net = sumolib.net.readNet(str(net_file), withPrograms=True)
        moves = webster.junction_lanes(net)
        plans = yaml.safe_load((manifest.root / "webster" / "plans.yaml").read_text())["plans"]
        for profile in ("medium", "high", "peak"):
            add = ET.parse(manifest.root / "webster" / f"{profile}.add.xml").getroot()
            counts = webster.movement_counts(manifest.route_files(profile, "train"), moves, 3600)
            for logic in add.iter("tlLogic"):
                tls = logic.get("id")
                phases = [(p.get("state"), float(p.get("duration"))) for p in logic.iter("phase")]
                assert [s for s, _ in phases] == [s for s, _ in original[tls]]
                for (state, d), (_, d0) in zip(phases, original[tls], strict=True):
                    if not is_green_phase(state):
                        assert d == d0  # yellow kept
                greens = [d for s, d in phases if is_green_phase(s)]
                assert greens == plans[profile][tls]["greens_s"]
                # Every link carrying training demand is green in some green phase.
                served = {i for s, _ in phases if is_green_phase(s)
                          for i, c in enumerate(s) if c in "Gg"}  # fmt: skip
                tl = net.getTLS(tls)
                for in_lane, out_lane, idx in tl.getConnections():
                    move = (in_lane.getEdge().getID(), out_lane.getEdge().getID())
                    if any(per_bin.get(move) for per_bin in counts.get(tls, {}).values()):
                        assert idx in served, (name, tls, idx)


def _active_programs(name: str) -> dict:
    import traci

    manifest = _manifest(name)
    add = webster.webster_file(manifest, "medium")
    traci.start([sumolib.checkBinary("sumo"), "-n", str(manifest.path(manifest.net_file)),
                 "--additional-files", str(add), "--no-step-log"])  # fmt: skip
    traci.simulationStep()
    result = {}
    for tls in manifest.tls_ids:
        program = traci.trafficlight.getProgram(tls)
        logic = next(
            lg for lg in traci.trafficlight.getAllProgramLogics(tls) if lg.programID == program
        )
        result[tls] = (program, [p.duration for p in logic.phases])
    traci.close()
    return result


@requires_sumo
def test_sumo_runs_the_webster_program_on_lust():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        active = pool.apply(_active_programs, ("lust",))
    manifest = _manifest("lust")
    plans = yaml.safe_load((manifest.root / "webster" / "plans.yaml").read_text())["plans"]
    for tls, (program, durations) in active.items():
        assert program == webster.PROGRAM_ID
        assert set(plans["medium"][tls]["greens_s"]) <= set(durations)


def test_saturation_measured_for_every_scenario():
    from trafficmarl.baselines.saturation import load_measured

    for name in SCENARIOS:
        summary = load_measured(_manifest(name))
        assert set(summary["by_type"]) == {"through", "left", "right"}
        for value in summary["by_type"].values():
            assert 1000 < value["saturation_flow"] < 2600
        assert 2.0 < summary["lost_time_per_phase_s"] < 8.0


def test_saturation_summary_math():
    from trafficmarl.baselines.saturation import summarize_tests

    cfg = {"startup_s": 10, "green_s": 40}
    rows = [
        # 2 lanes, 1 veh/s in total -> 1800 veh/h/lane; 8 crossings in the first 10 s
        # -> start-up lost time 10 - 8/1 = 2 s; 1 crossing in yellow -> end gain 1 s.
        {"type": "through", "lanes": 2, "rate": 1.0, "startup_crossings": 8,
         "yellow_crossings": 1, "saturated_to_end": True},
        {"type": "left", "lanes": 1, "rate": 0.5, "startup_crossings": 4,
         "yellow_crossings": 0, "saturated_to_end": True},
        {"type": "right", "lanes": 1, "rate": 0.4, "startup_crossings": 0,
         "yellow_crossings": 0, "saturated_to_end": False},  # queue ran out: ignored
    ]  # fmt: skip
    s = summarize_tests(rows, cfg, yellow=3)
    assert s["by_type"]["through"]["saturation_flow"] == 1800
    assert s["by_type"]["left"]["saturation_flow"] == 1800
    assert "right" not in s["by_type"] and s["valid_tests"] == 2
    assert s["startup_lost_s"] == pytest.approx((2 + 2) / 2)
    assert s["end_gain_s"] == pytest.approx(0.5)
    assert s["lost_time_per_phase_s"] == pytest.approx(2 + 3 - 0.5)


def test_webster_turning_flows_in_through_equivalents():
    counts = {0: {("a", "b"): 90.0, ("a", "c"): 90.0}}
    lanes = {("a", "b"): {"a_0"}, ("a", "c"): {"a_0"}}
    flows = webster.hourly_lane_flows(
        counts, lanes, duration=3600, bin_seconds=3600, busiest_window=False,
        weights={("a", "c"): 1800 / 1200},
    )  # fmt: skip
    # 90 through + 90 turning at s = 1200 -> 90 + 135 through-equivalents.
    assert flows["a_0"] == pytest.approx(225.0)
