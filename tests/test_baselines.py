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
