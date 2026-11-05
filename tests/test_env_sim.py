"""Environment tests that run SUMO.

libsumo hosts one simulation per process, so every simulation runs in its own
spawned worker process.

The constraint proofs do not trust the environment's own bookkeeping: SUMO
writes the state of every signal at every second (``SaveTLSStates``), and the
green durations are measured from that file.
"""

import multiprocessing
import shutil
import tempfile
from pathlib import Path

import pytest

from trafficmarl.envs.factory import experiment_config
from trafficmarl.scenarios.spec import available_scenarios

pytestmark = pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")

PROOF_SECONDS = 900
POLICIES = ("random", "keep", "switch")
SCENARIOS = ("single", "corridor", "grid3x3", "grid5x5", "lust")


def _in_process(func, *args):
    """Run ``func(*args)`` in a fresh process (one libsumo simulation per process)."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(func, args)


def proof_job(scenario: str, policy: str) -> dict:
    from trafficmarl.envs.rollout import green_intervals, run_rollout

    config = experiment_config(scenario, {"env": {"profile": "high"}})
    with tempfile.TemporaryDirectory() as tmp:
        result = run_rollout(
            config, policy=policy, split="eval", seed=1, max_seconds=PROOF_SECONDS,
            record_tls_states=Path(tmp),
        )  # fmt: skip
        result["intervals"] = green_intervals(result["tls_states_file"], result["green_states"])
    result["env_cfg"] = config["env"]
    return result


@pytest.fixture(scope="module")
def proofs():
    jobs = [(s, p) for s in SCENARIOS for p in POLICIES]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(len(jobs), 8), maxtasksperchild=1) as pool:
        results = pool.starmap(proof_job, jobs)
    return dict(zip(jobs, results, strict=True))


def test_all_scenarios_covered():
    assert set(SCENARIOS) == set(available_scenarios())


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("policy", POLICIES)
def test_green_durations_within_bounds(proofs, scenario, policy):
    """Ground truth from SUMO: no green shorter than min_green or longer than max_green."""
    result = proofs[(scenario, policy)]
    min_green, max_green = result["env_cfg"]["min_green"], result["env_cfg"]["max_green"]
    lengths = []
    for tls, intervals in result["intervals"].items():
        assert intervals, f"{tls}: no green interval recorded"
        for state, start, length, completed in intervals:
            assert length <= max_green, f"{tls} green {state} from t={start} lasted {length}s"
            if completed:
                assert length >= min_green, f"{tls} green {state} from t={start} lasted {length}s"
                lengths.append(length)
    assert lengths, "no completed green interval"
    if policy == "keep":  # always keeping must hit the maximum exactly
        assert max(lengths) == max_green
    if policy == "switch":  # always switching must reach the shortest green allowed
        assert min(lengths) < 2 * min_green


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_observations_and_masks_well_formed(proofs, scenario):
    for policy in POLICIES:
        result = proofs[(scenario, policy)]
        assert result["obs_min"] >= 0.0 and result["obs_max"] <= 1.0
        assert result["padded_mask_ok"]
        ep = result["episode"]
        counters = ep["constraints"]
        assert sum(c["decisions"] for c in counters.values()) == result["steps"] * len(counters)
        if policy == "keep":
            assert sum(c["max_green_forced"] for c in counters.values()) > 0
        if policy == "switch":
            assert sum(c["min_green_active"] for c in counters.values()) > 0


# ------------------------------------------------------------ maximum red
MAX_RED = 220
MAX_RED_SECONDS = 900


def max_red_job(scenario: str, max_red: int | None) -> dict:
    """Longest red of every green phase, measured by SUMO, under the starving ``lowest`` policy."""
    from trafficmarl.envs.rollout import green_intervals, run_rollout

    overrides = {"env": {"profile": "high", "max_red": max_red}}
    config = experiment_config(scenario, overrides)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_rollout(
            config, policy="lowest", split="eval", seed=1, max_seconds=MAX_RED_SECONDS,
            record_tls_states=Path(tmp),
        )  # fmt: skip
        intervals = green_intervals(result["tls_states_file"], result["green_states"])
    end = MAX_RED_SECONDS
    sumo_red = 0.0
    for tls, runs in intervals.items():
        for state in set(result["green_states"][tls]):
            last_end, longest = 0.0, 0.0
            for s, start, length, _ in runs:
                if s == state:
                    longest, last_end = max(longest, start - last_end), start + length
            sumo_red = max(sumo_red, longest, end - last_end)
    env_red = max(r for tls in result["episode"]["longest_red"].values() for r in tls.values())
    return {"sumo_red": sumo_red, "env_red": env_red, "env_cfg": config["env"],
            "yellow": config["scenario"]["signals"]["yellow_time"]}  # fmt: skip


@pytest.mark.parametrize("scenario", ["single", "lust"])
def test_max_red_bounds_starvation(scenario):
    """Ground truth from SUMO: the max-red mask ends the starvation the ``lowest`` policy causes.

    Phases that become overdue together are served one after the other, so a
    phase can wait one more minimum green and yellow beyond ``max_red`` (D40).
    """
    free = _in_process(max_red_job, scenario, None)
    bounded = _in_process(max_red_job, scenario, MAX_RED)
    assert free["sumo_red"] >= MAX_RED_SECONDS - 1  # some phase never served
    slack = bounded["env_cfg"]["min_green"] + bounded["yellow"]
    assert MAX_RED <= bounded["sumo_red"] <= MAX_RED + slack
    assert bounded["env_red"] == pytest.approx(bounded["sumo_red"], abs=1)  # bookkeeping agrees


# ------------------------------------------------------------ episode ends
def end_job(scenario: str, profile: str, overrides: dict, invalid_first_action: bool = False):
    from trafficmarl.envs.constraints import InvalidActionError
    from trafficmarl.envs.factory import make_env
    from trafficmarl.envs.policies import make_policy

    config = experiment_config(scenario, {"env": {"profile": profile}, **overrides})
    env = make_env(config, split="eval", seed=0)
    obs, _ = env.reset(seed=0)
    if invalid_first_action:
        agent = env.agents[0]
        try:
            env.step({a: 1 if a == agent else 0 for a in env.agents})
        except InvalidActionError:
            return {"raised": True}
        return {"raised": False}
    act = make_policy("random", 0)
    while env.agents:
        obs, rewards, terms, truncs, infos = env.step(act(obs, env.green_phases()))
    expected_penalty = env.gridlock_penalty()  # evaluated at the time the episode ended
    env.close()
    agent = next(iter(rewards))
    info = infos[agent]
    return {
        "terminated": terms[agent],
        "truncated": truncs[agent],
        "gridlock": info["gridlock"],
        "end_reason": info["episode"]["end_reason"],
        "reward": rewards[agent],
        "step_reward_without_terminal": info["traffic_reward"] - info["switch_penalty"],
        "expected_penalty": expected_penalty,
    }


def _default_reward_scale(scenario: str) -> float:
    """Measured per-step std of the traffic term for the default reward measure (D24)."""
    config = experiment_config(scenario)
    return config["scenario"]["reward_scale"][config["env"]["reward"]["measure"]]


def test_gridlock_terminates_with_penalty():
    # No vehicle can reach its destination in 5 s, so the 5 s window detects "gridlock".
    out = _in_process(end_job, "single", "medium", {"simulation": {"gridlock_window": 5}})
    assert out["end_reason"] == "gridlock" and out["gridlock"]
    assert out["terminated"] and not out["truncated"]
    assert out["expected_penalty"] > 0
    assert out["reward"] == pytest.approx(
        out["step_reward_without_terminal"] - out["expected_penalty"]
    )
    # The default cap (100 x the measured per-step std) is what limits the penalty here.
    assert out["expected_penalty"] == pytest.approx(100 * _default_reward_scale("single"))


def test_uncapped_penalty_exceeds_cap():
    overrides = {
        "simulation": {"gridlock_window": 5},
        "env": {"reward": {"gridlock_penalty_cap_std": None}},
    }
    out = _in_process(end_job, "single", "medium", overrides)
    assert out["expected_penalty"] > 100 * _default_reward_scale("single")


def test_time_limit_truncates_without_penalty():
    # No drain period: the episode stops at 3600 s while vehicles are still in the network.
    out = _in_process(end_job, "single", "medium", {"simulation": {"drain_time": 0}})
    assert out["end_reason"] == "time_limit" and not out["gridlock"]
    assert out["truncated"] and not out["terminated"]
    assert out["reward"] == pytest.approx(out["step_reward_without_terminal"])


def test_empty_network_terminates_without_penalty():
    out = _in_process(end_job, "single", "low", {})
    assert out["end_reason"] == "network_empty" and not out["gridlock"]
    assert out["terminated"] and not out["truncated"]
    assert out["reward"] == pytest.approx(out["step_reward_without_terminal"])


def test_masked_action_rejected_by_env():
    # At t=0 the minimum green allows only "keep" (phase 0); phase 1 must raise.
    assert _in_process(end_job, "single", "medium", {}, True) == {"raised": True}
