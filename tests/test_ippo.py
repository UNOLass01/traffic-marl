"""IPPO: RLlib environment wrapper, plateau rule, policy loading."""

import multiprocessing
import shutil

import pandas as pd
import pytest

from trafficmarl.agents.ippo import plateau

requires_sumo = pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")


def _progress(values):
    return pd.DataFrame({"episode_return_mean": values, "env_steps": range(len(values))})


def test_plateau_rule():
    rising = [float(i) for i in range(50)]
    assert not plateau(_progress(rising))["plateaued"]
    flat_end = [float(min(i, 25)) for i in range(50)]  # flat over the last two fifths
    assert plateau(_progress(flat_end))["plateaued"]
    assert not plateau(_progress([1.0, 2.0]))["plateaued"]  # too short to tell


def _episode(tmp: str) -> dict:
    import numpy as np

    from trafficmarl.agents.ippo import RLlibTrafficEnv
    from trafficmarl.envs.factory import experiment_config

    config = experiment_config("corridor", {"env": {"profile": "low"}})
    env = RLlibTrafficEnv({"config": config, "seed": 0, "run_dir": tmp})
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    steps, done = 0, False
    while not done:
        actions = {a: int(rng.choice(np.flatnonzero(o["action_mask"]))) for a, o in obs.items()}
        obs, rewards, terms, truncs, _ = env.step(actions)
        steps += 1
        done = terms["__all__"] or truncs["__all__"]
    env.close()
    return {"steps": steps, "agents": sorted(rewards), "terms": terms, "truncs": truncs}


@requires_sumo
def test_rllib_env_runs_an_episode_and_logs_it(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        out = pool.apply(_episode, (str(tmp_path),))
    assert len(out["agents"]) == 4
    assert out["terms"]["__all__"] != out["truncs"]["__all__"]  # exactly one end type
    logged = (tmp_path / "episodes").glob("*.jsonl")
    record = pd.read_json(next(logged), lines=True).iloc[0]
    assert record.env_steps == out["steps"]
    assert record.end_reason in ("network_empty", "time_limit", "gridlock")


def test_rollback_logs_drops_what_the_checkpoint_lost(tmp_path):
    import json
    import os

    from trafficmarl.agents.ippo import rollback_logs

    pd.DataFrame(
        {"iteration": [1, 2, 3, 4], "env_steps": [10, 20, 30, 40], "wall_time_s": [1, 2, 3, 4]}
    ).to_csv(tmp_path / "progress.csv", index=False)
    state = tmp_path / "checkpoint" / "algorithm_state.pkl"
    state.parent.mkdir()
    state.write_bytes(b"")
    os.utime(state, (1000.0, 1000.0))
    (tmp_path / "episodes").mkdir()
    log = tmp_path / "episodes" / "runner_1_0.jsonl"
    log.write_text("".join(json.dumps({"wall_time": t}) + "\n" for t in (900.0, 1000.0, 1100.0)))

    rows = rollback_logs(tmp_path, iteration=2)

    assert [r["env_steps"] for r in rows] == [10, 20]
    assert list(pd.read_csv(tmp_path / "progress.csv").iteration) == [1, 2]
    assert [json.loads(x)["wall_time"] for x in log.read_text().splitlines()] == [900.0, 1000.0]
    assert rollback_logs(tmp_path, iteration=2) == rows  # idempotent
