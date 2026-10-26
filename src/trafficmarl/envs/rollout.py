"""Run one episode with a non-learning policy and collect diagnostics.

Used by ``python -m trafficmarl env rollout`` (reward-scale checks) and by the
tests that prove the phase-duration constraints from SUMO's own output.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from trafficmarl.envs.factory import make_env
from trafficmarl.envs.policies import make_policy


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": arr.mean(), "std": arr.std(), "min": arr.min(), "max": arr.max()}


def tls_states_additional(tls_ids: list[str], directory: Path) -> tuple[Path, Path]:
    """Additional file making SUMO write every signal's state each second (ground truth)."""
    output = directory / "tls_states.xml"
    events = "".join(
        f'<timedEvent type="SaveTLSStates" source="{tls}" dest="{output}"/>' for tls in tls_ids
    )
    additional = directory / "tls_states.add.xml"
    additional.write_text(f"<additional>{events}</additional>", encoding="utf-8")
    return additional, output


def run_rollout(
    config: dict[str, Any],
    *,
    policy: str,
    split: str = "eval",
    seed: int = 0,
    max_seconds: float | None = None,
    record_tls_states: Path | None = None,
) -> dict[str, Any]:
    """Run one episode and return its summary, reward-component statistics and invariants.

    With ``record_tls_states`` (a directory), SUMO writes the state of every
    signal at every second to ``tls_states.xml`` there.
    """
    from trafficmarl.scenarios.spec import Manifest, output_dir

    extra: list[str] = []
    states_file = None
    if record_tls_states is not None:
        manifest = Manifest.load(output_dir(config["scenario"]))
        additional, states_file = tls_states_additional(manifest.tls_ids, Path(record_tls_states))
        extra = [str(additional)]
    env = make_env(config, split=split, seed=seed, extra_additional_files=extra or None)
    act = make_policy(policy, seed)
    started = time.monotonic()
    obs, _ = env.reset(seed=seed)
    obs_min, obs_max, padded_mask_ok = 1.0, 0.0, True
    n_green = env.spec.n_green
    steps, summary = 0, None
    while env.agents:
        for agent, o in obs.items():
            obs_min = min(obs_min, float(o["observations"].min()))
            obs_max = max(obs_max, float(o["observations"].max()))
            padded_mask_ok &= not o["action_mask"][n_green[agent] :].any()
        obs, _, _, _, infos = env.step(act(obs, env.green_phases()))
        steps += 1
        if infos and "episode" in next(iter(infos.values())):
            summary = next(iter(infos.values()))["episode"]
        elif max_seconds is not None and env._now() >= max_seconds:
            summary = env.episode_summary("stopped")
            break
    log = env.log
    result = {
        "scenario": config["scenario"]["name"],
        "profile": config["env"]["profile"],
        "policy": policy,
        "steps": steps,
        "wall_time_s": round(time.monotonic() - started, 1),
        "episode": summary,
        "obs_dim": env.spec.obs_dim,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "padded_mask_ok": bool(padded_mask_ok),
        "green_states": {
            a: [p.state for p in ts.green_phases] for a, ts in env.sumo_env.traffic_signals.items()
        },
        "reward_components": {
            a: {
                "traffic": _stats(log.traffic[a]),
                "switch_penalty": _stats(log.penalty[a]),
                "reward": _stats(log.reward[a]),
            }
            for a in log.traffic
        },
        "reward_components_all": {
            "traffic": _stats([v for a in log.traffic for v in log.traffic[a]]),
            "switch_penalty": _stats([v for a in log.penalty for v in log.penalty[a]]),
            "reward": _stats([v for a in log.reward for v in log.reward[a]]),
        },
    }
    env.close()
    if states_file is not None:
        result["tls_states_file"] = str(states_file)
    return result


def green_intervals(
    states_file: str | Path, green_states: dict[str, list[str]]
) -> dict[str, list[tuple[str, float, int, bool]]]:
    """Green intervals per signal from SUMO's per-second ``SaveTLSStates`` output.

    Returns ``(state, start time, duration in s, completed)`` for every maximal
    run of identical green states; the run still active at the end of the file
    is marked as not completed.
    """
    import xml.etree.ElementTree as ET

    runs: dict[str, list[list]] = {}
    for _, elem in ET.iterparse(states_file):
        if elem.tag != "tlsState":
            continue
        tls, state, t = elem.get("id"), elem.get("state"), float(elem.get("time"))
        tls_runs = runs.setdefault(tls, [])
        if tls_runs and tls_runs[-1][0] == state and tls_runs[-1][1] + tls_runs[-1][2] == t:
            tls_runs[-1][2] += 1
        else:
            tls_runs.append([state, t, 1])
        elem.clear()
    result = {}
    for tls, tls_runs in runs.items():
        greens = set(green_states[tls])
        result[tls] = [
            (state, start, length, i < len(tls_runs) - 1)
            for i, (state, start, length) in enumerate(tls_runs)
            if state in greens
        ]
    return result


def reward_scale_job(scenario: str, measure: str, profile: str = "medium") -> dict[str, Any]:
    """Std of the per-step traffic term under a random policy (calibrates the gridlock cap, D24).

    Runs in its own process (libsumo: one simulation per process).
    """
    from trafficmarl.envs.factory import experiment_config

    overrides = {
        "env": {
            "profile": profile,
            "reward": {"measure": measure, "gridlock_penalty_cap_std": None},
        }
    }
    result = run_rollout(experiment_config(scenario, overrides), policy="random", split="eval")
    traffic = result["reward_components_all"]["traffic"]
    return {
        "scenario": scenario,
        "measure": measure,
        "profile": profile,
        "std": float(traffic["std"]),
        "min": float(traffic["min"]),
        "max": float(traffic["max"]),
        "end_reason": result["episode"]["end_reason"],
        "steps": result["steps"],
    }
