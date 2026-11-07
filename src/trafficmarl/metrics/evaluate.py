"""Evaluation of any controller under identical conditions (docs/decisions.md D14, D26).

A controller is described by a picklable spec:

* ``control: program``: SUMO runs a signal program (the network's fixed-time
  program, or one loaded from ``additional`` files: actuated, Webster);
* ``control: agents``: a policy chooses phases through the action mask.

Every (seed, evaluation route file) pair is one episode, simulated in its own
process (libsumo hosts one simulation per process) with the same SUMO seed for
every method (common random numbers), the same episode rules and the same
SUMO output files, from which :mod:`trafficmarl.metrics.episode` computes the
metrics.
"""

from __future__ import annotations

import multiprocessing
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficmarl.coordination.communication import centralized_floats, floats_per_step
from trafficmarl.metrics import episode as ep
from trafficmarl.utils import paths
from trafficmarl.utils.config import deep_merge
from trafficmarl.utils.logging_utils import get_logger
from trafficmarl.utils.run_info import write_run_info

logger = get_logger(__name__)

ACTUATED = "@actuated"  # placeholder for the scenario's actuated program file
WEBSTER = "@webster"  # placeholder for the Webster plan of the evaluated demand profile

# Built-in controllers; later phases register more (Webster, max-pressure, RL policies).
CONTROLLERS: dict[str, dict[str, Any]] = {
    "fixed_time": {"control": "program", "additional": []},
    "actuated": {"control": "program", "additional": [ACTUATED]},
    "webster": {"control": "program", "additional": [WEBSTER]},
    "max_pressure": {"control": "agents", "policy": "max_pressure"},
    # Max-pressure with the optional max-red constraint (D32); ``env`` overrides the config.
    "max_pressure_maxred": {
        "control": "agents",
        "policy": "max_pressure",
        "env": {"max_red": 120},
    },  # fmt: skip
    # Needs ``checkpoint=<run_dir>/model.zip`` (trained with experiments/run_baselines.py).
    "centralized_ppo": {"control": "agents", "policy": "centralized_ppo"},
    # Needs ``checkpoint=<run_dir>/checkpoint`` (trained with experiments/train.py).
    "ippo": {"control": "agents", "policy": "ippo"},
    "mappo": {"control": "agents", "policy": "ippo"},  # same masked actor checkpoint layout
    # Needs ``checkpoint=<run_dir>/checkpoint`` (agents/offpolicy.py).
    "qmix": {"control": "agents", "policy": "qmix"},
    "maddpg": {"control": "agents", "policy": "maddpg"},
    # One module per traffic light, independent or federated (federated/trainer.py).
    "federated": {"control": "agents", "policy": "local_policies"},
    "random": {"control": "agents", "policy": "random"},
    "keep": {"control": "agents", "policy": "keep"},
    "switch": {"control": "agents", "policy": "switch"},
}

# Hook for policies that are not simple heuristics: name -> factory(spec, env) -> act(obs, env).
POLICY_FACTORIES: dict[str, Callable] = {}


def _resolve_additional(spec: dict, manifest, profile: str) -> list[str]:
    files = []
    for item in spec.get("additional", []):
        if item == ACTUATED:
            files.append(str(manifest.path(manifest.actuated_file)))
        elif item == WEBSTER:
            from trafficmarl.baselines.webster import webster_file

            files.append(str(webster_file(manifest, profile)))
        else:
            files.append(str(paths.resolve(item)))
    return files


def _make_policy(spec: dict, env, seed: int):
    name = spec["policy"]
    if name == "max_pressure":
        from trafficmarl.baselines.max_pressure import make_max_pressure

        return make_max_pressure(spec, env)
    if name == "centralized_ppo":
        from trafficmarl.baselines.centralized import make_centralized

        return make_centralized(spec, env)
    if name == "ippo":
        from trafficmarl.agents.ippo import make_ippo

        return make_ippo(spec, env)
    if name == "qmix":
        from trafficmarl.agents.qmix import make_qmix

        return make_qmix(spec, env)
    if name == "maddpg":
        from trafficmarl.agents.maddpg import make_maddpg

        return make_maddpg(spec, env)
    if name == "local_policies":
        from trafficmarl.federated.trainer import make_local_policies

        return make_local_policies(spec, env)
    if name in POLICY_FACTORIES:
        return POLICY_FACTORIES[name](spec, env)
    from trafficmarl.envs.policies import make_policy

    heuristic = make_policy(name, seed)
    return lambda obs, env: heuristic(obs, env.green_phases())


def evaluate_episode(
    config: dict, spec: dict, seed: int, route_index: int, sumo_seed: int | None = None
) -> dict[str, Any]:
    """Run one evaluation episode and return its metrics (one CSV row).

    ``seed`` names the run; the simulation uses ``sumo_seed`` (default: ``seed``).
    """
    from trafficmarl.envs.factory import make_env
    from trafficmarl.envs.rollout import green_intervals
    from trafficmarl.envs.traffic_env import green_phase_states
    from trafficmarl.scenarios.spec import Manifest, output_dir

    manifest = Manifest.load(output_dir(config["scenario"]))
    if spec.get("env"):  # controller-specific environment settings (e.g. max_red)
        config = deep_merge(config, {"env": spec["env"]})
    if "checkpoint" in spec:  # per-seed checkpoints: "<dir>/{seed}/..."
        spec = {**spec, "checkpoint": spec["checkpoint"].format(seed=seed)}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sumo_args, files = ep.output_args(tmp_path)
        tls_add, tls_states = ep.tls_states_additional(manifest.tls_ids, tmp_path)
        env = make_env(
            config,
            split="eval",
            seed=seed,
            extra_additional_files=[
                str(tls_add),
                *_resolve_additional(spec, manifest, config["env"]["profile"]),
            ],
            extra_sumo_args=sumo_args,
            control=spec["control"],
        )
        sumo_seed = seed if sumo_seed is None else sumo_seed
        obs, _ = env.reset(seed=seed, options={"route_index": route_index, "sumo_seed": sumo_seed})
        route_path = env.route_path
        policy = _make_policy(spec, env, seed) if spec["control"] == "agents" else None
        summary = None
        while env.agents:
            actions = policy(obs, env) if policy else None
            obs, _, _, _, infos = env.step(actions)
            if infos:
                summary = next(iter(infos.values())).get("episode", summary)
        communication = floats_per_step(
            n_agents=len(env.spec.topology),
            n_links=sum(len(t.neighbours) for t in env.spec.topology.values()),
            message_dim=env.spec.message_dim,
            extended=env.spec.extended,
            reward_type=config["env"]["reward"]["type"],
        )
        algorithm = config.get("algorithm", {}).get("name", spec.get("policy"))
        central = centralized_floats(
            algorithm,
            n_agents=len(env.spec.topology),
            obs_dim=env.spec.obs_dim,
            message_dim=env.spec.message_dim,
        )
        communication = {k: communication[k] + central[k] for k in communication}
        if spec["control"] != "agents":
            communication = {"execution": 0, "training": 0}
        env.close()  # flushes tripinfo (including unfinished vehicles) and summary

        end = float(summary["sim_time"])
        green_states = green_phase_states(str(manifest.path(manifest.net_file)))
        timing = ep.signal_timing(
            green_intervals(tls_states, green_states), green_states, end, manifest.begin
        )
        delays = ep.delay_metrics(
            ep.trip_records(files["tripinfo"]), ep.intended_departures(route_path), end
        )
        network_queue = ep.network_queue(files["summary"])

    threshold = float(config["metrics"]["teleport_flag_threshold"])
    counters = summary["constraints"]
    decisions = sum(c["decisions"] for c in counters.values())
    steps = decisions / len(counters) if decisions else 0
    return {
        "scenario": config["scenario"]["name"],
        "profile": config["env"]["profile"],
        "method": spec.get("name", spec.get("policy", "program")),
        "seed": seed,
        "sumo_seed": sumo_seed,
        "route_file": route_path.name,
        "end_reason": summary["end_reason"],
        "gridlock": summary["end_reason"] == "gridlock",
        "sim_time": end,
        **delays,
        "teleports": summary["teleports"],
        "teleport_rate": summary["teleport_rate"],
        "teleport_flag": summary["teleport_rate"] > threshold,
        "network_queue_mean": network_queue,
        "approach_queue_total": summary["approach_queue_total"],
        "approach_queue_per_signal": summary["approach_queue_per_signal"],
        "green_min_s": timing["green_min_s"],
        "green_max_s": timing["green_max_s"],
        "longest_red_max_s": timing["longest_red_max_s"],
        "longest_red_mean_s": timing["longest_red_mean_s"],
        "min_green_active_rate": (
            sum(c["min_green_active"] for c in counters.values()) / decisions
            if decisions
            else np.nan
        ),
        "max_green_forced": sum(c["max_green_forced"] for c in counters.values())
        if decisions
        else np.nan,
        "max_red_forced": sum(c["max_red_forced"] for c in counters.values()),
        "max_red_violations": sum(c["max_red_violations"] for c in counters.values()),
        "phase_changes_per_signal": timing["phase_changes_per_signal"],
        # Execution: messages the deployed controllers exchange (extended observation only).
        # Training: plus the shared rewards (coordination/communication.py).
        "communication_floats_per_step": communication["execution"],
        "communication_floats_total": communication["execution"] * steps,
        "training_communication_floats_per_step": communication["training"],
    }


def controller_spec(name: str, **overrides: Any) -> dict[str, Any]:
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller '{name}'; available: {sorted(CONTROLLERS)}")
    return {"name": name, **CONTROLLERS[name], **overrides}


def evaluate(
    config: dict,
    spec: dict,
    seeds: list[int],
    *,
    experiment: str,
    workers: int | None = None,
    sumo_seeds: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Evaluate ``spec`` on every evaluation route file for each seed.

    ``sumo_seeds`` gives a run several simulations (its seed -> SUMO seeds); by
    default a run's only SUMO seed is its own seed. Writes
    ``results/<experiment>/<method>/<seed>/metrics.csv`` (one row per episode) with the
    resolved config and git commit, and returns all rows.
    """
    from trafficmarl.scenarios.spec import Manifest, output_dir
    from trafficmarl.scenarios.validate import default_workers

    manifest = Manifest.load(output_dir(config["scenario"]))
    n_files = len(manifest.route_files(config["env"]["profile"], "eval"))
    sumo_seeds = sumo_seeds or {}
    jobs = [
        (config, spec, seed, i, sumo)
        for seed in seeds
        for i in range(n_files)
        for sumo in sumo_seeds.get(seed, [seed])
    ]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(len(jobs), workers or default_workers()), maxtasksperchild=1) as pool:
        rows = pool.starmap(evaluate_episode, jobs)
    table = pd.DataFrame(rows)
    method = spec.get("name", spec.get("policy"))
    for seed, group in table.groupby("seed"):
        run_dir = paths.run_dir(experiment, method, int(seed))
        group.to_csv(run_dir / "metrics.csv", index=False)
        write_run_info(run_dir, {**config, "evaluation": {"controller": spec, "seed": int(seed)}})
    logger.info("%s: %d episodes -> results/%s/%s/", method, len(rows), experiment, method)
    return table


def per_seed(table: pd.DataFrame, metric: str) -> pd.Series:
    """Per-seed value of ``metric`` (mean over that seed's evaluation episodes)."""
    return table.groupby("seed")[metric].mean()
