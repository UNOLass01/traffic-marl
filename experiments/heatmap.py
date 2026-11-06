"""Mean halting queue per intersection under each controller, for the congestion heatmaps.

One deterministic episode per controller on the first held-out route file. Writes
``results/heatmap_<scenario>_<profile>.csv`` with the position of each intersection.

Usage:
    python experiments/heatmap.py --scenario grid5x5 --profile high \
        --methods fixed_time actuated max_pressure ippo_local
"""

from __future__ import annotations

import argparse
import multiprocessing
from typing import Any

import pandas as pd

from trafficmarl.envs.factory import experiment_config
from trafficmarl.metrics.evaluate import _make_policy, _resolve_additional, controller_spec
from trafficmarl.utils import paths
from trafficmarl.utils.config import deep_merge, load_config

PROGRAMS = ("fixed_time", "webster", "actuated")
ALGORITHM = {"mappo": "mappo", "qmix": "qmix", "maddpg": "maddpg"}


def _spec_and_config(scenario: str, profile: str, method: str, seed: int) -> tuple[dict, dict]:
    config = experiment_config(scenario, {"env": {"profile": profile}})
    if method in PROGRAMS or method.startswith("max_pressure"):
        return controller_spec(method), config
    run = paths.RESULTS_DIR / f"marl_{scenario}" / method / str(seed)
    saved = load_config(run / "config.yaml")
    config = deep_merge(config, {"env": {**saved["env"], "profile": profile}})
    spec = controller_spec(ALGORITHM.get(method, "ippo"), checkpoint=str(run / "checkpoint"))
    return spec, config


def queues(scenario: str, profile: str, method: str, seed: int) -> list[dict[str, Any]]:
    """Run one episode; return one row per intersection."""
    import sumolib

    from trafficmarl.envs.factory import make_env
    from trafficmarl.scenarios.spec import Manifest, output_dir

    spec, config = _spec_and_config(scenario, profile, method, seed)
    if spec.get("env"):
        config = deep_merge(config, {"env": spec["env"]})
    manifest = Manifest.load(output_dir(config["scenario"]))
    env = make_env(
        config,
        split="eval",
        seed=seed,
        extra_additional_files=_resolve_additional(spec, manifest, profile),
        control=spec["control"],
    )
    obs, _ = env.reset(seed=seed, options={"route_index": 0, "sumo_seed": seed})
    policy = _make_policy(spec, env, seed) if spec["control"] == "agents" else None
    total, steps = dict.fromkeys(env.possible_agents, 0.0), 0
    while env.agents:
        obs, *_ = env.step(policy(obs, env) if policy else None)
        steps += 1
        if env.agents:
            for agent in env.possible_agents:
                total[agent] += env.meter.stats(env.sumo_env.sumo, agent).halting_total
    env.close()
    net = sumolib.net.readNet(str(manifest.path(manifest.net_file)))
    rows = []
    for agent in env.possible_agents:
        x, y = net.getNode(agent).getCoord()
        rows.append({"method": method, "tls": agent, "x": x, "y": y, "queue": total[agent] / steps})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="grid5x5")
    parser.add_argument("--profile", default="high")
    parser.add_argument("--methods", nargs="+", default=["fixed_time", "actuated", "ippo_local"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ctx = multiprocessing.get_context("spawn")
    jobs = [(args.scenario, args.profile, m, args.seed) for m in args.methods]
    with ctx.Pool(min(len(jobs), 4), maxtasksperchild=1) as pool:
        rows = [r for group in pool.starmap(queues, jobs) for r in group]
    out = paths.RESULTS_DIR / f"heatmap_{args.scenario}_{args.profile}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
