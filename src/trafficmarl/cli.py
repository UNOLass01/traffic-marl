"""Command-line entry point: ``python -m trafficmarl <command>``.

Training, evaluation and baseline commands are added in later phases; the
experiment scripts in ``experiments/`` will be thin wrappers around them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from collections.abc import Sequence

import yaml

from trafficmarl import __version__
from trafficmarl.utils import paths
from trafficmarl.utils.config import ConfigError, apply_overrides, load_config

KEY_PACKAGES = (
    "torch",
    "ray",
    "stable-baselines3",
    "sumo-rl",
    "gymnasium",
    "pettingzoo",
    "supersuit",
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _cmd_info(_: argparse.Namespace) -> int:
    """Print the environment summary used to check a fresh setup."""
    rows = {
        "trafficmarl": __version__,
        "python": sys.version.split()[0],
        "conda env": os.environ.get("CONDA_DEFAULT_ENV", "-"),
        "SUMO_HOME": os.environ.get("SUMO_HOME", "NOT SET"),
        "LIBSUMO_AS_TRACI": os.environ.get("LIBSUMO_AS_TRACI", "0"),
        "project root": str(paths.PROJECT_ROOT),
        **{name: _package_version(name) for name in KEY_PACKAGES},
    }
    for key, value in rows.items():
        print(f"{key:<18} {value}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    """Print a config after resolving includes and overrides."""
    try:
        config = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(yaml.safe_dump(config, sort_keys=False), end="")
    return 0


def _scenario_names(requested: list[str]) -> list[str]:
    from trafficmarl.scenarios.spec import available_scenarios

    known = available_scenarios()
    names = known if requested == ["all"] else requested
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise SystemExit(f"error: unknown scenario(s) {unknown}; available: {known}")
    return names


def _cmd_scenarios_build(args: argparse.Namespace) -> int:
    """Generate networks, route files and manifests."""
    from trafficmarl.scenarios import build_scenario

    for name in _scenario_names(args.names):
        manifest = build_scenario(name)
        n_vehicles = sum(manifest.vehicles.values())
        print(
            f"{name}: {len(manifest.tls_ids)} TLS, {len(manifest.vehicles)} route files, "
            f"{n_vehicles} vehicles -> {manifest.root}"
        )
    return 0


def _cmd_scenarios_validate(args: argparse.Namespace) -> int:
    """Simulate every route file headless (parallel, resumable); CSV + log report."""
    import pandas as pd

    from trafficmarl.scenarios.validate import plan_jobs, read_results, run_validation
    from trafficmarl.utils.logging_utils import setup_logging

    setup_logging("INFO")
    jobs = plan_jobs(_scenario_names(args.names), args.profiles)
    report = paths.resolve(args.output)
    run_validation(jobs, report, workers=args.workers, fresh=args.fresh)
    table = pd.DataFrame(read_results(report, jobs))
    columns = [
        "scenario", "profile", "split", "controller", "inserted", "arrived", "teleports",
        "teleport_ratio", "mean_waiting_time", "drained_without_teleport", "passed",
    ]  # fmt: skip
    table = table.sort_values(["scenario", "profile", "split", "route_file"])
    print(table[columns].to_string(index=False))
    print(f"report: {report} (log: {report.with_suffix('.log')})")
    return 0 if (table["passed"] == "True").all() else 1


def _cmd_env_rollout(args: argparse.Namespace) -> int:
    """Run one episode with a heuristic policy and print reward-component statistics."""
    import json

    from trafficmarl.envs.factory import experiment_config
    from trafficmarl.envs.rollout import run_rollout

    overrides = {
        "env": {
            "profile": args.profile,
            "observation": {"type": args.obs},
            "reward": {"type": args.reward},
        }
    }
    config = apply_overrides(experiment_config(args.scenario, overrides), args.overrides)
    result = run_rollout(
        config, policy=args.policy, split=args.split, seed=args.seed, max_seconds=args.seconds
    )
    ep = result["episode"]
    print(
        f"{args.scenario} {args.profile}/{args.split} policy={args.policy}: "
        f"{result['steps']} steps, end={ep['end_reason']} at t={ep['sim_time']:.0f}s, "
        f"arrived {ep['arrived']}/{ep['departed']}, teleports {ep['teleports']} "
        f"({result['wall_time_s']}s)"
    )
    header = f"{'agent':>8} {'component':>15} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}"
    print(header)
    rows = {**result["reward_components"], "ALL": result["reward_components_all"]}
    for agent, comps in rows.items():
        for name in ("traffic", "switch_penalty", "reward"):
            st = comps[name]
            print(
                f"{agent:>8} {name:>15} {st['mean']:9.4f} {st['std']:9.4f} "
                f"{st['min']:9.4f} {st['max']:9.4f}"
            )
    out = (
        paths.RESULTS_DIR
        / "env_rollouts"
        / (
            f"{args.scenario}_{args.profile}_{args.split}_{args.policy}_{args.obs}_{args.reward}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"saved: {out}")
    return 0


def _cmd_env_topology(args: argparse.Namespace) -> int:
    """Approach coverage, neighbour slots and message volume per scenario."""
    import json

    from trafficmarl.envs.features import FeatureSpec
    from trafficmarl.envs.topology import build_topology, summarize
    from trafficmarl.envs.traffic_env import count_green_phases
    from trafficmarl.scenarios.spec import Manifest, load_scenario_config, output_dir

    base = load_config(paths.CONFIGS_DIR / "base.yaml")
    max_distance = base["env"]["observation"]["max_distance"]
    summaries = {}
    for name in _scenario_names(args.names):
        manifest = Manifest.load(output_dir(load_scenario_config(name)))
        net_file = str(manifest.path(manifest.net_file))
        topology = build_topology(net_file, max_distance)
        n_green = count_green_phases(net_file)
        spec = FeatureSpec(
            agents=sorted(topology), topology=topology, n_green=n_green,
            max_lanes=max(len(t.stop_lanes) for t in topology.values()),
            max_phases=max(n_green.values()), max_distance=max_distance,
            max_green=base["env"]["max_green"], extended=True, agent_id=False,
        )  # fmt: skip
        summary = summarize(topology, n_green, spec.message_dim)
        summary["obs_dim_local"] = spec.local_dim
        summary["obs_dim_extended"] = spec.obs_dim
        summaries[name] = summary
        print(name, json.dumps(summary))
    out = paths.RESULTS_DIR / "env_topology" / "topology_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trafficmarl",
        description="Multi-agent RL traffic signal control with SUMO.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    info = sub.add_parser("info", help="show environment and dependency versions")
    info.set_defaults(func=_cmd_info)

    config = sub.add_parser("config", help="print a resolved experiment config")
    config.add_argument("config", help="path to a YAML config (relative to the project root)")
    config.add_argument(
        "overrides",
        nargs="*",
        metavar="key.path=value",
        help="dotted-path overrides applied after loading",
    )
    config.set_defaults(func=_cmd_config)

    scenarios = sub.add_parser("scenarios", help="build and validate SUMO scenarios")
    scen_sub = scenarios.add_subparsers(dest="scenarios_command", metavar="<action>", required=True)
    build = scen_sub.add_parser("build", help="generate networks and route files")
    build.add_argument("names", nargs="+", help="scenario names (configs/scenarios/) or 'all'")
    build.set_defaults(func=_cmd_scenarios_build)

    validate = scen_sub.add_parser(
        "validate", help="run each route file headless (parallel, resumes from the report)"
    )
    validate.add_argument("names", nargs="+", help="scenario names (configs/scenarios/) or 'all'")
    validate.add_argument("--profiles", nargs="+", help="only these demand profiles")
    validate.add_argument(
        "--workers", type=int, default=None, help="parallel SUMO processes (default: CPUs - 1)"
    )
    validate.add_argument("--fresh", action="store_true", help="discard previous results")
    validate.add_argument(
        "--output", default="results/scenario_validation/validation.csv", help="CSV report"
    )
    validate.set_defaults(func=_cmd_scenarios_validate)

    env = sub.add_parser("env", help="inspect and exercise the multi-agent environment")
    env_sub = env.add_subparsers(dest="env_command", metavar="<action>", required=True)
    rollout = env_sub.add_parser("rollout", help="one episode with a heuristic policy")
    rollout.add_argument("scenario")
    rollout.add_argument("--policy", choices=("random", "keep", "switch"), default="random")
    rollout.add_argument("--profile", default="medium")
    rollout.add_argument("--split", choices=("train", "eval"), default="eval")
    rollout.add_argument("--obs", choices=("local", "extended"), default="local")
    rollout.add_argument("--reward", choices=("local", "neighborhood", "global"), default="local")
    rollout.add_argument("--seconds", type=float, default=None, help="stop early (sim time)")
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("overrides", nargs="*", metavar="key.path=value")
    rollout.set_defaults(func=_cmd_env_rollout)
    topology = env_sub.add_parser("topology", help="topology summary per scenario")
    topology.add_argument("names", nargs="+", help="scenario names or 'all'")
    topology.set_defaults(func=_cmd_env_topology)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
