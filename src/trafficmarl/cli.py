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
from trafficmarl.utils.config import ConfigError, load_config

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
