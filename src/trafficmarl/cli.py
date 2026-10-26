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
    """Simulate every route file headless and check teleports; write a CSV report."""
    import pandas as pd

    from trafficmarl.scenarios.spec import Manifest, load_scenario_config, output_dir
    from trafficmarl.scenarios.validate import validate

    rows = []
    for name in _scenario_names(args.names):
        cfg = load_scenario_config(name)
        manifest = Manifest.load(output_dir(cfg))
        rows += validate(manifest, **cfg["validation"])
    table = pd.DataFrame(rows)
    report = paths.RESULTS_DIR / "scenario_validation" / "validation.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(report, index=False)
    columns = [
        "scenario",
        "profile",
        "split",
        "inserted",
        "arrived",
        "teleports",
        "mean_waiting_time",
        "mean_travel_time",
        "running_at_end",
        "passed",
    ]
    print(table[columns].to_string(index=False))
    print(f"report: {report}")
    return 0 if table["passed"].all() else 1


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
    for action, func, help_text in (
        ("build", _cmd_scenarios_build, "generate networks and route files"),
        ("validate", _cmd_scenarios_validate, "run each route file headless and check teleports"),
    ):
        p = scen_sub.add_parser(action, help=help_text)
        p.add_argument("names", nargs="+", help="scenario names (configs/scenarios/) or 'all'")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
