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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)
