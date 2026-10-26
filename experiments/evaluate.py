"""Evaluate a controller as described by an experiment config.

Usage:
    python experiments/evaluate.py --config configs/experiments/<name>.yaml --method fixed_time
    python experiments/evaluate.py --config ... --method fixed_time evaluation.seeds=[0,1]

The config provides the scenario (via ``defaults``), ``env.profile``,
``experiment.name`` and ``evaluation.seeds`` (default: ``experiment.seeds``).
Learned policies (``--checkpoint``) are added in Phase 5.
"""

from __future__ import annotations

import argparse

from trafficmarl.metrics.evaluate import controller_spec, evaluate
from trafficmarl.utils.config import load_config
from trafficmarl.utils.logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("overrides", nargs="*", metavar="key.path=value")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config, args.overrides)
    seeds = config.get("evaluation", {}).get("seeds", config["experiment"]["seeds"])
    evaluate(
        config, controller_spec(args.method), list(seeds),
        experiment=config["experiment"]["name"], workers=args.workers,
    )  # fmt: skip


if __name__ == "__main__":
    main()
