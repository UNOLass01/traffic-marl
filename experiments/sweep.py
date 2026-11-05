"""Run a list of training configs over seeds, one training at a time (resumable).

Usage:
    python experiments/sweep.py --config configs/experiments/reward_check.yaml
    python experiments/sweep.py --config ... --extra algorithm.total_env_steps=600000

The sweep file lists ``runs`` (an experiment config and its overrides), the
``seeds`` and whether to ``evaluate`` each final policy. Trainings run
sequentially: each one starts its own Ray instance with one SUMO process per
environment runner. Runs already trained are skipped (see experiments/train.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trafficmarl.utils import paths  # noqa: E402
from trafficmarl.utils.config import load_config  # noqa: E402
from trafficmarl.utils.logging_utils import setup_logging  # noqa: E402
from train import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="overrides added to every run")
    parser.add_argument("--workers", type=int, default=None, help="evaluation processes")
    args = parser.parse_args()

    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    setup_logging("INFO", paths.RESULTS_DIR / f"sweep_{Path(args.config).stem}.log")
    for seed in sweep["seeds"]:
        for item in sweep["runs"]:
            config = load_config(item["config"], [*item.get("overrides", []), *args.extra])
            run(config, int(seed), bool(sweep.get("evaluate", False)), args.workers)


if __name__ == "__main__":
    main()
