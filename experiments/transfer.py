"""Transfer: evaluate policies trained on one network on a larger one, without retraining.

The observation layout does not depend on the network (D20), so a shared policy
runs unchanged elsewhere. Results go to ``results/transfer_<source>_<target>/``.

Usage:
    python experiments/transfer.py --source grid3x3 --target grid5x5 \
        --methods ippo_local ippo_extended mappo qmix
"""

from __future__ import annotations

import argparse

from trafficmarl.envs.factory import experiment_config
from trafficmarl.metrics.evaluate import controller_spec, evaluate
from trafficmarl.utils import paths
from trafficmarl.utils.config import deep_merge, load_config
from trafficmarl.utils.logging_utils import setup_logging

ALGORITHM = {"mappo": "mappo", "qmix": "qmix", "maddpg": "maddpg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="grid3x3")
    parser.add_argument("--target", default="grid5x5")
    parser.add_argument(
        "--methods", nargs="+", default=["ippo_local", "ippo_extended", "mappo", "qmix"]
    )
    parser.add_argument("--profiles", nargs="+", default=["medium", "high", "peak"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    setup_logging("INFO", paths.RESULTS_DIR / f"transfer_{args.source}_{args.target}" / "run.log")
    for method in args.methods:
        trained = paths.RESULTS_DIR / f"marl_{args.source}" / method
        saved = load_config(trained / str(args.seeds[0]) / "config.yaml")
        spec = controller_spec(
            ALGORITHM.get(method, "ippo"), checkpoint=str(trained / "{seed}" / "checkpoint")
        )
        spec["name"] = method
        for profile in args.profiles:
            config = deep_merge(
                experiment_config(args.target), {"env": {**saved["env"], "profile": profile}}
            )
            evaluate(
                config, spec, args.seeds,
                experiment=f"transfer_{args.source}_{args.target}/eval_{profile}",
                workers=args.workers,
            )  # fmt: skip


if __name__ == "__main__":
    main()
