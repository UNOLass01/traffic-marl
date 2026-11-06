"""Train a MARL method from an experiment config, then optionally evaluate it.

Usage:
    python experiments/train.py --config configs/experiments/ippo_single.yaml --seed 0
    python experiments/train.py --config ... --seed 0 env.reward.measure=queue \
        experiment.method=ippo_queue --evaluate

Training writes ``results/<experiment>/<method>/<seed>/``: resolved config,
git commit, ``progress.csv`` (one row per iteration), ``episodes/`` (one line
per training episode), TensorBoard logs, ``checkpoint/`` and
``training_stats.json``. With ``--evaluate`` the final policy is evaluated
(deterministically, held-out route files, SUMO seed = training seed) on every
profile of ``evaluation.profiles``, into
``results/<experiment>/eval_<profile>/<method>/<seed>/metrics.csv``.
A run that already reached ``algorithm.total_env_steps`` is not trained again; a
run trained with a smaller budget is extended from its checkpoint (D38).
"""

from __future__ import annotations

import argparse
import json

from trafficmarl.utils import paths
from trafficmarl.utils.config import deep_merge, load_config
from trafficmarl.utils.logging_utils import get_logger, setup_logging

logger = get_logger("train")
TRAINERS = {
    "ippo": "trafficmarl.agents.ippo",
    "mappo": "trafficmarl.agents.mappo",
    "qmix": "trafficmarl.agents.qmix",
    "maddpg": "trafficmarl.agents.maddpg",
    "federated": "trafficmarl.federated.trainer",
    "centralized_ppo": "trafficmarl.baselines.centralized",
}
CHECKPOINT_NAME = {"centralized_ppo": "model.zip"}  # the others save "checkpoint/"


def run(config: dict, seed: int, evaluate_after: bool, workers: int | None = None) -> dict:
    import importlib

    from trafficmarl.metrics.evaluate import controller_spec, evaluate

    name = config["experiment"]["name"]
    algorithm = config["algorithm"]["name"]
    method = config["experiment"].get("method", algorithm)
    run_dir = paths.RESULTS_DIR / name / method / str(seed)
    stats_file = run_dir / "training_stats.json"
    algo_cfg = config["algorithm"]
    budget = int(algo_cfg.get("total_env_steps", algo_cfg.get("total_timesteps")))
    stats = json.loads(stats_file.read_text()) if stats_file.is_file() else None
    trained_now = not (stats and stats["env_steps"] >= budget)
    if not trained_now:
        logger.info("already trained: %s", run_dir)
    else:
        trainer = importlib.import_module(TRAINERS[algorithm])
        stats = trainer.train(config, seed, run_dir)
        logger.info("trained %s/%s seed %d: %s", name, method, seed, stats)
    if evaluate_after:
        checkpoint = str(run_dir / CHECKPOINT_NAME.get(algorithm, "checkpoint"))
        spec = {**controller_spec(algorithm, checkpoint=checkpoint), "name": method}
        for profile in config.get("evaluation", {}).get("profiles", [config["env"]["profile"]]):
            experiment = f"{name}/eval_{profile}"
            done = (paths.RESULTS_DIR / experiment / method / str(seed) / "metrics.csv").is_file()
            if done and not trained_now:  # a new or extended policy is evaluated again
                continue
            eval_config = deep_merge(config, {"env": {"profile": profile}})
            # A LuST run covers several SUMO seeds (D34): 5 runs x 2 = SUMO seeds 0-9.
            per_run = int(config.get("evaluation", {}).get("sumo_seeds_per_run", 1))
            n_runs = len(config["experiment"]["seeds"])
            sumo_seeds = {seed: [seed + i * n_runs for i in range(per_run)]}
            evaluate(eval_config, spec, [seed], experiment=experiment, workers=workers,
                     sumo_seeds=sumo_seeds)  # fmt: skip
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="training seed, repeatable (default: experiment.seeds)")  # fmt: skip
    parser.add_argument("--evaluate", action="store_true", help="evaluate the final policy")
    parser.add_argument("--workers", type=int, default=None, help="evaluation processes")
    parser.add_argument("overrides", nargs="*", metavar="key.path=value")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    name = config["experiment"]["name"]
    setup_logging("INFO", paths.RESULTS_DIR / name / "train.log")
    for seed in args.seed if args.seed is not None else config["experiment"]["seeds"]:
        run(config, seed, args.evaluate, args.workers)


if __name__ == "__main__":
    main()
