"""Run every baseline: Webster plans, program baselines, max-pressure, centralized PPO.

Resumable: an evaluation whose ``metrics.csv`` files exist, or a training whose
``model.zip`` exists, is skipped.

Usage:
    python experiments/run_baselines.py                       # full Phase 4 set
    python experiments/run_baselines.py --scenarios single --profiles medium --seeds 0 1
"""

from __future__ import annotations

import argparse

from trafficmarl.baselines import centralized
from trafficmarl.envs.factory import experiment_config
from trafficmarl.metrics.evaluate import controller_spec, evaluate
from trafficmarl.utils import paths
from trafficmarl.utils.config import deep_merge, load_config
from trafficmarl.utils.logging_utils import get_logger, setup_logging

logger = get_logger("run_baselines")
PROGRAM_METHODS = ["fixed_time", "webster", "actuated", "max_pressure", "max_pressure_maxred"]


def _done(experiment: str, method: str, seeds: list[int]) -> bool:
    return all(
        (paths.RESULTS_DIR / experiment / method / str(s) / "metrics.csv").is_file() for s in seeds
    )


def _train_job(args: tuple[str, int, int | None]) -> dict:
    scenario, seed, timesteps = args
    run_dir = paths.RESULTS_DIR / "baselines" / "train_centralized_ppo" / scenario / str(seed)
    if (run_dir / "model.zip").is_file():
        return {"scenario": scenario, "seed": seed, "skipped": True}
    config = deep_merge(
        experiment_config(scenario),
        load_config(paths.CONFIGS_DIR / "algorithms/centralized_ppo.yaml"),
    )
    if timesteps:
        config["algorithm"]["total_timesteps"] = timesteps
    stats = centralized.train(config, seed, run_dir)
    logger.info("trained centralized_ppo %s seed %d: %s", scenario, seed, stats)
    return {"scenario": scenario, **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenarios", nargs="+", default=["single", "corridor", "grid3x3", "grid5x5", "lust"]
    )
    parser.add_argument("--profiles", nargs="+", default=["medium", "high", "peak"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--lust-seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="LuST has one evaluation file per profile and a high variance: 10 seeds (D34)",
    )
    parser.add_argument("--methods", nargs="+", default=PROGRAM_METHODS)
    parser.add_argument(
        "--centralized",
        nargs="*",
        default=["single", "corridor", "grid3x3"],
        help="scenarios on which the centralized agent is trained (5 seeds)",
    )
    parser.add_argument(
        "--centralized-probe",
        nargs="*",
        default=["grid5x5"],
        help="scenarios with a single short scalability probe (seed 0)",
    )
    parser.add_argument("--probe-timesteps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    setup_logging("INFO", paths.RESULTS_DIR / "baselines" / "run_baselines.log")

    # 1. Program baselines and max-pressure.
    for scenario in args.scenarios:
        for profile in args.profiles:
            experiment = f"baselines/{scenario}_{profile}"
            config = experiment_config(scenario, {"env": {"profile": profile}})
            seeds = args.lust_seeds if scenario == "lust" else args.seeds
            for method in args.methods:
                if _done(experiment, method, seeds):
                    continue
                spec = controller_spec(method)
                if method == "max_pressure_maxred":
                    spec["env"] = {"max_red": config["baselines"]["max_pressure"]["max_red"]}
                evaluate(config, spec, seeds, experiment=experiment, workers=args.workers)

    # 2. Centralized single agent: train on medium demand, evaluate on every profile.
    # Trainings run one at a time in this process: each starts its own SUMO subprocesses
    # (which a daemonic pool worker may not do), and one at a time keeps memory below
    # ~3.5 GB on an 8 GB machine.
    jobs = [(s, seed, None) for s in args.centralized for seed in args.seeds]
    jobs += [(s, 0, args.probe_timesteps) for s in args.centralized_probe]
    for job in jobs:
        logger.info("training finished: %s", _train_job(job))
    for scenario, seeds in [(s, args.seeds) for s in args.centralized] + [
        (s, [0]) for s in args.centralized_probe
    ]:
        checkpoint = str(
            paths.RESULTS_DIR
            / "baselines"
            / "train_centralized_ppo"
            / scenario
            / "{seed}"
            / "model.zip"
        )
        spec = controller_spec("centralized_ppo", checkpoint=checkpoint)
        for profile in args.profiles:
            experiment = f"baselines/{scenario}_{profile}"
            if _done(experiment, "centralized_ppo", seeds):
                continue
            config = experiment_config(scenario, {"env": {"profile": profile}})
            evaluate(config, spec, seeds, experiment=experiment, workers=args.workers)


if __name__ == "__main__":
    main()
