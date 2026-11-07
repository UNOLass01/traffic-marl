"""Run a list of training configs over seeds (resumable).

Usage:
    python experiments/sweep.py --config configs/experiments/reward_check.yaml
    python experiments/sweep.py --config ... --extra algorithm.total_env_steps=600000
    python experiments/sweep.py --config configs/experiments/final_matrix.yaml \
        --parallel 2 --stop-starting "2026-11-07 16:00"

The sweep file lists ``runs`` (an experiment config and its overrides), the
``seeds`` and whether to ``evaluate`` each final policy. It may instead list
``tiers``, each with its own ``seeds`` and ``runs``, in order of priority.

By default trainings run one at a time in this process, each with its own Ray
instance and one SUMO process per environment runner. With ``--parallel N``, N
trainings run side by side as separate processes, taking jobs in file order; no
job starts after ``--stop-starting``, so a sweep with a deadline loses only its
lowest-priority jobs. Runs already trained are skipped (see experiments/train.py).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trafficmarl.utils import paths  # noqa: E402
from trafficmarl.utils.config import load_config  # noqa: E402
from trafficmarl.utils.logging_utils import get_logger, setup_logging  # noqa: E402
from train import run  # noqa: E402

logger = get_logger("sweep")
RAY_START_STAGGER_S = 20  # two Ray instances starting in the same instant can race for ports


def expand_jobs(sweep: dict) -> list[tuple[dict, int]]:
    """Return every (run, seed) of a sweep file, in order of priority."""
    tiers = sweep.get("tiers") or [{"seeds": sweep["seeds"], "runs": sweep["runs"]}]
    return [(item, int(seed)) for tier in tiers for seed in tier["seeds"] for item in tier["runs"]]


def job_command(item: dict, seed: int, extra: list[str], evaluate: bool) -> list[str]:
    """Build the train.py command line of one job."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--config",
        item["config"],
    ]
    command += ["--seed", str(seed), *(["--evaluate"] if evaluate else [])]
    return command + [*item.get("overrides", []), *extra]


def run_parallel(jobs: list[list[str]], n_lanes: int, stop_at: datetime | None, stem: str) -> int:
    """Run commands on ``n_lanes`` lanes; return the number of failed jobs."""
    queue = list(reversed(jobs))
    lock = threading.Lock()
    failures: list[list[str]] = []

    def lane(index: int) -> None:
        time.sleep(index * RAY_START_STAGGER_S)
        log_path = paths.RESULTS_DIR / f"sweep_{stem}_lane{index}.out"
        while True:
            with lock:
                if not queue:
                    return
                if stop_at is not None and datetime.now() >= stop_at:
                    logger.info(
                        "lane %d: stop time reached, %d jobs not started", index, len(queue)
                    )
                    return
                command = queue.pop()
            logger.info("lane %d: start %s", index, " ".join(command[2:]))
            with open(log_path, "a", encoding="utf-8") as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            logger.info("lane %d: exit %d for %s", index, code, " ".join(command[2:]))
            if code != 0:
                with lock:
                    failures.append(command)

    threads = [threading.Thread(target=lane, args=(i,)) for i in range(n_lanes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return len(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="overrides added to every run")
    parser.add_argument("--workers", type=int, default=None, help="evaluation processes")
    parser.add_argument("--parallel", type=int, default=1, help="trainings side by side")
    parser.add_argument("--stop-starting", default=None, metavar="'YYYY-MM-DD HH:MM'",
                        help="start no job after this time (with --parallel)")  # fmt: skip
    args = parser.parse_args()

    sweep = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stem = Path(args.config).stem
    setup_logging("INFO", paths.RESULTS_DIR / f"sweep_{stem}.log")
    evaluate = bool(sweep.get("evaluate", False))
    if args.parallel > 1:
        stop_at = datetime.fromisoformat(args.stop_starting) if args.stop_starting else None
        commands = [job_command(i, s, args.extra, evaluate) for i, s in expand_jobs(sweep)]
        failed = run_parallel(commands, args.parallel, stop_at, stem)
        logger.info("sweep finished, %d failed jobs", failed)
        sys.exit(1 if failed else 0)
    for item, seed in expand_jobs(sweep):
        config = load_config(item["config"], [*item.get("overrides", []), *args.extra])
        run(config, seed, evaluate, args.workers)


if __name__ == "__main__":
    main()
