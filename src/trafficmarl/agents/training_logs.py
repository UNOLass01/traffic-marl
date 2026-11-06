"""Run logs shared by every trainer: episode records, restart rollback, plateau rule.

This module stays free of Ray imports because the environment workers of the
off-policy trainers (QMIX, MADDPG) use it and must remain light processes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)

SUMMARY_KEYS = ("route_file", "end_reason", "sim_time", "departed", "arrived", "teleports",
                "teleport_rate", "approach_queue_per_signal", "gridlock_penalty")  # fmt: skip


def append_episode(
    path: Path | None, episode: int, steps: int, returns: dict[str, float], summary: dict[str, Any]
) -> None:
    """Append one finished training episode to a runner's ``episodes/*.jsonl`` file."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "wall_time": time.time(),
        "episode": episode,
        "env_steps": steps,
        "return_mean": float(np.mean(list(returns.values()))),
        **{k: summary[k] for k in SUMMARY_KEYS},
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def rollback_logs(
    run_dir: Path, iteration: int, marker: str = "algorithm_state.pkl"
) -> list[dict[str, Any]]:
    """Cut the run's logs back to the checkpoint at ``iteration``; return the kept progress rows.

    A run interrupted between two checkpoints has logged iterations (and
    episodes) whose training is lost on restore. Keeping them would count
    their environment steps twice on the learning curve, so progress rows
    after ``iteration`` and episodes that ended after the checkpoint was
    written are removed. ``marker`` is the checkpoint file whose modification
    time dates the checkpoint. After a completed run this changes nothing.
    """
    progress = pd.read_csv(run_dir / "progress.csv")
    kept = progress[progress.iteration <= iteration]
    if len(kept) < len(progress):
        kept.to_csv(run_dir / "progress.csv", index=False)
        logger.info("dropped %d iterations logged after the checkpoint", len(progress) - len(kept))
    saved_at = (run_dir / "checkpoint" / marker).stat().st_mtime
    for path in (run_dir / "episodes").glob("*.jsonl"):
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x]
        keep = [x for x in lines if json.loads(x)["wall_time"] <= saved_at]
        if len(keep) < len(lines):
            path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
    return kept.to_dict("records")


def load_episodes(run_dir: Path) -> pd.DataFrame:
    files = sorted((run_dir / "episodes").glob("*.jsonl"))
    records = [json.loads(line) for f in files for line in f.read_text().splitlines() if line]
    return pd.DataFrame(records).sort_values("wall_time") if records else pd.DataFrame()


def plateau(progress: pd.DataFrame, tolerance: float = 0.05) -> dict[str, Any]:
    """Has the learning curve flattened? (decisions.md D38)

    The curve (episode return per iteration) is cut into five equal parts by
    environment steps. It has plateaued when the gain between the last two
    fifths is below ``tolerance`` of the gain between the first and last fifth.
    """
    curve = progress.dropna(subset=["episode_return_mean"])
    if len(curve) < 10:
        return {"plateaued": False, "reason": "too few iterations"}
    parts = np.array_split(curve.episode_return_mean.to_numpy(), 5)
    first, fourth, last = parts[0].mean(), parts[3].mean(), parts[4].mean()
    total_gain = last - first
    recent_gain = last - fourth
    flat = total_gain <= 0 or recent_gain <= tolerance * abs(total_gain)
    return {
        "plateaued": bool(flat),
        "first_fifth": round(float(first), 2),
        "fourth_fifth": round(float(fourth), 2),
        "last_fifth": round(float(last), 2),
        "recent_gain_share": round(float(recent_gain / total_gain), 3) if total_gain else None,
    }


def training_stats(
    run_dir: Path, seed: int, steps: int, iteration: int, wall: float, device: str, rows: list[dict]
) -> dict[str, Any]:
    """Summary of a finished training, also written to ``training_stats.json``."""
    episodes = load_episodes(run_dir)
    stats = {
        "seed": seed,
        "env_steps": steps,
        "iterations": iteration,
        "wall_time_s": round(wall, 1),
        "env_steps_per_s": round(steps / wall, 1) if wall else 0.0,
        "episodes": len(episodes),
        "training_gridlocks": int((episodes.end_reason == "gridlock").sum())
        if len(episodes)
        else 0,
        "device": device,
        "plateau": plateau(pd.DataFrame(rows)),
    }
    (run_dir / "training_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
