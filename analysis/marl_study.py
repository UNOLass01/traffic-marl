"""Shared figures and tables for the MARL studies (reward check, coordination, ...).

A study lives in ``results/<prefix>_<scenario>/``: one folder per method with
one training run per seed, and ``eval_<profile>/<method>/<seed>/metrics.csv``
for the evaluation of each final policy on the held-out route files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_table import build, per_seed  # noqa: E402

from trafficmarl.utils import paths  # noqa: E402

BASELINES = ["fixed_time", "webster", "actuated", "max_pressure", "max_pressure_maxred",
             "centralized_ppo"]  # fmt: skip


def study_root(prefix: str, scenario: str) -> Path:
    return paths.RESULTS_DIR / f"{prefix}_{scenario}"


def episodes(run_dir: Path) -> pd.DataFrame:
    """Training episodes in completion order, with cumulative environment steps."""
    files = sorted((run_dir / "episodes").glob("*.jsonl"))
    records = [json.loads(x) for f in files for x in f.read_text().splitlines() if x]
    table = pd.DataFrame(records).sort_values("wall_time")
    table["cum_env_steps"] = table.env_steps.cumsum()
    return table


def curves(run_dirs: list[Path], grid: np.ndarray, window: int = 10) -> np.ndarray:
    """One row per seed: rolling-mean queue per intersection, interpolated on ``grid``."""
    rows = []
    for run_dir in run_dirs:
        ep = episodes(run_dir)
        smooth = ep.approach_queue_per_signal.rolling(window, min_periods=1).mean()
        rows.append(np.interp(grid, ep.cum_env_steps, smooth, right=np.nan))
    return np.array(rows)


def learning_curves(
    out: Path, prefix: str, scenarios: list[str], variants: dict[str, tuple[str, str]]
) -> None:
    """Learning curves of every variant, one panel per scenario.

    The y-axis is the mean halting queue per intersection during each training
    episode, a traffic quantity that means the same thing under every reward
    (the returns do not: each reward has its own scale). Mean across seeds with
    a min-max band.
    """
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 4),
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for ax, scenario in zip(axes[0], scenarios, strict=True):
        root = study_root(prefix, scenario)
        for method, (label, color) in variants.items():
            run_dirs = [r.parent for r in sorted((root / method).glob("*/episodes"))]
            if not run_dirs:
                continue
            longest = max(episodes(r).cum_env_steps.iloc[-1] for r in run_dirs)
            grid = np.linspace(0, longest, 200)
            rows = curves(run_dirs, grid)
            mean = np.nanmean(rows, axis=0)
            ax.fill_between(grid, np.nanmin(rows, axis=0), np.nanmax(rows, axis=0),
                            color=color, alpha=0.15, linewidth=0)  # fmt: skip
            ax.plot(grid, mean, color=color, linewidth=2, label=f"{label} (n={len(run_dirs)})")
        ax.set_title(f"{scenario} (medium demand)", fontsize=11, loc="left")
        ax.set_xlabel("environment steps (decisions of the whole network)")
        ax.set_ylabel("halting vehicles per intersection")
        ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def evaluation_metrics(prefix: str, scenarios: list[str], baselines=BASELINES) -> pd.DataFrame:
    """All evaluation episodes of the study, plus the Phase 4 baselines of the same scenarios."""
    frames = []
    for scenario in scenarios:
        root = study_root(prefix, scenario)
        frames += [pd.read_csv(f) for f in root.glob("eval_*/*/*/metrics.csv")]
        trained = {p.parent.name for p in root.glob("eval_*/*/*")}
        for method in set(baselines) - trained:  # a run of this study replaces the Phase 4 one
            frames += [pd.read_csv(f) for f in (paths.RESULTS_DIR / "baselines").glob(
                f"{scenario}_*/{method}/*/metrics.csv")]  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def evaluation_table(prefix: str, scenarios: list[str], baselines=BASELINES) -> pd.DataFrame:
    """Mean and 95 % CI per scenario, demand and method, with the D33 verdict."""
    return build(per_seed(evaluation_metrics(prefix, scenarios, baselines)))


def training_table(prefix: str, scenarios: list[str]) -> str:
    lines = ["| Scenario | Method | Seed | Env steps | Wall time (min) | Steps/s | Gridlocks "
             "| Plateaued | Gain in last fifth (share of total) |", "|" + "---|" * 9]  # fmt: skip
    for scenario in scenarios:
        root = study_root(prefix, scenario)
        for stats_file in sorted(root.glob("*/*/training_stats.json")):
            s = json.loads(stats_file.read_text())
            p = s.get("plateau") or {}  # the SB3 centralized trainer has no plateau record
            plateaued = {True: "yes", False: "no"}.get(p.get("plateaued"), "n/a")
            lines.append(
                f"| {scenario} | {stats_file.parent.parent.name} | {s['seed']} | {s['env_steps']} "
                f"| {s['wall_time_s'] / 60:.1f} | {s['env_steps_per_s']:.0f} "
                f"| {s['training_gridlocks']} | {plateaued} "
                f"| {p.get('recent_gain_share')} |"
            )
    return "\n".join(lines)
