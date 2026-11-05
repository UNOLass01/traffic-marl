"""Phase 5 reward check: IPPO with the waiting vs the queue reward (docs/decisions.md D38).

Produces, from ``results/reward_check_<scenario>/``:

* ``results/figures/reward_check_curves.png``: learning curves. The y-axis is the
  mean halting queue per intersection during each training episode, a traffic
  quantity that means the same thing under both rewards (the returns do not:
  each reward has its own scale). Mean across seeds with a min-max band.
* ``results/tables/reward_check.md``: evaluation on the held-out route files of
  every profile, next to the Phase 4 baselines, with the verdict of the
  acceptance rule against the best program-based baseline (D33), and the
  training budget and plateau status of every run.

Usage:
    python analysis/reward_check.py
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
from baseline_table import build, per_seed, to_markdown  # noqa: E402

from trafficmarl.utils import paths  # noqa: E402

SCENARIOS = ["single", "corridor"]
VARIANTS = {
    "ippo_waiting": ("waiting reward", "#2a78d6"),
    "ippo_queue": ("queue reward", "#eb6834"),
}
BASELINES = ["fixed_time", "webster", "actuated", "max_pressure", "max_pressure_maxred",
             "centralized_ppo"]  # fmt: skip


def _episodes(run_dir: Path) -> pd.DataFrame:
    """Training episodes in completion order, with cumulative environment steps."""
    files = sorted((run_dir / "episodes").glob("*.jsonl"))
    records = [json.loads(x) for f in files for x in f.read_text().splitlines() if x]
    table = pd.DataFrame(records).sort_values("wall_time")
    table["cum_env_steps"] = table.env_steps.cumsum()
    return table


def _curve(run_dirs: list[Path], grid: np.ndarray, window: int = 10) -> np.ndarray:
    """One row per seed: rolling-mean queue per intersection, interpolated on ``grid``."""
    rows = []
    for run_dir in run_dirs:
        ep = _episodes(run_dir)
        smooth = ep.approach_queue_per_signal.rolling(window, min_periods=1).mean()
        rows.append(np.interp(grid, ep.cum_env_steps, smooth, right=np.nan))
    return np.array(rows)


def figure(out: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(11, 4), constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        root = paths.RESULTS_DIR / f"reward_check_{scenario}"
        for method, (label, color) in VARIANTS.items():
            runs = sorted((root / method).glob("*/episodes"))
            if not runs:
                continue
            run_dirs = [r.parent for r in runs]
            longest = max(_episodes(r).cum_env_steps.iloc[-1] for r in run_dirs)
            grid = np.linspace(0, longest, 200)
            curves = _curve(run_dirs, grid)
            mean = np.nanmean(curves, axis=0)
            ax.fill_between(grid, np.nanmin(curves, axis=0), np.nanmax(curves, axis=0),
                            color=color, alpha=0.15, linewidth=0)  # fmt: skip
            ax.plot(grid, mean, color=color, linewidth=2, label=f"{label} (n={len(run_dirs)})")
            last = np.flatnonzero(~np.isnan(mean))[-1]
            ax.annotate(label, (grid[last], mean[last]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color="#3d3d3a",
                        va="center")  # fmt: skip
        ax.set_title(f"{scenario} (medium demand)", fontsize=11, loc="left")
        ax.set_xlabel("environment steps (decisions of the whole network)")
        ax.set_ylabel("halting vehicles per intersection")
        ax.grid(axis="y", color="#e4e3dc", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _load_metrics(pattern_root: Path, pattern: str) -> list[pd.DataFrame]:
    return [pd.read_csv(f) for f in pattern_root.glob(pattern)]


def evaluation_table() -> str:
    frames = []
    for scenario in SCENARIOS:
        root = paths.RESULTS_DIR / f"reward_check_{scenario}"
        frames += _load_metrics(root, "eval_*/*/*/metrics.csv")
        for method in BASELINES:
            frames += _load_metrics(
                paths.RESULTS_DIR / "baselines", f"{scenario}_*/{method}/*/metrics.csv"
            )
    table = build(per_seed(pd.concat(frames, ignore_index=True)))
    return to_markdown(table)


def training_table() -> str:
    lines = ["| Scenario | Method | Seed | Env steps | Wall time (min) | Steps/s | Gridlocks "
             "| Plateaued | Gain in last fifth (share of total) |", "|" + "---|" * 9]  # fmt: skip
    for scenario in SCENARIOS:
        root = paths.RESULTS_DIR / f"reward_check_{scenario}"
        for stats_file in sorted(root.glob("ippo_*/*/training_stats.json")):
            s = json.loads(stats_file.read_text())
            p = s["plateau"]
            lines.append(
                f"| {scenario} | {stats_file.parent.parent.name} | {s['seed']} | {s['env_steps']} "
                f"| {s['wall_time_s'] / 60:.1f} | {s['env_steps_per_s']:.0f} "
                f"| {s['training_gridlocks']} | {'yes' if p['plateaued'] else 'no'} "
                f"| {p.get('recent_gain_share')} |"
            )
    return "\n".join(lines)


def main() -> None:
    figure(paths.RESULTS_DIR / "figures" / "reward_check_curves.png")
    markdown = (
        "## Training\n\n" + training_table() + "\n\n## Evaluation (held-out route files)\n\n"
        + evaluation_table()
    )  # fmt: skip
    out = paths.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward_check.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
