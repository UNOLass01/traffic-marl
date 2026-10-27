"""Webster before and after the saturation-flow correction (docs/decisions.md D31).

Compares the Webster results of the first Phase 4 run (1800 veh/h/lane for every
movement, lost time = yellow) with the corrected plans (measured saturation flow
per movement type and measured lost time), next to SUMO's default program.

Usage:
    python analysis/webster_comparison.py   # writes results/tables/webster_v1_v2.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trafficmarl.metrics.stats import summarize
from trafficmarl.utils import paths

METRICS = {
    "mean_waiting_time": "Waiting (s)",
    "mean_total_delay": "Total delay (s)",
    "max_waiting_time": "Max wait (s)",
    "teleport_rate": "Teleports (%)",
}
ORDER = ["single", "corridor", "grid3x3", "grid5x5", "lust"]


def _per_seed(root: Path, method: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in root.glob(f"*_*/{method}/*/metrics.csv")]
    table = pd.concat(frames, ignore_index=True)
    return table.groupby(["scenario", "profile", "seed"])[list(METRICS)].mean().reset_index()


def _cell(values: pd.Series, metric: str) -> str:
    s = summarize(values)
    scale = 100 if metric == "teleport_rate" else 1
    ci = f" ± {scale * s.ci_half_width:.1f}" if s.n > 1 else ""
    return f"{scale * s.mean:.1f}{ci}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", default=str(paths.RESULTS_DIR / "baselines_v1"))
    parser.add_argument("--new", default=str(paths.RESULTS_DIR / "baselines"))
    args = parser.parse_args()
    runs = {
        "webster (1800, L = yellow)": _per_seed(Path(args.old), "webster"),
        "webster (measured)": _per_seed(Path(args.new), "webster"),
        "fixed_time (SUMO default)": _per_seed(Path(args.new), "fixed_time"),
    }
    header = ["Scenario", "Demand", "Plan", *METRICS.values()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    keys = runs["webster (measured)"][["scenario", "profile"]].drop_duplicates()
    keys = keys.sort_values(
        ["scenario", "profile"], key=lambda c: c.map({s: i for i, s in enumerate(ORDER)})
        if c.name == "scenario" else c.map({"medium": 0, "high": 1, "peak": 2}),
    )  # fmt: skip
    for scenario, profile in keys.itertuples(index=False):
        for plan, data in runs.items():
            group = data[(data.scenario == scenario) & (data.profile == profile)]
            cells = [_cell(group[m], m) for m in METRICS]
            lines.append("| " + " | ".join([scenario, profile, plan, *cells]) + " |")
    markdown = "\n".join(lines)
    out = paths.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "webster_v1_v2.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
