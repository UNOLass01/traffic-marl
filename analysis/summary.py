"""Compact tables for the report (docs/results.md), from ``results/tables/main.csv``.

One table per metric: rows are methods, columns scenario x demand, cells are the mean over
runs (the full table with confidence intervals and n is ``results/tables/main.md``). Run
``analysis/study.py`` first.

Usage:
    python analysis/summary.py          # results/tables/summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_table import METHOD_ORDER, SCENARIO_ORDER  # noqa: E402

from trafficmarl.utils import paths  # noqa: E402

METRICS = {
    "mean_waiting_time": ("Mean waiting time per vehicle (s)", "{:.1f}"),
    "network_queue_mean": ("Mean network queue (halting vehicles)", "{:.1f}"),
    "throughput": ("Throughput (vehicles that completed their trip)", "{:.0f}"),
    "co2_total_kg": ("CO2 (kg per episode)", "{:.0f}"),
}
PROFILES = ["medium", "peak"]


def table(main: pd.DataFrame, metric: str, fmt: str) -> str:
    columns = [(s, p) for s in SCENARIO_ORDER for p in PROFILES if s in set(main.scenario)]
    header = "| Method | " + " | ".join(f"{s} {p}" for s, p in columns) + " |"
    lines = [header, "|" + "---|" * (len(columns) + 1)]
    for method in METHOD_ORDER:
        rows = main[main.method == method]
        if rows.empty:
            continue
        cells = []
        for scenario, profile in columns:
            row = rows[(rows.scenario == scenario) & (rows.profile == profile)]
            if row.empty:
                cells.append("")
                continue
            r = row.iloc[0]
            value = fmt.format(r[metric])
            cells.append(value + (" (n=1)" if r.n_seeds == 1 else ""))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    main_csv = pd.read_csv(paths.RESULTS_DIR / "tables" / "main.csv")
    parts = []
    for metric, (title, fmt) in METRICS.items():
        parts += [f"### {title}", "", table(main_csv, metric, fmt), ""]
    out = paths.RESULTS_DIR / "tables" / "summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
