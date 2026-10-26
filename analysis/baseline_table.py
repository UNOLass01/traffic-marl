"""Baseline results table (Phase 4) from ``results/baselines/``.

For every scenario, demand profile and method: per-seed means (over the
evaluation route files), then mean and 95 % confidence interval across seeds,
and the verdict of the acceptance rule (docs/decisions.md D27) against Webster.

Usage:
    python analysis/baseline_table.py            # writes results/tables/baselines.{csv,md}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trafficmarl.metrics.stats import compare, summarize
from trafficmarl.utils import paths

REFERENCE = "webster"
METHOD_ORDER = ["fixed_time", "webster", "actuated", "max_pressure", "centralized_ppo"]
SCENARIO_ORDER = ["single", "corridor", "grid3x3", "grid5x5", "lust"]
PROFILE_ORDER = ["medium", "high", "peak"]
METRICS = {
    "mean_waiting_time": "Waiting (s)",
    "mean_total_delay": "Total delay (s)",
    "throughput": "Throughput",
    "unfinished_total": "Unfinished",
    "teleport_rate": "Teleports (%)",
    "network_queue_mean": "Queue (veh)",
    "co2_total_kg": "CO2 (kg)",
    "max_waiting_time": "Max wait (s)",
}


def load(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in root.glob("*_*/*/*/metrics.csv")]
    if not frames:
        raise SystemExit(f"no metrics.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def per_seed(table: pd.DataFrame) -> pd.DataFrame:
    numeric = list(METRICS) + ["teleports", "teleport_flag", "gridlock"]
    return table.groupby(["scenario", "profile", "method", "seed"])[numeric].mean().reset_index()


def build(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, profile), group in seeds.groupby(["scenario", "profile"]):
        ref = group[group.method == REFERENCE]
        for method, m in group.groupby("method"):
            row = {"scenario": scenario, "profile": profile, "method": method, "n_seeds": len(m)}
            for metric in METRICS:
                s = summarize(m[metric])
                row[metric] = s.mean
                row[f"{metric}_ci"] = s.ci_half_width
            row["teleport_flag_any"] = bool(m.teleport_flag.any())
            row["gridlock_any"] = bool(m.gridlock.any())
            if method != REFERENCE and len(ref) and len(m) > 1:
                v = compare(
                    m.mean_waiting_time, ref.mean_waiting_time, m.teleports, ref.teleports,
                    m.unfinished_total, ref.unfinished_total,
                )  # fmt: skip
                row["beats_webster"] = v.beats
                row["p_waiting_vs_webster"] = v.p_waiting
            rows.append(row)
    table = pd.DataFrame(rows)
    table["_s"] = table.scenario.map({s: i for i, s in enumerate(SCENARIO_ORDER)})
    table["_p"] = table.profile.map({p: i for i, p in enumerate(PROFILE_ORDER)})
    table["_m"] = table.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    return table.sort_values(["_s", "_p", "_m"]).drop(columns=["_s", "_p", "_m"])


def to_markdown(table: pd.DataFrame) -> str:
    header = ["Scenario", "Demand", "Method", "n", *METRICS.values(), "vs Webster"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table.itertuples():
        cells = [r.scenario, r.profile, r.method, str(r.n_seeds)]
        for metric in METRICS:
            value, ci = getattr(r, metric), getattr(r, f"{metric}_ci")
            if metric == "teleport_rate":
                value, ci = 100 * value, 100 * ci
            ci_text = f" ± {ci:.1f}" if ci == ci else ""  # NaN when n = 1
            cells.append(f"{value:.1f}{ci_text}")
        if r.teleport_flag_any:
            cells[header.index("Teleports (%)")] += " ⚑"
        verdict = getattr(r, "beats_webster", float("nan"))
        cells.append(
            "reference"
            if r.method == REFERENCE
            else ("beats" if verdict is True else ("—" if verdict == verdict else "n/a"))
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(paths.RESULTS_DIR / "baselines"))
    args = parser.parse_args()
    table = build(per_seed(load(Path(args.root))))
    out = paths.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "baselines.csv", index=False)
    markdown = to_markdown(table)
    (out / "baselines.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
