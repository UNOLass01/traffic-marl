"""Phase 6: IPPO without coordination vs implicit and explicit coordination (D43).

Produces, from ``results/coordination_<scenario>/``:

* ``results/figures/coordination_curves.png``: learning curves of the four
  variants (halting queue per intersection during training).
* ``results/tables/coordination.md``:
  - each coordinated variant against independent IPPO (``ippo_local``) on the
    held-out route files: change in mean waiting time and p95 total delay,
    Welch p-value, throughput, and the communication it needs at execution and
    during training (floats per decision step, coordination/communication.py);
  - the full evaluation table with the D33 verdict against the best
    program-based baseline;
  - the training budget and plateau status of every run.

Usage:
    python analysis/coordination.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_table import PROFILE_ORDER, to_markdown  # noqa: E402
from marl_study import (  # noqa: E402
    evaluation_metrics,
    evaluation_table,
    learning_curves,
    training_table,
)

from trafficmarl.metrics.stats import significantly_lower, summarize, welch_p  # noqa: E402
from trafficmarl.utils import paths  # noqa: E402

PREFIX = "marl"
SCENARIOS = ["corridor", "grid3x3"]
REFERENCE = "ippo_local"
VARIANTS = {
    "ippo_local": ("independent (local reward)", "#3d3d3a"),
    "ippo_neighborhood": ("neighbourhood reward", "#2a78d6"),
    "ippo_global": ("global reward", "#eb6834"),
    "ippo_extended": ("neighbour observation", "#1baa7c"),
}
PER_SEED = ["mean_waiting_time", "p95_total_delay", "throughput", "unfinished_total",
            "communication_floats_per_step", "training_communication_floats_per_step"]  # fmt: skip


def versus_independent(metrics: pd.DataFrame) -> str:
    """Each coordinated variant against independent IPPO, per scenario and demand."""
    seeds = (
        metrics[metrics.method.isin(VARIANTS)]
        .groupby(["scenario", "profile", "method", "seed"])[PER_SEED]
        .mean()
        .reset_index()
    )
    header = ["Scenario", "Demand", "Method", "n", "Waiting (s)", "Δ waiting", "p (Welch)",
              "Lower?", "p95 delay (s)", "Δ p95 delay", "Throughput", "Floats/step (exec)",
              "Floats/step (train)"]  # fmt: skip
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for scenario in SCENARIOS:
        for profile in PROFILE_ORDER:
            group = seeds[(seeds.scenario == scenario) & (seeds.profile == profile)]
            ref = group[group.method == REFERENCE]
            if ref.empty:
                continue
            for method in VARIANTS:
                m = group[group.method == method]
                if m.empty:
                    continue
                w, d = summarize(m.mean_waiting_time), summarize(m.p95_total_delay)
                cells = [scenario, profile, method, str(len(m)),
                         f"{w.mean:.1f} ± {w.ci_half_width:.1f}"]  # fmt: skip
                if method == REFERENCE:
                    cells += ["reference", "", "", f"{d.mean:.1f} ± {d.ci_half_width:.1f}", ""]
                else:
                    ref_w, ref_d = ref.mean_waiting_time.mean(), ref.p95_total_delay.mean()
                    lower = significantly_lower(m.mean_waiting_time, ref.mean_waiting_time)
                    cells += [
                        f"{100 * (w.mean / ref_w - 1):+.0f} %",
                        f"{welch_p(m.mean_waiting_time, ref.mean_waiting_time):.3f}",
                        "yes" if lower else "no",
                        f"{d.mean:.1f} ± {d.ci_half_width:.1f}",
                        f"{100 * (d.mean / ref_d - 1):+.0f} %",
                    ]
                cells += [
                    f"{m.throughput.mean():.0f}",
                    f"{m.communication_floats_per_step.mean():.0f}",
                    f"{m.training_communication_floats_per_step.mean():.0f}",
                ]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    learning_curves(
        paths.RESULTS_DIR / "figures" / "coordination_curves.png", PREFIX, SCENARIOS, VARIANTS
    )
    markdown = (
        "## Coordinated variants against independent IPPO (held-out route files)\n\n"
        + versus_independent(evaluation_metrics(PREFIX, SCENARIOS, baselines=[]))
        + "\n\n## Evaluation against the baselines\n\n"
        + to_markdown(evaluation_table(PREFIX, SCENARIOS))
        + "\n\n## Training\n\n" + training_table(PREFIX, SCENARIOS)
    )  # fmt: skip
    out = paths.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coordination.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
