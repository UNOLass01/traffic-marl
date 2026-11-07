"""Phase 5 reward check: IPPO with the waiting vs the queue reward (docs/decisions.md D38),
and the queue reward with a 220 s max red (D40).

Produces, from ``results/reward_check_<scenario>/``:

* ``results/figures/reward_check_curves.png``: learning curves (halting queue
  per intersection during training; mean across seeds with a min-max band).
* ``results/tables/reward_check.md``: evaluation on the held-out route files of
  every profile, next to the Phase 4 baselines, with the verdict of the
  acceptance rule against the best program-based baseline (D33), and the
  training budget and plateau status of every run.

Usage:
    python analysis/reward_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_table import to_markdown  # noqa: E402
from marl_study import evaluation_table, learning_curves, training_table  # noqa: E402

from trafficmarl.utils import paths  # noqa: E402

PREFIX = "reward_check"
SCENARIOS = ["single", "corridor"]
VARIANTS = {
    "ippo_waiting": ("waiting reward", "#2a78d6"),
    "ippo_queue": ("queue reward", "#eb6834"),
    "ippo_queue_maxred": ("queue reward, max red 220 s", "#1baa7c"),
}


def main() -> None:
    learning_curves(
        paths.RESULTS_DIR / "figures" / "reward_check_curves.png", PREFIX, SCENARIOS, VARIANTS
    )
    markdown = (
        "## Training\n\n" + training_table(PREFIX, SCENARIOS)
        + "\n\n## Evaluation (held-out route files)\n\n"
        + to_markdown(evaluation_table(PREFIX, SCENARIOS))
    )  # fmt: skip
    out = paths.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward_check.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
