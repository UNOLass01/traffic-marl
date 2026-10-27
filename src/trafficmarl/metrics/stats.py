"""Statistics over seeds and the acceptance rule of every comparison (docs/decisions.md D27, D33).

A method **beats** a reference when

1. its mean waiting time is lower, and the difference is significant: the 95 %
   confidence intervals across seeds do not overlap, or Welch's t-test gives
   p < 0.05;
2. its 95th-percentile total delay per vehicle is not higher; and
3. it does not increase teleports or unfinished vehicles.

For 2 and 3, an increase counts when the method's mean is higher **and** that
difference is not explained by seed noise (one-sided Welch p < 0.05, or any
positive difference when both samples have zero variance).

For the MARL methods the reference is the best program-based baseline of each
scenario and demand (D33, :func:`best_program_baseline`).

Every statistic is computed on per-seed values (the mean over a seed's
evaluation episodes), so n = number of seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.05


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float

    @property
    def ci_half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2


def summarize(values) -> Summary:
    """Mean, sample std and t-based 95 % confidence interval of per-seed values."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return Summary(n, mean, float("nan"), float("nan"), float("nan"))
    std = float(arr.std(ddof=1))
    half = float(stats.t.ppf(1 - ALPHA / 2, n - 1)) * std / np.sqrt(n)
    return Summary(n, mean, std, mean - half, mean + half)


def welch_p(a, b, alternative: str = "two-sided") -> float:
    """Welch's t-test p-value; NaN when both samples have zero variance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 and b.std() == 0:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)


def significantly_lower(method, baseline) -> bool:
    m, b = summarize(method), summarize(baseline)
    if not m.mean < b.mean:
        return False
    if m.ci_high < b.ci_low:
        return True
    p = welch_p(method, baseline)
    return bool(p < ALPHA) if not np.isnan(p) else True  # zero variance: any difference is real


def increased(method, baseline) -> bool:
    """True when the method's mean is higher and the difference is not seed noise."""
    m_mean, b_mean = float(np.mean(method)), float(np.mean(baseline))
    if m_mean <= b_mean:
        return False
    p = welch_p(method, baseline, alternative="greater")
    return True if np.isnan(p) else bool(p < ALPHA)


@dataclass(frozen=True)
class Verdict:
    beats: bool
    waiting_lower: bool
    teleports_increased: bool
    unfinished_increased: bool
    p_waiting: float
    tail_delay_increased: bool = False


def compare(
    method_waiting, baseline_waiting, method_teleports, baseline_teleports,
    method_unfinished, baseline_unfinished, method_tail_delay=None, baseline_tail_delay=None,
) -> Verdict:  # fmt: skip
    """Apply the acceptance rule (module docstring) to per-seed values.

    The tail-delay condition is checked when both ``*_tail_delay`` series are given.
    """
    lower = significantly_lower(method_waiting, baseline_waiting)
    tele = increased(method_teleports, baseline_teleports)
    unfinished = increased(method_unfinished, baseline_unfinished)
    tail = (
        increased(method_tail_delay, baseline_tail_delay)
        if method_tail_delay is not None and baseline_tail_delay is not None
        else False
    )
    return Verdict(
        beats=lower and not tele and not unfinished and not tail,
        waiting_lower=lower,
        teleports_increased=tele,
        unfinished_increased=unfinished,
        p_waiting=welch_p(method_waiting, baseline_waiting),
        tail_delay_increased=tail,
    )


PROGRAM_BASELINES = ("fixed_time", "webster", "actuated")


def best_program_baseline(mean_waiting: dict[str, float]) -> str:
    """The program-based baseline with the lowest mean waiting time (D33)."""
    candidates = {m: v for m, v in mean_waiting.items() if m in PROGRAM_BASELINES}
    if not candidates:
        raise ValueError(f"no program baseline among {sorted(mean_waiting)}")
    return min(candidates, key=candidates.get)
