"""Statistics over seeds and the acceptance rule used in every comparison (docs/decisions.md D27).

A method **beats** a baseline when

1. its mean waiting time is lower, and the difference is significant: the 95 %
   confidence intervals across seeds do not overlap, or Welch's t-test gives
   p < 0.05; and
2. it does not increase teleports or unfinished vehicles: for each of these, an
   increase counts when the method's mean is higher **and** that difference is
   not explained by seed noise (one-sided Welch p < 0.05, or any positive
   difference when both samples have zero variance).

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


def compare(
    method_waiting, baseline_waiting, method_teleports, baseline_teleports,
    method_unfinished, baseline_unfinished,
) -> Verdict:  # fmt: skip
    """Apply the acceptance rule (module docstring) to per-seed values."""
    lower = significantly_lower(method_waiting, baseline_waiting)
    tele = increased(method_teleports, baseline_teleports)
    unfinished = increased(method_unfinished, baseline_unfinished)
    return Verdict(
        beats=lower and not tele and not unfinished,
        waiting_lower=lower,
        teleports_increased=tele,
        unfinished_increased=unfinished,
        p_waiting=welch_p(method_waiting, baseline_waiting),
    )
