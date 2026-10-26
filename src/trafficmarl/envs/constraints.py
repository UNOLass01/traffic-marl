"""Phase-duration constraints enforced by action masking (docs/decisions.md D13).

SUMO-RL ignores phase-change requests during the minimum green without telling
the agent, and never enforces the maximum green. Here every decision gets a
mask over the (padded) green phases instead:

* minimum green: until SUMO-RL's own rule allows a change
  (``time since last change >= yellow + min_green``), only "keep the current
  phase" is valid, so no request can be silently ignored;
* maximum green: if keeping the phase would extend its green beyond
  ``max_green`` before the next decision, "keep" is masked out;
* padding: phases that do not exist at this intersection are always masked;
* optional maximum red (off by default, D19): if a phase would stay red longer
  than ``max_red`` unless chosen now, only such phases are valid.

Each signal also records how often a constraint was active and the longest
red time of each phase, reported for every method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class InvalidActionError(ValueError):
    """An agent chose a phase that its action mask forbids."""


@dataclass
class SignalConstraintState:
    n_green: int
    green_phase: int = 0
    green_start: float = 0.0  # time the current green began (after its yellow)
    red_start: dict[int, float] = field(default_factory=dict)  # phase -> start of its red
    longest_red: dict[int, float] = field(default_factory=dict)
    decisions: int = 0
    switches: int = 0
    min_green_active: int = 0  # decisions where only "keep" was valid
    max_green_forced: int = 0  # decisions where "keep" was masked (forced switch)
    max_red_forced: int = 0
    max_red_violations: int = 0  # max_red could not be honoured because of min green


class PhaseConstraints:
    """Mask computation and bookkeeping for all signals of an environment."""

    def __init__(
        self,
        n_green: dict[str, int],
        n_actions: int,
        *,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        max_red: int | None = None,
    ):
        if max_green < min_green + delta_time:
            raise ValueError("max_green must be at least min_green + delta_time")
        if yellow_time >= delta_time:
            raise ValueError("yellow_time must be shorter than delta_time")
        self.n_green = n_green
        self.n_actions = n_actions
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.max_red = max_red
        self.state: dict[str, SignalConstraintState] = {}

    def reset(self, begin: float = 0.0) -> None:
        """Every signal starts in green phase 0 at ``begin`` (SUMO-RL's convention)."""
        self.state = {}
        for tls, n in self.n_green.items():
            st = SignalConstraintState(n_green=n, green_start=begin)
            st.red_start = {p: begin for p in range(1, n)}
            st.longest_red = {p: 0.0 for p in range(n)}
            self.state[tls] = st

    def _can_switch(self, tls: str, time_since_change: float) -> bool:
        # Same condition as TrafficSignal.set_next_phase in SUMO-RL.
        return time_since_change >= self.yellow_time + self.min_green

    def mask(self, tls: str, now: float, time_since_change: float) -> np.ndarray:
        st = self.state[tls]
        valid = np.zeros(self.n_actions, dtype=np.int8)
        current = st.green_phase
        if not self._can_switch(tls, time_since_change):
            valid[current] = 1
            return valid
        valid[: st.n_green] = 1
        if now + self.delta_time - st.green_start > self.max_green:
            valid[current] = 0
        if self.max_red is not None:
            urgent = [
                p
                for p, start in st.red_start.items()
                if now + self.delta_time + self.yellow_time - start > self.max_red
            ]
            if urgent:
                valid[:] = 0
                valid[urgent] = 1
        return valid

    def apply(self, tls: str, action: int, now: float, time_since_change: float) -> bool:
        """Check ``action`` against the mask, update bookkeeping; return True on a switch."""
        valid = self.mask(tls, now, time_since_change)
        if not 0 <= action < self.n_actions or not valid[action]:
            raise InvalidActionError(
                f"signal {tls}: action {action} is masked (valid: {np.flatnonzero(valid).tolist()})"
            )
        st = self.state[tls]
        st.decisions += 1
        if valid.sum() == 1 and valid[st.green_phase]:
            st.min_green_active += 1
        elif not valid[st.green_phase]:
            if now + self.delta_time - st.green_start > self.max_green:
                st.max_green_forced += 1
            else:
                st.max_red_forced += 1
        overdue = self.max_red is not None and any(
            now + self.delta_time - start > self.max_red for start in st.red_start.values()
        )
        if overdue and not self._can_switch(tls, time_since_change):
            st.max_red_violations += 1
        if action == st.green_phase:
            return False
        # Switch: yellow now, new green after the yellow.
        old, new = st.green_phase, action
        st.switches += 1
        st.red_start[old] = now + self.yellow_time
        red_begin = st.red_start.pop(new)
        st.longest_red[new] = max(st.longest_red[new], now + self.yellow_time - red_begin)
        st.green_phase = new
        st.green_start = now + self.yellow_time
        return True

    def longest_red(self, tls: str, now: float) -> dict[int, float]:
        """Longest red per phase so far, including reds still running at ``now``."""
        st = self.state[tls]
        result = dict(st.longest_red)
        for p, start in st.red_start.items():
            result[p] = max(result[p], now - start)
        return result

    def counters(self, tls: str) -> dict[str, int]:
        st = self.state[tls]
        return {
            "decisions": st.decisions,
            "switches": st.switches,
            "min_green_active": st.min_green_active,
            "max_green_forced": st.max_green_forced,
            "max_red_forced": st.max_red_forced,
            "max_red_violations": st.max_red_violations,
        }
