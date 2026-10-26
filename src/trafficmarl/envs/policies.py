"""Non-learning policies over the action mask, used for rollouts, sanity checks and tests.

* ``random``: uniform over the valid phases;
* ``keep``: keep the current phase whenever allowed (exercises the maximum green);
* ``switch``: move to the next valid phase whenever allowed (exercises the minimum green).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Policy = Callable[[dict[str, np.ndarray], dict[str, int]], dict[str, int]]
POLICIES = ("random", "keep", "switch")


def _next_valid(mask: np.ndarray, current: int) -> int:
    """First valid phase after ``current`` in cyclic order (``current`` itself if alone)."""
    n = len(mask)
    for step in range(1, n + 1):
        candidate = (current + step) % n
        if mask[candidate] and candidate != current:
            return candidate
    return current


def make_policy(name: str, seed: int = 0) -> Callable[[dict, dict[str, int]], dict[str, int]]:
    """Policy taking (observations, current green phase per agent) and returning actions."""
    rng = np.random.default_rng(seed)

    def act(observations: dict, current: dict[str, int]) -> dict[str, int]:
        actions = {}
        for agent, obs in observations.items():
            mask = obs["action_mask"]
            if name == "random":
                actions[agent] = int(rng.choice(np.flatnonzero(mask)))
            elif name == "keep":
                actions[agent] = (
                    current[agent] if mask[current[agent]] else _next_valid(mask, current[agent])
                )
            elif name == "switch":
                actions[agent] = _next_valid(mask, current[agent])
            else:
                raise ValueError(f"Unknown policy '{name}' (expected one of {POLICIES})")
        return actions

    return act
