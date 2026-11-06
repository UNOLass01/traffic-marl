"""Phase-duration action: a continuous action becomes a phase through a green-time target.

The main action of the MDP chooses the next green phase. This alternative
(docs/mdp.md, used by MADDPG) only sets how long greens last: at each decision
the agent outputs ``a`` in [-1, 1], read as a target green duration between the
minimum and maximum green, and phases follow their fixed cyclic order. The
target is revised at every decision, so it is an adjustment of the running
green, not a commitment.

Constraints stay enforced by the same mask (D13): a forced "keep" or a forced
switch overrides the target, and with a maximum red the next phase in cyclic
order *among the valid ones* is taken, which is the overdue phase.
"""

from __future__ import annotations

import numpy as np


def target_green(action: float, min_green: float, max_green: float) -> float:
    """Map ``action`` in [-1, 1] linearly to a green duration in [min_green, max_green]."""
    fraction = (float(np.clip(action, -1.0, 1.0)) + 1.0) / 2.0
    return min_green + fraction * (max_green - min_green)


def duration_to_phase(
    action: float,
    *,
    current: int,
    mask: np.ndarray,
    green_age: float,
    min_green: float,
    max_green: float,
) -> int:
    """Phase to request now: keep ``current`` until its target is reached, then move on."""
    valid = np.flatnonzero(mask)
    if mask[current] and (
        len(valid) == 1 or green_age < target_green(action, min_green, max_green)
    ):
        return current
    n_actions = len(mask)
    for offset in range(1, n_actions + 1):
        phase = (current + offset) % n_actions
        if mask[phase]:
            return phase
    raise ValueError("the action mask allows no phase")


def to_phases(actions: np.ndarray, env, masks: np.ndarray) -> dict[str, int]:
    """Convert one continuous action per agent into the environment's phase actions."""
    greens = env.green_phases()
    return {
        agent: duration_to_phase(
            actions[i],
            current=greens[agent],
            mask=masks[i],
            green_age=env.green_age(agent),
            min_green=env.constraints.min_green,
            max_green=env.constraints.max_green,
        )
        for i, agent in enumerate(env.possible_agents)
    }
