"""Observation vectors and per-agent reward terms (layout documented in docs/mdp.md).

Local observation of one intersection (all values in [0, 1]):

    queue[L] | density[L] | speed[L] | phase one-hot[P] | time since change | min-green flag
    | n-green-phases one-hot[P] | n-lanes / L | (optional) agent one-hot[N]

``L`` and ``P`` are the largest lane and green-phase counts of the scenario;
smaller intersections are zero-padded. Queue and density are divided by the
number of vehicles that fit in ``max_distance`` (7.5 m per vehicle), speed by
the lane's speed limit; an empty lane has speed 1 (free flow) while a padded
lane has 0. The extended observation appends four neighbour slots (N, E, S, W),
each holding ``mean queue, mean density, phase one-hot[P], present flag``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficmarl.envs.measure import VEHICLE_SPACING, ApproachMeter, ApproachStats
from trafficmarl.envs.topology import SLOTS, SignalTopology

REWARD_MEASURES = ("waiting", "queue", "pressure")


@dataclass
class FeatureSpec:
    """Dimensions and options shared by all agents of one environment."""

    agents: list[str]
    topology: dict[str, SignalTopology]
    n_green: dict[str, int]
    max_lanes: int
    max_phases: int
    max_distance: float
    max_green: int
    extended: bool
    agent_id: bool

    @property
    def capacity(self) -> float:
        """Vehicles that fit in ``max_distance`` on one lane."""
        return self.max_distance / VEHICLE_SPACING

    @property
    def local_dim(self) -> int:
        dim = 3 * self.max_lanes + self.max_phases + 2 + self.max_phases + 1
        return dim + (len(self.agents) if self.agent_id else 0)

    @property
    def message_dim(self) -> int:
        """Floats one neighbour sends: mean queue, mean density, phase one-hot."""
        return 2 + self.max_phases

    @property
    def obs_dim(self) -> int:
        return self.local_dim + (len(SLOTS) * (self.message_dim + 1) if self.extended else 0)


def _pad(values: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[: len(values)] = values
    return out


def lane_features(
    spec: FeatureSpec, stats: ApproachStats
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queue = np.clip(stats.queue / spec.capacity, 0.0, 1.0)
    density = np.clip(stats.count / spec.capacity, 0.0, 1.0)
    speed = np.where(np.isnan(stats.speed), 1.0, np.clip(stats.speed / stats.max_speed, 0.0, 1.0))
    return queue, density, speed


def phase_one_hot(spec: FeatureSpec, phase: int) -> np.ndarray:
    one_hot = np.zeros(spec.max_phases, dtype=np.float32)
    one_hot[phase] = 1.0
    return one_hot


def message(spec: FeatureSpec, stats: ApproachStats, phase: int) -> np.ndarray:
    """What an intersection shares with its neighbours (``message_dim`` floats)."""
    queue, density, _ = lane_features(spec, stats)
    return np.concatenate([[queue.mean(), density.mean()], phase_one_hot(spec, phase)]).astype(
        np.float32
    )


def observe(
    spec: FeatureSpec,
    meter: ApproachMeter,
    sumo,
    tls: str,
    green_phase: dict[str, int],
    time_since_change: float,
    keep_only: bool,
) -> np.ndarray:
    """Observation vector of ``tls`` (see module docstring for the layout)."""
    topo = spec.topology[tls]
    queue, density, speed = lane_features(spec, meter.stats(sumo, tls))
    n_lanes = len(topo.stop_lanes)
    parts = [
        _pad(queue, spec.max_lanes),
        _pad(density, spec.max_lanes),
        _pad(speed, spec.max_lanes),
        phase_one_hot(spec, green_phase[tls]),
        [min(time_since_change / spec.max_green, 1.0), float(keep_only)],
        phase_one_hot(spec, spec.n_green[tls] - 1),
        [n_lanes / spec.max_lanes],
    ]
    if spec.agent_id:
        parts.append(phase_one_hot_n(len(spec.agents), spec.agents.index(tls)))
    if spec.extended:
        for slot in SLOTS:
            members = topo.slots[slot]
            if not members:
                parts.append(np.zeros(spec.message_dim + 1, dtype=np.float32))
                continue
            # Several neighbours in one compass slot are averaged (docs/mdp.md).
            msgs = [message(spec, meter.stats(sumo, n), green_phase[n]) for n in members]
            parts.append(np.concatenate([np.mean(msgs, axis=0), [1.0]]))
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])


def phase_one_hot_n(size: int, index: int) -> np.ndarray:
    one_hot = np.zeros(size, dtype=np.float32)
    one_hot[index] = 1.0
    return one_hot


def traffic_term(
    spec: FeatureSpec, stats: ApproachStats, measure: str, scale: float, previous: float
) -> tuple[float, float]:
    """Local traffic reward of one intersection and the new reference value.

    * ``waiting``: minus the change, over one decision step, of the accumulated
      waiting time of vehicles on the approach, divided by ``scale`` seconds.
      Vehicles that leave the approach take their waiting time with them, so
      clearing a queue is rewarded (SUMO-RL's ``diff-waiting-time`` idea, on
      the full approach).
    * ``queue``: minus the halting vehicles on the approach / lane capacity.
    * ``pressure``: minus |vehicles approaching - vehicles leaving| / capacity.
    """
    if measure == "waiting":
        return -(stats.waiting - previous) / scale, stats.waiting
    if measure == "queue":
        return -stats.halting_total / spec.capacity, previous
    if measure == "pressure":
        return -abs(float(stats.count.sum()) - stats.outgoing) / spec.capacity, previous
    raise ValueError(f"Unknown reward measure '{measure}' (expected one of {REWARD_MEASURES})")


def mix_rewards(
    traffic: dict[str, float],
    penalty: dict[str, float],
    kind: str,
    neighbours: dict[str, list[str]],
    alpha: float,
) -> dict[str, float]:
    """Combine traffic terms according to the cooperation scheme; penalties stay individual.

    * ``local``: r_i = t_i - p_i
    * ``neighborhood``: r_i = t_i + alpha * mean_{j in N(i)} t_j - p_i
    * ``global``: r_i = mean_j t_j - p_i

    The switch penalty p_i is not shared: it is the cost of agent i's own
    decision, and sharing it would blur the credit for switching.
    """
    if kind == "local":
        return {a: traffic[a] - penalty[a] for a in traffic}
    if kind == "neighborhood":
        out = {}
        for a in traffic:
            nbrs = [traffic[n] for n in neighbours[a] if n in traffic]
            out[a] = traffic[a] + (alpha * float(np.mean(nbrs)) if nbrs else 0.0) - penalty[a]
        return out
    if kind == "global":
        shared = float(np.mean(list(traffic.values())))
        return {a: shared - penalty[a] for a in traffic}
    raise ValueError(f"Unknown reward type '{kind}' (expected local, neighborhood or global)")
