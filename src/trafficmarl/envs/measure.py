"""Per-approach traffic measurements read from TraCI/libsumo.

Observations and rewards are both built from these numbers, measured once per
decision step and cached, so the two always describe the same traffic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficmarl.envs.topology import SignalTopology

HALTING_SPEED = 0.1  # m/s, SUMO's definition of a halting vehicle
VEHICLE_SPACING = 7.5  # m per queued car (5 m length + 2.5 m gap), sets the capacity


@dataclass
class ApproachStats:
    """Measurements of one traffic light at one simulation time."""

    queue: np.ndarray  # per stop lane, weighted halting vehicles
    count: np.ndarray  # per stop lane, weighted vehicles
    speed: np.ndarray  # per stop lane, mean speed (m/s), NaN if empty
    max_speed: np.ndarray  # per stop lane, speed limit (m/s)
    waiting: float  # accumulated waiting time of vehicles on the approach (s)
    halting_total: float  # halting vehicles on the approach (no double counting)
    outgoing: float  # vehicles on the outgoing lanes


class ApproachMeter:
    """Measures every traffic light's approach; results are cached per simulation time."""

    def __init__(self, topology: dict[str, SignalTopology]):
        self.topology = topology
        self._cache: dict[str, ApproachStats] = {}
        self._cache_time: float | None = None
        self._max_speed: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._cache.clear()
        self._cache_time = None

    def stats(self, sumo, tls_id: str) -> ApproachStats:
        now = sumo.simulation.getTime()
        if now != self._cache_time:
            self._cache.clear()
            self._cache_time = now
        if tls_id not in self._cache:
            self._cache[tls_id] = self._measure(sumo, self.topology[tls_id])
        return self._cache[tls_id]

    def _vehicles_in(self, sumo, lane: str, start: float, end: float) -> list[tuple[float, float]]:
        """(speed, accumulated waiting time) of the vehicles on ``lane`` in [start, end]."""
        result = []
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            if start <= sumo.vehicle.getLanePosition(vid) <= end:
                result.append(
                    (sumo.vehicle.getSpeed(vid), sumo.vehicle.getAccumulatedWaitingTime(vid))
                )
        return result

    def _measure(self, sumo, topo: SignalTopology) -> ApproachStats:
        n = len(topo.stop_lanes)
        queue, count, speed_sum = np.zeros(n), np.zeros(n), np.zeros(n)
        seen: dict[tuple[str, float, float], list[tuple[float, float]]] = {}
        for i, stop in enumerate(topo.stop_lanes):
            for seg in topo.approach[stop]:
                key = (seg.lane, seg.start, seg.end)
                if key not in seen:
                    seen[key] = self._vehicles_in(sumo, seg.lane, seg.start, seg.end)
                vehicles = seen[key]
                count[i] += seg.weight * len(vehicles)
                queue[i] += seg.weight * sum(v < HALTING_SPEED for v, _ in vehicles)
                speed_sum[i] += seg.weight * sum(v for v, _ in vehicles)
        all_vehicles = [veh for vehicles in seen.values() for veh in vehicles]
        if topo.tls_id not in self._max_speed:
            self._max_speed[topo.tls_id] = np.array(
                [sumo.lane.getMaxSpeed(lane) for lane in topo.stop_lanes]
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            speed = np.where(count > 0, speed_sum / count, np.nan)
        return ApproachStats(
            queue=queue,
            count=count,
            speed=speed,
            max_speed=self._max_speed[topo.tls_id],
            waiting=float(sum(w for _, w in all_vehicles)),
            halting_total=float(sum(v < HALTING_SPEED for v, _ in all_vehicles)),
            outgoing=float(
                sum(sumo.lane.getLastStepVehicleNumber(lane) for lane in topo.out_lanes)
            ),
        )
