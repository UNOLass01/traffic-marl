"""Max-pressure control (Varaiya, 2013) under the shared phase constraints.

At every decision each intersection picks, among the phases its action mask
allows, the one with the largest pressure

    P(p) = sum over movements (in lane -> out lane) green in p of (n_in - n_out),

where n_in counts the vehicles on the incoming lane's approach (the same
200 m measurement the agents observe) and n_out the vehicles on the outgoing
lane. Ties keep the current phase. Because it acts through the same mask, it
obeys exactly the same minimum/maximum green as the learning agents.

Reference: P. Varaiya, "Max pressure control of a network of signalized
intersections", Transportation Research Part C 36, 2013.
"""

from __future__ import annotations

import numpy as np


class MaxPressurePolicy:
    """Callable ``(observations, env) -> actions`` for :mod:`trafficmarl.metrics.evaluate`."""

    def __init__(self, env):
        self._links: dict[str, list[list[tuple[int, str]]]] = {}
        self._env = env

    def _phase_links(self, env, tls: str) -> list[list[tuple[int, str]]]:
        """Per green phase: (stop-lane index, out lane) of every green movement."""
        if tls not in self._links:
            ts = env.sumo_env.traffic_signals[tls]
            stop_index = {lane: i for i, lane in enumerate(env.spec.topology[tls].stop_lanes)}
            links = ts.sumo.trafficlight.getControlledLinks(tls)
            per_phase = []
            for phase in ts.green_phases:
                movements = []
                for k, signal in enumerate(phase.state):
                    if signal in "Gg" and k < len(links) and links[k]:
                        in_lane, out_lane, _ = links[k][0]
                        if in_lane in stop_index:
                            movements.append((stop_index[in_lane], out_lane))
                per_phase.append(movements)
            self._links[tls] = per_phase
        return self._links[tls]

    def __call__(self, observations: dict, env) -> dict[str, int]:
        sumo = env.sumo_env.sumo
        green = env.green_phases()
        actions = {}
        for tls, obs in observations.items():
            mask = obs["action_mask"]
            counts = env.meter.stats(sumo, tls).count
            out_cache: dict[str, int] = {}
            best, best_pressure = green[tls], -np.inf
            for phase, movements in enumerate(self._phase_links(env, tls)):
                if not mask[phase]:
                    continue
                pressure = 0.0
                for idx, out_lane in movements:
                    if out_lane not in out_cache:
                        out_cache[out_lane] = sumo.lane.getLastStepVehicleNumber(out_lane)
                    pressure += counts[idx] - out_cache[out_lane]
                if pressure > best_pressure or (pressure == best_pressure and phase == green[tls]):
                    best, best_pressure = phase, pressure
            actions[tls] = int(best)
        return actions


def make_max_pressure(spec: dict, env) -> MaxPressurePolicy:
    return MaxPressurePolicy(env)
