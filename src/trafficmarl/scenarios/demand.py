"""Seeded traffic demand for the synthetic networks.

Vehicles enter at the network fringe following a Poisson process whose rate may
vary over time, and are routed with fixed turning probabilities at every
intersection (a random walk until they leave the network).

Why turning probabilities instead of uniform random origin-destination pairs:
with uniform ODs on a 4-arm junction one third of the traffic turns left, which
saturates the short left-turn lanes long before the through lanes and makes
every controller look bad for the same uninteresting reason. Turning ratios are
also the usual way demand is specified in the traffic-signal-control literature.

All randomness comes from one ``numpy.random.Generator``, so a seed fully
determines the route file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sumolib

from trafficmarl.utils.paths import open_output

# SUMO connection directions mapped to the three turn classes used in configs;
# 't' (turnaround) is never chosen.
DIRECTION_CLASS = {"s": "straight", "r": "right", "R": "right", "l": "left", "L": "left"}
MAX_ROUTE_EDGES = 500


@dataclass(frozen=True)
class Vehicle:
    vid: str
    depart: float
    edges: tuple[str, ...]


def rate_multiplier(profile: Sequence[float], t: float, duration: float) -> float:
    """Demand multiplier at time ``t``.

    ``profile`` lists multipliers at equally spaced knots from ``0`` to
    ``duration``, linearly interpolated in between; a single value means
    constant demand. E.g. ``[0.6, 1.6, 0.6]`` ramps up to a peak at mid-episode.
    """
    if len(profile) == 0:
        raise ValueError("Demand profile must contain at least one multiplier")
    if len(profile) == 1:
        return float(profile[0])
    knots = np.linspace(0.0, duration, num=len(profile))
    return float(np.interp(t, knots, profile))


def poisson_departures(
    rate_per_hour: float,
    profile: Sequence[float],
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Departure times of a non-homogeneous Poisson process on ``[0, duration)``.

    Uses thinning (Lewis & Shedler, 1979): candidates are drawn at the maximum
    rate and each is kept with probability ``rate(t) / max_rate``.
    """
    if rate_per_hour <= 0:
        return np.empty(0)
    peak = rate_per_hour * max(profile) / 3600.0
    n_candidates = rng.poisson(peak * duration)
    candidates = np.sort(rng.uniform(0.0, duration, size=n_candidates))
    accept_prob = np.array(
        [rate_per_hour * rate_multiplier(profile, t, duration) / 3600.0 / peak for t in candidates]
    )
    keep = rng.uniform(size=n_candidates) < accept_prob
    return candidates[keep]


def entry_and_exit_edges(net: sumolib.net.Net) -> tuple[list[str], set[str]]:
    """Edges starting at / ending in a dead-end fringe node (sorted for determinism)."""
    entries = sorted(e.getID() for e in net.getEdges() if e.getFromNode().getType() == "dead_end")
    exits = {e.getID() for e in net.getEdges() if e.getToNode().getType() == "dead_end"}
    if not entries or not exits:
        raise ValueError("Network has no dead-end fringe nodes to insert/remove vehicles")
    return entries, exits


def _successors_by_class(edge: sumolib.net.edge.Edge) -> dict[str, list[str]]:
    by_class: dict[str, set[str]] = {}
    for to_edge, connections in edge.getOutgoing().items():
        for conn in connections:
            turn = DIRECTION_CLASS.get(conn.getDirection())
            if turn is not None:
                by_class.setdefault(turn, set()).add(to_edge.getID())
    return {k: sorted(v) for k, v in by_class.items()}


def random_walk_route(
    net: sumolib.net.Net,
    entry: str,
    exits: set[str],
    turn_probabilities: Mapping[str, float],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Follow turning probabilities from ``entry`` until an exit edge is reached.

    Probabilities are renormalised over the turn classes available at each
    edge, so a T-junction or a straight-only segment needs no special case.
    """
    route = [entry]
    current = net.getEdge(entry)
    while route[-1] not in exits:
        options = _successors_by_class(current)
        classes = [c for c in sorted(options) if turn_probabilities.get(c, 0.0) > 0]
        if not classes:
            raise ValueError(f"Dead end at edge '{current.getID()}' (route {route})")
        weights = np.array([turn_probabilities[c] for c in classes], dtype=float)
        turn = classes[rng.choice(len(classes), p=weights / weights.sum())]
        targets = options[turn]
        nxt = targets[rng.integers(len(targets))]
        route.append(nxt)
        current = net.getEdge(nxt)
        if len(route) > MAX_ROUTE_EDGES:
            raise ValueError(f"Route from '{entry}' did not reach an exit (loop?)")
    return tuple(route)


def entry_weight(entry: str, weights: Mapping[str, float]) -> float:
    """Multiplier for an entry edge: the longest matching id prefix in ``weights`` wins."""
    matches = [prefix for prefix in weights if entry.startswith(prefix)]
    return float(weights[max(matches, key=len)]) if matches else 1.0


def generate_vehicles(
    net: sumolib.net.Net,
    *,
    entry_rate: float,
    profile: Sequence[float],
    duration: float,
    turn_probabilities: Mapping[str, float],
    seed: int,
    entry_weights: Mapping[str, float] | None = None,
) -> list[Vehicle]:
    """All vehicles of one demand realisation, sorted by departure time."""
    rng = np.random.default_rng(seed)
    entries, exits = entry_and_exit_edges(net)
    weights = entry_weights or {}
    pending: list[tuple[float, str, tuple[str, ...]]] = []
    for entry in entries:
        rate = entry_rate * entry_weight(entry, weights)
        for depart in poisson_departures(rate, profile, duration, rng):
            route = random_walk_route(net, entry, exits, turn_probabilities, rng)
            pending.append((round(float(depart), 2), entry, route))
    pending.sort(key=lambda v: (v[0], v[1]))
    return [Vehicle(f"veh{i}", depart, route) for i, (depart, _, route) in enumerate(pending)]


VTYPE_XML = (
    '    <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" '
    'length="5.0" minGap="2.5" maxSpeed="13.89"/>\n'
)


def write_route_file(vehicles: Sequence[Vehicle], path: str | Path, comment: str = "") -> Path:
    """Write vehicles with explicit routes to a SUMO ``.rou.xml`` (or ``.rou.xml.gz``) file."""
    path = Path(path)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>\n']
    if comment:
        lines.append(f"<!-- {comment} -->\n")
    lines.append("<routes>\n")
    lines.append(VTYPE_XML)
    for v in vehicles:
        lines.append(
            f'    <vehicle id="{v.vid}" type="car" depart="{v.depart:.2f}" '
            f'departLane="best" departSpeed="max">\n'
            f'        <route edges="{" ".join(v.edges)}"/>\n'
            "    </vehicle>\n"
        )
    lines.append("</routes>\n")
    with open_output(path) as fh:
        fh.write("".join(lines).encode("utf-8"))
    return path
