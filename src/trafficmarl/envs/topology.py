"""Static structure of a network, computed once per scenario from its ``.net.xml``.

* **Approaches.** For every stop-line lane of a traffic light we collect the lane
  segments upstream of the stop line, across consecutive edges, up to
  ``max_distance`` metres. Tracing stops at another signalised junction and at
  the network border, so an observation never reaches into the neighbouring
  intersection. An upstream lane that feeds k stop-line lanes (e.g. the
  segment before a left-turn bay) contributes with weight 1/k to each of
  them, so per-lane counts add up to the true number of vehicles.
* **Neighbour graph.** Two traffic lights are neighbours when a road path links
  them without crossing a third traffic light (undirected).
* **Compass slots.** The extended observation has four neighbour slots (N, E, S,
  W) chosen by the bearing from the intersection to the neighbour. Several
  neighbours in the same slot are averaged; how often this happens is part of
  the topology summary.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import sumolib

SLOTS = ("N", "E", "S", "W")


@dataclass(frozen=True)
class Segment:
    """Part of a lane, from ``start`` to ``end`` (m from the lane's beginning)."""

    lane: str
    start: float
    end: float
    weight: float  # share of this segment attributed to the stop-line lane

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class SignalTopology:
    tls_id: str
    stop_lanes: list[str]  # controlled incoming lanes, in SUMO-RL's order
    approach: dict[str, list[Segment]]  # stop lane -> upstream segments (stop lane included)
    approach_length: dict[str, float]  # stop lane -> traced length (<= max_distance)
    out_lanes: list[str]
    centroid: tuple[float, float]
    neighbours: list[str] = field(default_factory=list)
    slots: dict[str, list[str]] = field(default_factory=dict)  # slot -> neighbour ids

    @property
    def all_segments(self) -> list[Segment]:
        """Union of approach segments without double counting (one entry per lane part)."""
        merged: dict[tuple[str, float, float], Segment] = {}
        for segments in self.approach.values():
            for seg in segments:
                merged[(seg.lane, seg.start, seg.end)] = Segment(seg.lane, seg.start, seg.end, 1.0)
        return list(merged.values())


def _tls_nodes(net: sumolib.net.Net) -> dict[str, str]:
    """Junction id -> traffic-light id, for every signalised junction."""
    nodes = {}
    for tl in net.getTrafficLights():
        for conn in tl.getConnections():
            nodes[conn[0].getEdge().getToNode().getID()] = tl.getID()
    return nodes


def _controlled_lanes(tl: sumolib.net.TLS) -> list[str]:
    """Incoming lanes in link-index order, deduplicated (same order as SUMO-RL's ``ts.lanes``)."""
    ordered = sorted(tl.getConnections(), key=lambda c: c[2])
    return list(dict.fromkeys(c[0].getID() for c in ordered))


def _feeders(lane: sumolib.net.lane.Lane) -> list[sumolib.net.lane.Lane]:
    """Non-internal lanes with a connection into ``lane``."""
    return [inc for inc in lane.getIncoming() if not inc.getID().startswith(":")]


def trace_approach(
    net: sumolib.net.Net, stop_lane: str, max_distance: float, tls_nodes: dict[str, str]
) -> list[tuple[str, float, float]]:
    """Upstream lane parts within ``max_distance`` of the stop line of ``stop_lane``.

    Breadth-first over feeding lanes; a lane whose upstream junction is
    signalised is still included, but tracing does not continue past it.
    """
    lane = net.getLane(stop_lane)
    parts: dict[str, tuple[float, float]] = {}
    queue = deque([(lane, max_distance)])
    while queue:
        current, remaining = queue.popleft()
        length = current.getLength()
        take = min(length, remaining)
        start = length - take
        lane_id = current.getID()
        if lane_id in parts and parts[lane_id][0] <= start:
            continue
        parts[lane_id] = (start, length)
        left = remaining - take
        upstream_node = current.getEdge().getFromNode().getID()
        if left <= 0 or upstream_node in tls_nodes:
            continue
        for feeder in _feeders(current):
            queue.append((feeder, left))
    return [(lane_id, s, e) for lane_id, (s, e) in parts.items()]


def bearing_slot(origin: tuple[float, float], target: tuple[float, float]) -> str:
    """Compass slot of ``target`` seen from ``origin`` (N = +y), sectors of 90 degrees."""
    angle = math.degrees(math.atan2(target[0] - origin[0], target[1] - origin[1])) % 360.0
    return SLOTS[int(((angle + 45.0) % 360.0) // 90.0)]


def _neighbour_graph(net: sumolib.net.Net, tls_nodes: dict[str, str]) -> dict[str, set[str]]:
    """Traffic lights reachable by road without crossing another traffic light."""
    graph: dict[str, set[str]] = {tl: set() for tl in set(tls_nodes.values())}
    for node_id, tl in tls_nodes.items():
        seen: set[str] = set()
        queue = deque(e for e in net.getNode(node_id).getOutgoing())
        while queue:
            edge = queue.popleft()
            if edge.getID() in seen:
                continue
            seen.add(edge.getID())
            to_node = edge.getToNode().getID()
            other = tls_nodes.get(to_node)
            if other is not None:
                if other != tl:
                    graph[tl].add(other)
                    graph[other].add(tl)
                continue
            queue.extend(edge.getToNode().getOutgoing())
    return graph


def build_topology(net_file: str, max_distance: float) -> dict[str, SignalTopology]:
    """Topology of every traffic light in ``net_file``."""
    net = sumolib.net.readNet(net_file, withPrograms=True, withInternal=False)
    tls_nodes = _tls_nodes(net)
    result: dict[str, SignalTopology] = {}
    for tl in net.getTrafficLights():
        stop_lanes = _controlled_lanes(tl)
        traced = {lane: trace_approach(net, lane, max_distance, tls_nodes) for lane in stop_lanes}
        # How many stop lanes each upstream lane part feeds -> weights 1/k.
        usage: dict[str, int] = {}
        for parts in traced.values():
            for lane_id, _, _ in parts:
                usage[lane_id] = usage.get(lane_id, 0) + 1
        approach = {
            stop: [Segment(lane_id, s, e, 1.0 / usage[lane_id]) for lane_id, s, e in parts]
            for stop, parts in traced.items()
        }
        lengths = {
            stop: min(max_distance, _longest_path(net, stop, parts))
            for stop, parts in traced.items()
        }
        nodes = {c[0].getEdge().getToNode() for c in tl.getConnections()}
        cx = sum(n.getCoord()[0] for n in nodes) / len(nodes)
        cy = sum(n.getCoord()[1] for n in nodes) / len(nodes)
        out_lanes = list(dict.fromkeys(c[1].getID() for c in tl.getConnections()))
        result[tl.getID()] = SignalTopology(
            tl.getID(), stop_lanes, approach, lengths, out_lanes, (cx, cy)
        )

    graph = _neighbour_graph(net, tls_nodes)
    for tl_id, topo in result.items():
        topo.neighbours = sorted(graph[tl_id])
        topo.slots = {slot: [] for slot in SLOTS}
        for other in topo.neighbours:
            topo.slots[bearing_slot(topo.centroid, result[other].centroid)].append(other)
    return result


def _longest_path(
    net: sumolib.net.Net, stop_lane: str, parts: list[tuple[str, float, float]]
) -> float:
    """Traced distance upstream of the stop line (longest chain of the traced parts)."""
    lengths = {lane_id: e - s for lane_id, s, e in parts}

    def depth(lane_id: str, visited: frozenset[str]) -> float:
        lane = net.getLane(lane_id)
        best = 0.0
        for feeder in _feeders(lane):
            fid = feeder.getID()
            if fid in lengths and fid not in visited:
                best = max(best, depth(fid, visited | {fid}))
        return lengths[lane_id] + best

    return depth(stop_lane, frozenset({stop_lane}))


def summarize(
    topology: dict[str, SignalTopology], n_green: dict[str, int], floats_per_message: int
) -> dict:
    """Facts reported for each scenario (approach coverage, neighbour slots, message volume)."""
    lanes = [len(t.stop_lanes) for t in topology.values()]
    lengths = [v for t in topology.values() for v in t.approach_length.values()]
    neighbours = [len(t.neighbours) for t in topology.values()]
    shared_slots = sum(1 for t in topology.values() for ids in t.slots.values() if len(ids) > 1)
    agents_with_shared = sum(
        1 for t in topology.values() if any(len(i) > 1 for i in t.slots.values())
    )
    messages = sum(neighbours)
    return {
        "agents": len(topology),
        "green_phases": dict(
            sorted({k: list(n_green.values()).count(k) for k in set(n_green.values())}.items())
        ),
        "stop_lanes_min_max": [min(lanes), max(lanes)],
        "approach_length_mean_m": round(sum(lengths) / len(lengths), 1),
        "approach_length_min_m": round(min(lengths), 1),
        "neighbours_min_mean_max": [
            min(neighbours),
            round(sum(neighbours) / len(neighbours), 2),
            max(neighbours),
        ],
        "agents_with_more_than_4_neighbours": sum(n > 4 for n in neighbours),
        "slots_with_several_neighbours": shared_slots,
        "agents_with_aggregated_slot": agents_with_shared,
        "messages_per_decision_step": messages,
        "floats_per_decision_step": messages * floats_per_message,
    }
