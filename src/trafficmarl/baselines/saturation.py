"""Empirical saturation flow and lost time per movement type (docs/decisions.md D31).

Webster's formula needs the saturation flow s (veh/h of green per lane) and the
lost time per phase. Instead of textbook values, both are measured in SUMO on
each scenario's own network, vehicle types and step length with a saturated
discharge test, one movement at a time in an otherwise empty network:

1. every traffic light shows red; a queue of vehicles that all make the tested
   movement (in edge -> out edge) is built on the approach and on its upstream
   edges (up to the previous traffic light), longer than what one green can
   discharge;
2. the tested movement gets ``green_s`` of green then the scenario's yellow;
3. the times at which vehicles cross the stop line give
   * the saturation flow: crossings per second between ``startup_s`` after the
     start of green and the end of green, divided by the number of lanes
     serving the movement;
   * the start-up lost time: ``startup_s`` minus the crossings before
     ``startup_s`` divided by the saturation rate;
   * the end gain: crossings during the yellow divided by the saturation rate.

The lost time per phase used by Webster is then start-up lost time + yellow -
end gain (there is no all-red in our programs). Movements are classified by
SUMO's connection direction: through (s), left (l, L) and right (r, R);
U-turns are not tested.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sumolib
import yaml

DIRECTION_TYPE = {"s": "through", "l": "left", "L": "left", "r": "right", "R": "right"}
TYPES = ("through", "left", "right")
SATURATION_FILE = "saturation.yaml"


@dataclass(frozen=True)
class Movement:
    tls: str
    in_edge: str
    out_edge: str
    kind: str  # through / left / right
    lanes: int  # incoming lanes serving the movement
    links: tuple[int, ...]  # signal indices of the movement
    upstream: tuple[str, ...]  # route edges before ``in_edge``


def movement_types(net: sumolib.net.Net) -> dict[str, dict[tuple[str, str], str]]:
    """Per traffic light: movement (in edge, out edge) -> through / left / right."""
    result: dict[str, dict[tuple[str, str], str]] = {}
    for tl in net.getTrafficLights():
        kinds: dict[tuple[str, str], str] = {}
        for in_lane, out_lane, _ in tl.getConnections():
            for conn in in_lane.getOutgoing():
                if conn.getToLane() == out_lane and conn.getDirection() in DIRECTION_TYPE:
                    move = (in_lane.getEdge().getID(), out_lane.getEdge().getID())
                    kinds[move] = DIRECTION_TYPE[conn.getDirection()]
        result[tl.getID()] = kinds
    return result


def _upstream(edge: sumolib.net.edge.Edge, min_length: float) -> list[str]:
    """Edges leading straight into ``edge`` up to ``min_length`` or the previous signal."""
    route: list[str] = []
    length = edge.getLength()
    current = edge
    while length < min_length:
        node = current.getFromNode()
        if node.getType().startswith("traffic_light"):
            break
        straight = [
            e
            for e, conns in current.getIncoming().items()
            if any(c.getDirection() == "s" for c in conns) and e.getFunction() != "internal"
        ]
        if not straight:
            break
        current = straight[0]
        if current.getID() in route:
            break
        route.insert(0, current.getID())
        length += current.getLength()
    return route


def candidate_movements(net: sumolib.net.Net, min_storage: float) -> list[Movement]:
    """Every testable movement with at least ``min_storage`` metres of approach."""
    result = []
    for tl in net.getTrafficLights():
        links: dict[tuple[str, str], set[int]] = {}
        lanes: dict[tuple[str, str], set[str]] = {}
        for in_lane, out_lane, idx in tl.getConnections():
            move = (in_lane.getEdge().getID(), out_lane.getEdge().getID())
            links.setdefault(move, set()).add(idx)
            lanes.setdefault(move, set()).add(in_lane.getID())
        kinds = movement_types(net)[tl.getID()]
        for move, kind in sorted(kinds.items()):
            in_edge = net.getEdge(move[0])
            upstream = _upstream(in_edge, min_storage)
            storage = in_edge.getLength() + sum(net.getEdge(e).getLength() for e in upstream)
            if storage < min_storage:
                continue
            result.append(
                Movement(tl.getID(), move[0], move[1], kind, len(lanes[move]),
                         tuple(sorted(links[move])), tuple(upstream))
            )  # fmt: skip
    return result


def _vehicle_types(route_file: Path) -> tuple[list[ET.Element], Counter]:
    """vType definitions of a route file and how often vehicles use each type."""
    opener = gzip.open if route_file.suffix == ".gz" else open
    definitions, usage = [], Counter()
    with opener(route_file, "rb") as fh:
        for _, elem in ET.iterparse(fh):
            if elem.tag in ("vType", "vTypeDistribution") and elem.get("id"):
                definitions.append(elem)
            elif elem.tag == "vehicle":
                usage[elem.get("type", "DEFAULT_VEHTYPE")] += 1
                elem.clear()
    return definitions, usage


@dataclass
class Discharge:
    rate: float  # vehicles per second over all serving lanes
    startup_crossings: int
    yellow_crossings: int
    saturated_to_end: bool  # queue still present when the green ended


def _run_tests(
    manifest, movements: list[Movement], cfg: dict[str, Any], yellow: int, seed: int
) -> list[dict[str, Any]]:
    """Simulate every discharge test in one SUMO run (one test after another)."""
    import traci  # libsumo when LIBSUMO_AS_TRACI=1

    route_file = manifest.route_files("medium", "train")[0]
    definitions, usage = _vehicle_types(route_file)
    with_vtypes = Path(cfg["work_dir"]) / "vtypes.add.xml"
    root = ET.Element("additional")
    root.extend(definitions)
    ET.ElementTree(root).write(with_vtypes, encoding="UTF-8")
    additional = [str(manifest.path(f)) for f in manifest.additional_files] + [str(with_vtypes)]
    traci.start([
        sumolib.checkBinary("sumo"), "-n", str(manifest.path(manifest.net_file)),
        "--additional-files", ",".join(additional), "--seed", str(seed),
        "--time-to-teleport", "-1", "--no-warnings", "--no-step-log",
    ])  # fmt: skip
    rng = np.random.default_rng(seed)
    types, weights = zip(*usage.items(), strict=True)
    probs = np.array(weights, dtype=float) / sum(weights)
    n_links = {t: len(traci.trafficlight.getControlledLinks(t)) for t in manifest.tls_ids}
    for tls, n in n_links.items():
        traci.trafficlight.setRedYellowGreenState(tls, "r" * n)

    rows = []
    for k, mv in enumerate(movements):
        route_id = f"test{k}"
        traci.route.add(route_id, [*mv.upstream, mv.in_edge, mv.out_edge])
        out_len = traci.lane.getLength(f"{mv.out_edge}_0")
        vehicles = [f"{route_id}.{i}" for i in range(cfg["vehicles_per_lane"] * mv.lanes)]
        for vid in vehicles:
            traci.vehicle.add(
                vid, route_id, typeID=str(rng.choice(types, p=probs)), depart="now",
                departLane="best", departPos="base", departSpeed="max",
                arrivalPos=f"{min(out_len - 0.1, 20.0):.1f}",
            )  # fmt: skip
        for _ in range(cfg["buildup_s"]):
            traci.simulationStep()
        result = _discharge(traci, mv, n_links[mv.tls], vehicles, cfg, yellow)
        traci.trafficlight.setRedYellowGreenState(mv.tls, "r" * n_links[mv.tls])
        # Drop what is left of the queue, including vehicles still waiting to be inserted,
        # so the next test starts from an empty network.
        left = set(traci.vehicle.getIDList()) | set(traci.simulation.getPendingVehicles())
        for vid in set(vehicles) & left:
            traci.vehicle.remove(vid)
        for _ in range(cfg["clearance_s"]):
            traci.simulationStep()
        rows.append({"tls": mv.tls, "in_edge": mv.in_edge, "out_edge": mv.out_edge,
                     "type": mv.kind, "lanes": mv.lanes, **result.__dict__})  # fmt: skip
    traci.close()
    return rows


def _discharge(traci, mv: Movement, n_links: int, vehicles: list[str], cfg, yellow: int):
    """Give the movement green then yellow and record stop-line crossing times."""

    def state(char: str) -> str:
        return "".join(char if i in mv.links else "r" for i in range(n_links))

    in_length = traci.lane.getLength(f"{mv.in_edge}_0")
    tested = set(vehicles)
    last_seen: dict[str, tuple[float, float, float]] = {}  # vid -> (time, pos, speed)
    crossings: list[float] = []
    traci.trafficlight.setRedYellowGreenState(mv.tls, state("G"))
    start = traci.simulation.getTime()
    green_end = start + cfg["green_s"]
    queue_left_at_end = False
    while traci.simulation.getTime() < green_end + yellow:
        now = traci.simulation.getTime()
        if now >= green_end and not queue_left_at_end:
            queue_left_at_end = True
            traci.trafficlight.setRedYellowGreenState(mv.tls, state("y"))
        on_edge = set(traci.edge.getLastStepVehicleIDs(mv.in_edge)) & tested
        for vid, (t, pos, speed) in list(last_seen.items()):
            if vid not in on_edge:
                # Interpolate the crossing time within the last step.
                dt = (in_length - pos) / speed if speed > 0.1 else 1.0
                crossings.append(t + min(max(dt, 0.0), 1.0))
                del last_seen[vid]
        for vid in on_edge:
            last_seen[vid] = (now, traci.vehicle.getLanePosition(vid), traci.vehicle.getSpeed(vid))
        traci.simulationStep()

    crossed = np.array(sorted(crossings)) - start
    waiting = len(tested) - len(crossed)
    window = (crossed >= cfg["startup_s"]) & (crossed < cfg["green_s"])
    rate = window.sum() / (cfg["green_s"] - cfg["startup_s"])
    return Discharge(
        rate=float(rate),
        startup_crossings=int((crossed < cfg["startup_s"]).sum()),
        yellow_crossings=int((crossed >= cfg["green_s"]).sum()),
        saturated_to_end=bool(waiting > 0),
    )


def summarize_tests(rows: list[dict[str, Any]], cfg: dict[str, Any], yellow: int) -> dict:
    """Per movement type: saturation flow (veh/h/lane); per scenario: lost time per phase."""
    valid = [r for r in rows if r["saturated_to_end"] and r["rate"] > 0]
    result: dict[str, Any] = {"tests": len(rows), "valid_tests": len(valid), "by_type": {}}
    startup_lost, end_gain = [], []
    for kind in TYPES:
        per_lane = [3600 * r["rate"] / r["lanes"] for r in valid if r["type"] == kind]
        if per_lane:
            result["by_type"][kind] = {
                "saturation_flow": round(float(np.mean(per_lane)), 1),
                "std": round(float(np.std(per_lane, ddof=1)), 1) if len(per_lane) > 1 else 0.0,
                "n": len(per_lane),
            }
    for r in valid:
        startup_lost.append(cfg["startup_s"] - r["startup_crossings"] / r["rate"])
        end_gain.append(r["yellow_crossings"] / r["rate"])
    result["startup_lost_s"] = round(float(np.mean(startup_lost)), 2)
    result["end_gain_s"] = round(float(np.mean(end_gain)), 2)
    result["yellow_s"] = yellow
    result["lost_time_per_phase_s"] = round(
        result["startup_lost_s"] + yellow - result["end_gain_s"], 2
    )
    return result


def measure(manifest, yellow: int, cfg: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Run the discharge tests on a scenario; write ``<scenario>/webster/saturation.yaml``."""
    import tempfile

    net = sumolib.net.readNet(str(manifest.path(manifest.net_file)), withPrograms=True)
    movements = candidate_movements(net, cfg["min_storage_m"])
    rng = np.random.default_rng(seed)
    sample = []
    for kind in TYPES:
        of_kind = [m for m in movements if m.kind == kind]
        take = min(len(of_kind), cfg["max_tests_per_type"])
        sample += [of_kind[i] for i in sorted(rng.choice(len(of_kind), take, replace=False))]
    with tempfile.TemporaryDirectory() as tmp:
        rows = _run_tests(manifest, sample, {**cfg, "work_dir": tmp}, yellow, seed)
    summary = summarize_tests(rows, cfg, yellow)
    summary["candidate_movements"] = len(movements)
    out = manifest.root / "webster" / SATURATION_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"config": cfg, "summary": summary, "tests": rows}, fh, sort_keys=False)
    return summary


def load_measured(manifest) -> dict[str, Any]:
    path = manifest.root / "webster" / SATURATION_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `python -m trafficmarl baselines saturation`")
    return yaml.safe_load(path.read_text(encoding="utf-8"))["summary"]
