"""Webster fixed-time plans computed from the training demand (docs/decisions.md D28).

For each traffic light and demand profile:

1. **Flows.** Every vehicle of the *training* route files that crosses the
   junction adds one to its movement (incoming edge -> outgoing edge). A
   movement's flow is shared equally among the incoming lanes that serve it,
   giving a flow per incoming lane. Rates are hourly, averaged over the
   training files. For the non-stationary ``peak`` profile the plan uses the
   busiest 15-minute window at that junction (by departure time), scaled to an
   hourly rate.
2. **Critical flow ratios.** Each movement's flow is divided by the saturation
   flow of its type (through, left, right), measured in SUMO on the scenario's
   own network (:mod:`trafficmarl.baselines.saturation`, D31), so a lane's
   flow ratio is sum_m q_m / s_type(m). In practice flows are converted to
   through-vehicle equivalents (q_m x s_through / s_type(m)) and divided by
   s_through. For each green phase p, y_p is the largest lane flow ratio among
   the lanes with a protected ('G') movement in p; Y = sum of y_p.
3. **Cycle.** Webster's optimum C0 = (1.5 L + 5) / (1 - Y), where L is the lost
   time per cycle: one lost time per green phase, measured (start-up lost time
   + yellow - end gain, D31). If Y >= 1 or C0 exceeds the cap (default 150 s),
   the capped cycle is used.
4. **Greens.** The cycle minus all non-green phase time is split among green
   phases in proportion to y_p, then clipped to [min green, max green], the
   constraints shared by every method (D13).

Reference: F. V. Webster, "Traffic Signal Settings", Road Research Technical
Paper 39, HMSO, 1958.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sumolib
import yaml

from trafficmarl.baselines.saturation import TYPES, load_measured, movement_types
from trafficmarl.scenarios.tls import is_green_phase

PROGRAM_ID = "webster"
WEBSTER_DIR = "webster"


@dataclass
class SignalPlan:
    tls_id: str
    cycle: float
    flow_ratio_sum: float  # Y
    capped: bool
    greens: list[float]  # one per green phase, in program order
    critical_ratios: list[float] = field(default_factory=list)


def _iter_vehicles(route_file: Path):
    opener = gzip.open if route_file.suffix == ".gz" else open
    with opener(route_file, "rb") as fh:
        for _, elem in ET.iterparse(fh):
            if elem.tag == "vehicle":
                route = elem.find("route")
                edges = route.get("edges").split() if route is not None else []
                yield float(elem.get("depart")), edges
                elem.clear()


def junction_lanes(net: sumolib.net.Net) -> dict[str, dict[tuple[str, str], set[str]]]:
    """Per traffic light: movement (in edge, out edge) -> incoming lanes serving it."""
    result: dict[str, dict[tuple[str, str], set[str]]] = {}
    for tl in net.getTrafficLights():
        movements: dict[tuple[str, str], set[str]] = defaultdict(set)
        for in_lane, out_lane, _ in tl.getConnections():
            movements[(in_lane.getEdge().getID(), out_lane.getEdge().getID())].add(in_lane.getID())
        result[tl.getID()] = dict(movements)
    return result


def movement_counts(
    route_files: list[Path],
    movements: dict[str, dict[tuple[str, str], set[str]]],
    bin_seconds: float,
) -> dict[str, dict[int, dict[tuple[str, str], float]]]:
    """Vehicles per (traffic light, departure-time bin, movement), averaged over route files."""
    index = {m: tls for tls, moves in movements.items() for m in moves}
    counts: dict[str, dict[int, dict[tuple[str, str], float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for route_file in route_files:
        for depart, edges in _iter_vehicles(route_file):
            time_bin = int(depart // bin_seconds)
            for move in zip(edges, edges[1:], strict=False):
                tls = index.get(move)
                if tls is not None:
                    counts[tls][time_bin][move] += 1.0 / len(route_files)
    return counts


def hourly_lane_flows(
    counts: dict[int, dict[tuple[str, str], float]],
    lanes: dict[tuple[str, str], set[str]],
    *,
    duration: float,
    bin_seconds: float,
    busiest_window: bool,
    weights: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Hourly flow per incoming lane: mean over the whole period, or the busiest bin.

    ``weights`` converts each movement's vehicles into through-vehicle equivalents.
    """
    if busiest_window and counts:
        busiest = max(counts, key=lambda b: sum(counts[b].values()))
        moves, hours = counts[busiest], bin_seconds / 3600.0
    else:
        moves = defaultdict(float)
        for per_bin in counts.values():
            for move, n in per_bin.items():
                moves[move] += n
        hours = duration / 3600.0
    flows: dict[str, float] = defaultdict(float)
    for move, n in moves.items():
        serving = lanes.get(move, set())
        weight = weights.get(move, 1.0) if weights else 1.0
        for lane in serving:
            flows[lane] += weight * n / hours / len(serving)
    return dict(flows)


def plan_signal(
    tl: sumolib.net.TLS,
    phases: list[tuple[str, float]],
    lane_flows: dict[str, float],
    cfg: dict[str, Any],
) -> SignalPlan:
    """Webster plan for one traffic light (see module docstring)."""
    link_lane = {idx: in_lane.getID() for in_lane, _, idx in tl.getConnections()}
    green_idx = [i for i, (state, _) in enumerate(phases) if is_green_phase(state)]
    ratios = []
    for i in green_idx:
        state = phases[i][0]
        protected = {link_lane[k] for k, c in enumerate(state) if c == "G" and k in link_lane}
        ratios.append(
            max((lane_flows.get(lane, 0.0) for lane in protected), default=0.0)
            / cfg["saturation_flow"]
        )
    total_ratio = float(sum(ratios))
    lost = cfg["lost_time_per_phase"] * len(green_idx)
    non_green = sum(d for i, (_, d) in enumerate(phases) if i not in green_idx)
    min_cycle = non_green + cfg["min_green"] * len(green_idx)
    capped = total_ratio >= 1.0
    cycle = cfg["max_cycle"] if capped else (1.5 * lost + 5.0) / (1.0 - total_ratio)
    if cycle > cfg["max_cycle"]:
        cycle, capped = cfg["max_cycle"], True
    cycle = max(cycle, min_cycle)
    available = cycle - non_green
    shares = (
        np.array(ratios) / total_ratio if total_ratio > 0 else np.full(len(ratios), 1 / len(ratios))
    )
    greens = np.clip(np.round(available * shares), cfg["min_green"], cfg["max_green"])
    return SignalPlan(
        tl.getID(), float(greens.sum() + non_green), total_ratio, capped, greens.tolist(), ratios
    )


def program_phases(net_file: Path) -> dict[str, list[tuple[str, float]]]:
    """Static program of each traffic light as (state, duration) pairs."""
    result = {}
    for tl in ET.parse(net_file).getroot().iter("tlLogic"):
        result[tl.get("id")] = [
            (p.get("state"), float(p.get("duration"))) for p in tl.iter("phase")
        ]
    return result


def write_programs(
    plans: dict[str, SignalPlan], phases: dict[str, list[tuple[str, float]]], out: Path
) -> None:
    root = ET.Element("additional")
    for tls, plan in sorted(plans.items()):
        logic = ET.SubElement(
            root, "tlLogic", id=tls, type="static", programID=PROGRAM_ID, offset="0"
        )
        greens = iter(plan.greens)
        for state, duration in phases[tls]:
            value = next(greens) if is_green_phase(state) else duration
            ET.SubElement(logic, "phase", duration=f"{value:g}", state=state)
    ET.indent(root, space="    ")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)


def resolve_parameters(manifest, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Saturation flow per movement type and lost time per phase, measured or configured.

    Returns the plan config (``saturation_flow`` = the through value used as the
    reference, numeric ``lost_time_per_phase``) and the flow per type.
    """
    cfg = dict(cfg)
    measured: dict[str, Any] = {}
    if "measured" in (cfg["saturation_flow"], cfg.get("lost_time_per_phase")):
        measured = load_measured(manifest)
    if cfg["saturation_flow"] == "measured":
        by_type = {k: v["saturation_flow"] for k, v in measured["by_type"].items()}
        through = by_type["through"]
        saturation = {k: float(by_type.get(k, through)) for k in TYPES}
    else:
        saturation = {k: float(cfg["saturation_flow"]) for k in TYPES}
    if cfg.get("lost_time_per_phase") == "measured":
        cfg["lost_time_per_phase"] = float(measured["lost_time_per_phase_s"])
    elif cfg.get("lost_time_per_phase") is None:
        cfg["lost_time_per_phase"] = cfg["yellow_time"]
    cfg["saturation_flow"] = saturation["through"]
    return cfg, saturation


def build_plans(
    manifest, profiles: list[str], cfg: dict[str, Any]
) -> dict[str, dict[str, SignalPlan]]:
    """Webster programs for every profile of a scenario, written to ``<scenario>/webster/``.

    ``cfg`` is ``baselines.webster`` plus ``yellow_time`` (the scenario's), used as the
    lost time per phase when ``lost_time_per_phase`` is null. Measured values come
    from ``<scenario>/webster/saturation.yaml`` (``baselines saturation`` command).
    """
    net_file = manifest.path(manifest.net_file)
    net = sumolib.net.readNet(str(net_file), withPrograms=True)
    cfg, saturation = resolve_parameters(manifest, cfg)
    kinds = movement_types(net)
    weights = {
        tls: {move: saturation["through"] / saturation[kind] for move, kind in moves.items()}
        for tls, moves in kinds.items()
    }
    lanes = junction_lanes(net)
    phases = program_phases(net_file)
    tls_by_id = {tl.getID(): tl for tl in net.getTrafficLights()}
    result: dict[str, dict[str, SignalPlan]] = {}
    summary: dict[str, Any] = {}
    for profile in profiles:
        counts = movement_counts(manifest.route_files(profile, "train"), lanes, cfg["peak_window"])
        plans = {}
        for tls, tl in tls_by_id.items():
            flows = hourly_lane_flows(
                counts.get(tls, {}), lanes[tls], duration=manifest.duration,
                bin_seconds=cfg["peak_window"], busiest_window=profile == "peak",
                weights=weights.get(tls),
            )  # fmt: skip
            plans[tls] = plan_signal(tl, phases[tls], flows, cfg)
        write_programs(plans, phases, manifest.root / WEBSTER_DIR / f"{profile}.add.xml")
        result[profile] = plans
        summary[profile] = {
            tls: {
                "cycle_s": round(p.cycle, 1),
                "Y": round(p.flow_ratio_sum, 3),
                "capped": p.capped,
                "greens_s": [round(g, 1) for g in p.greens],
            }  # fmt: skip
            for tls, p in sorted(plans.items())
        }
    with (manifest.root / WEBSTER_DIR / "plans.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"config": cfg, "saturation_flow_by_type": saturation, "plans": summary},
            fh,
            sort_keys=False,
        )
    return result


def webster_file(manifest, profile: str) -> Path:
    path = manifest.root / WEBSTER_DIR / f"{profile}.add.xml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `python -m trafficmarl baselines webster`")
    return path
