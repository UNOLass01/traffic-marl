"""Cropped LuST (Luxembourg SUMO Traffic) scenario.

Procedure (every step is driven by ``configs/scenarios/lust.yaml``):

1. Select the traffic lights whose controlled junctions lie inside a square of
   half-side ``half_size`` around a geographic centre (Place d'Armes).
2. Keep every passenger edge inside the square plus every edge touching a
   selected junction, so selected junctions keep all their approaches and their
   original signal programs stay valid. Traffic lights left on the border of
   the crop (approaches cut off) are turned into plain junctions, and only the
   largest connected component is kept. Traffic lights whose program has a
   single distinct green state (no decision can change anything) are turned
   into plain junctions too.
3. Load LuST's static programs (``tll.static.xml``) as the only programs of the
   network (fixed-time baseline) and move LuST's original actuated programs to
   a separate additional file (actuated baseline; their green phases get
   SUMO's default 5-50 s range, which LuST omits). Keeping both in the network
   crashes SUMO under SUMO-RL, which rewrites the first program of each
   traffic light (the actuated one) while the static one is active.
4. Cut the original LuST routes to the sub-network with SUMO's ``cutRoutes.py``,
   which shifts departure times to the moment each vehicle enters the crop.
5. Extract disjoint one-hour windows (train: 07:00-08:00, eval: 08:00-09:00),
   shift them to start at t=0, and scale demand by the profile multipliers
   (seeded thinning / duplication) to obtain low, medium, high and peak files.

Reference: L. Codeca, R. Frank, S. Faye, T. Engel, "Luxembourg SUMO Traffic
(LuST) Scenario: Traffic Demand Evaluation", IEEE ITS Magazine, 2017.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import sumolib

from trafficmarl.scenarios.demand import rate_multiplier
from trafficmarl.scenarios.spec import SPLITS, Manifest, output_dir, route_file_name, write_sumocfg
from trafficmarl.scenarios.synthetic import profile_seed
from trafficmarl.scenarios.tls import ACTUATED_FILE, is_green_phase, split_tls_programs
from trafficmarl.utils import paths, sumo
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)

DUPLICATE_JITTER_S = 30.0  # duplicates depart up to +-30 s from their original


def _tls_nodes(tl: sumolib.net.TLS) -> set:
    return {conn[0].getEdge().getToNode() for conn in tl.getConnections()}


def select_crop(
    net: sumolib.net.Net, crop_cfg: dict[str, Any]
) -> tuple[list[str], set[str], list[str]]:
    """Return (selected TLS ids, edges to keep, border TLS ids to unset)."""
    cx, cy = net.convertLonLat2XY(crop_cfg["center_lon"], crop_cfg["center_lat"])
    half = crop_cfg["half_size"]

    def inside(xy: tuple[float, float]) -> bool:
        return abs(xy[0] - cx) <= half and abs(xy[1] - cy) <= half

    selected = []
    for tl in net.getTrafficLights():
        coords = [n.getCoord() for n in _tls_nodes(tl)]
        centroid = (np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords]))
        if inside(centroid):
            selected.append(tl)
    selected_nodes = {n.getID() for tl in selected for n in _tls_nodes(tl)}

    keep = set()
    for edge in net.getEdges():
        if not edge.allows("passenger"):
            continue
        a, b = edge.getFromNode(), edge.getToNode()
        both_inside = inside(a.getCoord()) and inside(b.getCoord())
        if both_inside or a.getID() in selected_nodes or b.getID() in selected_nodes:
            keep.add(edge.getID())
    kept_nodes = {
        n.getID() for e in keep for n in (net.getEdge(e).getFromNode(), net.getEdge(e).getToNode())
    }
    border = [
        tl.getID()
        for tl in net.getTrafficLights()
        if tl not in selected and any(n.getID() in kept_nodes for n in _tls_nodes(tl))
    ]
    return sorted(tl.getID() for tl in selected), keep, sorted(border)


def degenerate_signals(net_file: Path) -> list[str]:
    """Traffic lights whose static program has fewer than two distinct green states.

    Such a light (in LuST, pedestrian lights without their pedestrian phase)
    shows the same movements green in every phase: no decision can change
    anything, and its fixed program only inserts yellows without ever showing
    red. These lights are removed (turned into plain junctions) for all methods.
    """
    result = []
    for tl in ET.parse(net_file).getroot().iter("tlLogic"):
        if tl.get("type") != "static":
            continue
        greens = {p.get("state") for p in tl.iter("phase") if is_green_phase(p.get("state", ""))}
        if len(greens) < 2:
            result.append(tl.get("id"))
    return sorted(set(result))


def crop_network(
    source: Path, tll_static: Path, keep: set[str], unset: list[str], out: Path
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        edge_list = Path(tmp) / "edges.txt"
        edge_list.write_text("\n".join(sorted(keep)), encoding="utf-8")
        cmd = [
            sumo.binary("netconvert"),
            "--sumo-net-file", source,
            "--keep-edges.input-file", edge_list,
            "--keep-edges.components", "1",
            "--tllogic-files", tll_static,
            "--no-warnings",
            "--output-file", out,
        ]  # fmt: skip
        if unset:
            cmd += ["--tls.unset", ",".join(unset)]
        sumo.run(cmd)


def cut_routes(crop_net: Path, full_net: Path, route_files: list[Path], out: Path) -> None:
    cmd = [*sumo.tool("route/cutRoutes.py"), crop_net, *route_files]
    cmd += ["--orig-net", full_net, "--routes-output", out]
    sumo.run(cmd)


def scaled_window(
    vehicles: list[ET.Element],
    start: float,
    duration: float,
    profile: list[float],
    rng: np.random.Generator,
) -> list[ET.Element]:
    """Vehicles departing in ``[start, start + duration)``, shifted to t=0 and rescaled.

    Each vehicle is kept ``floor(m)`` times plus once more with probability
    ``m - floor(m)``, where ``m`` is the profile multiplier at its departure
    time: ``m < 1`` thins the demand, ``m > 1`` duplicates vehicles (with a
    small departure jitter so copies do not enter on top of each other).
    """
    out = []
    for veh in vehicles:
        t = float(veh.get("depart")) - start
        if not 0.0 <= t < duration:
            continue
        m = rate_multiplier(profile, t, duration)
        copies = int(np.floor(m)) + int(rng.uniform() < m - np.floor(m))
        for k in range(copies):
            clone = copy.deepcopy(veh)
            depart = (
                t
                if k == 0
                else float(np.clip(t + rng.uniform(-1, 1) * DUPLICATE_JITTER_S, 0, duration - 1))
            )
            clone.set("depart", f"{depart:.2f}")
            if k > 0:
                clone.set("id", f"{veh.get('id')}_dup{k}")
            out.append(clone)
    out.sort(key=lambda v: float(v.get("depart")))
    return out


def write_vehicles(vehicles: list[ET.Element], path: Path, comment: str) -> None:
    root = ET.Element("routes")
    root.extend(vehicles)
    ET.indent(root, space="    ")
    with paths.open_output(path) as fh:
        fh.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write(f"<!-- {comment} -->\n".encode())
        ET.ElementTree(root).write(fh, encoding="utf-8")


def plot_crop(
    full_net: sumolib.net.Net, crop: sumolib.net.Net, crop_cfg: dict[str, Any], out: Path
) -> None:
    """Map of the cropped area over the surrounding LuST network (report figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cx, cy = full_net.convertLonLat2XY(crop_cfg["center_lon"], crop_cfg["center_lat"])
    half, margin = crop_cfg["half_size"], 2.0 * crop_cfg["half_size"]
    fig, ax = plt.subplots(figsize=(8, 8))
    for edge in full_net.getEdges():
        xs, ys = zip(*edge.getShape(), strict=True)
        if (
            min(xs) < cx + margin
            and max(xs) > cx - margin
            and min(ys) < cy + margin
            and max(ys) > cy - margin
        ):
            ax.plot(xs, ys, color="#c8c8c8", lw=0.6, zorder=1)
    # The cropped net keeps the original coordinates' offset only up to
    # netconvert's own re-centring; map it back through geo-coordinates.
    for edge in crop.getEdges():
        lonlat = [crop.convertXY2LonLat(x, y) for x, y in edge.getShape()]
        pts = [full_net.convertLonLat2XY(lon, lat) for lon, lat in lonlat]
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, color="#1f5fa8", lw=1.4, zorder=2)
    for tl in crop.getTrafficLights():
        coords = [n.getCoord() for n in _tls_nodes(tl)]
        lon, lat = crop.convertXY2LonLat(
            np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])
        )
        x, y = full_net.convertLonLat2XY(lon, lat)
        ax.scatter([x], [y], s=60, color="#d62728", edgecolor="black", zorder=3)
    ax.add_patch(
        Rectangle(
            (cx - half, cy - half), 2 * half, 2 * half, fill=False, ls="--", lw=1.2, color="black"
        )
    )
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, UTM 32N décalé)")
    ax.set_ylabel("y (m)")
    n_tls = len(crop.getTrafficLights())
    ax.set_title(f"LuST — zone découpée autour de la Place d'Armes ({n_tls} carrefours à feux)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build(scenario_cfg: dict[str, Any]) -> Manifest:
    """Crop the network, cut and window the routes, write the manifest and the map figure."""
    name = scenario_cfg["name"]
    src = scenario_cfg["source"]
    crop_cfg = scenario_cfg["crop"]
    lust_dir = paths.resolve(src["dir"])
    full_net_path = lust_dir / src["net_file"]
    root = output_dir(scenario_cfg)
    root.mkdir(parents=True, exist_ok=True)

    full_net = sumolib.net.readNet(str(full_net_path), withPrograms=True)
    selected, keep, border = select_crop(full_net, crop_cfg)
    net_file = f"{name}.net.xml"
    crop_network(full_net_path, lust_dir / src["tls_static_file"], keep, border, root / net_file)
    degenerate = degenerate_signals(root / net_file)
    if degenerate:
        crop_network(
            full_net_path, lust_dir / src["tls_static_file"], keep, border + degenerate,
            root / net_file,
        )  # fmt: skip
    n_actuated = split_tls_programs(root / net_file, root / ACTUATED_FILE)
    crop = sumolib.net.readNet(str(root / net_file), withPrograms=True)
    tls_ids = sorted(tl.getID() for tl in crop.getTrafficLights())

    vtypes = "vtypes.add.xml"
    shutil.copy(lust_dir / src["vtypes_file"], root / vtypes)

    dem = scenario_cfg["demand"]
    manifest = Manifest(
        name=name,
        root=root,
        net_file=net_file,
        tls_ids=tls_ids,
        begin=0,
        duration=int(dem["duration"]),
        additional_files=[vtypes],
        actuated_file=ACTUATED_FILE,
        sumo_version=sumo.version(),
        robustness_only_profiles=list(dem.get("robustness_only", [])),
        notes={
            "source": "LuST scenario (Codeca et al., 2017), MIT license",
            "crop_center_lonlat": [crop_cfg["center_lon"], crop_cfg["center_lat"]],
            "crop_half_size_m": crop_cfg["half_size"],
            "tls_in_square": len(selected),
            "tls_dropped_disconnected": sorted(set(selected) - set(tls_ids)),
            "tls_unset_on_border": border,
            "tls_unset_degenerate": degenerate,
            "edges": len(crop.getEdges()),
            "actuated_programs": n_actuated,
            "route_source": src["route_files"],
            "windows_s": dem["windows"],
        },
    )

    with tempfile.TemporaryDirectory() as tmp:
        cut_file = Path(tmp) / "cut.rou.xml"
        cut_routes(
            root / net_file, full_net_path, [lust_dir / f for f in src["route_files"]], cut_file
        )
        vehicles = list(ET.parse(cut_file).getroot().iter("vehicle"))

    for profile, multipliers in dem["profiles"].items():
        for split in SPLITS:
            for index, start in enumerate(dem["windows"][split]):
                seed = profile_seed(dem["seeds"][split], profile)
                selected_vehicles = scaled_window(
                    vehicles, start, dem["duration"], multipliers, np.random.default_rng(seed)
                )
                rel = route_file_name(profile, split, index)
                write_vehicles(
                    selected_vehicles,
                    root / rel,
                    comment=(
                        f"scenario={name} profile={profile} split={split} "
                        f"window_start={start} seed={seed}"
                    ),
                )
                manifest.routes.setdefault(profile, {}).setdefault(split, []).append(rel)
                manifest.vehicles[rel] = len(selected_vehicles)

    manifest.sumocfg = write_sumocfg(
        manifest, manifest.routes["medium"]["eval"][0], end=manifest.duration
    )
    manifest.save()
    plot_crop(full_net, crop, crop_cfg, paths.resolve(scenario_cfg["figure"]))
    logger.info("built %s: %d TLS (of %d selected)", name, len(tls_ids), len(selected))
    return manifest
