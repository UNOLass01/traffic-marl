"""Synthetic scenarios (single intersection, corridor, grids) built with ``netgenerate``.

Every intersection gets two lanes per approach plus a dedicated left-turn lane,
so SUMO's default static program has four green phases (NS through, NS left,
EW through, EW left): the standard action space in traffic-signal-control
benchmarks and SUMO's own "default program" baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sumolib

from trafficmarl.scenarios import demand
from trafficmarl.scenarios.spec import SPLITS, Manifest, output_dir, route_file_name, write_sumocfg
from trafficmarl.utils import sumo
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_network(net_cfg: dict[str, Any], net_file: Path) -> None:
    """Generate the road network; ``ny = 1`` gives a corridor, ``nx = ny = 1`` a single junction."""
    cmd = [
        sumo.binary("netgenerate"),
        "--grid",
        f"--grid.x-number={net_cfg['nx']}",
        f"--grid.y-number={net_cfg['ny']}",
        f"--grid.length={net_cfg['block_length']}",
        f"--grid.attach-length={net_cfg['attach_length']}",
        f"--default.lanenumber={net_cfg['lanes']}",
        f"--default.speed={net_cfg['speed']}",
        f"--turn-lanes={1 if net_cfg['left_turn_lane_length'] > 0 else 0}",
        f"--turn-lanes.length={net_cfg['left_turn_lane_length']}",
        "--tls.guess=true",
        "--tls.default-type=static",
        f"--tls.cycle.time={net_cfg['tls_cycle']}",
        "--no-turnarounds=true",
        "--seed=0",
        f"--output-file={net_file}",
    ]
    sumo.run(cmd)


def profile_seed(split_seed: int, profile: str) -> int:
    """Distinct, reproducible seed per (split seed, profile).

    Without this, ``low`` and ``high`` with the same split seed would share
    arrival-time patterns, which would correlate results across profiles.
    """
    entropy = [split_seed, *profile.encode()]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def build(scenario_cfg: dict[str, Any]) -> Manifest:
    """Build network, all route files and the manifest for one synthetic scenario."""
    name = scenario_cfg["name"]
    root = output_dir(scenario_cfg)
    root.mkdir(parents=True, exist_ok=True)
    net_file = f"{name}.net.xml"
    build_network(scenario_cfg["network"], root / net_file)
    net = sumolib.net.readNet(str(root / net_file), withPrograms=True)

    dem = scenario_cfg["demand"]
    manifest = Manifest(
        name=name,
        root=root,
        net_file=net_file,
        tls_ids=sorted(tl.getID() for tl in net.getTrafficLights()),
        begin=0,
        duration=int(dem["duration"]),
    )
    for profile, multipliers in dem["profiles"].items():
        for split in SPLITS:
            for index, split_seed in enumerate(dem["splits"][split]):
                seed = profile_seed(split_seed, profile)
                vehicles = demand.generate_vehicles(
                    net,
                    entry_rate=dem["entry_rate"],
                    profile=multipliers,
                    duration=dem["duration"],
                    turn_probabilities=dem["turn_probabilities"],
                    seed=seed,
                    entry_weights=dem.get("entry_weights"),
                )
                rel = route_file_name(profile, split, index)
                demand.write_route_file(
                    vehicles,
                    root / rel,
                    comment=f"scenario={name} profile={profile} split={split} seed={seed}",
                )
                manifest.routes.setdefault(profile, {}).setdefault(split, []).append(rel)
                manifest.vehicles[rel] = len(vehicles)
    gui_route = manifest.routes["medium"]["eval"][0]
    manifest.sumocfg = write_sumocfg(manifest, gui_route, end=manifest.duration)
    manifest.save()
    logger.info(
        "built %s: %d TLS, %d route files", name, len(manifest.tls_ids), len(manifest.vehicles)
    )
    return manifest
