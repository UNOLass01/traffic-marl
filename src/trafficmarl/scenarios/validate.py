"""Headless validation of built scenarios under SUMO's own (default) TLS programs.

Each route file is simulated until every vehicle has left the network (or a
time cap is reached). A scenario passes when SUMO reports no error and the
number of teleports, SUMO's signal of gridlock, stays below a small fraction of
the inserted vehicles.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from trafficmarl.scenarios.spec import Manifest
from trafficmarl.utils import sumo
from trafficmarl.utils.seeding import sumo_seed_args

TIME_TO_TELEPORT = 300  # SUMO default: a vehicle stuck this long (s) is teleported


@dataclass
class RunStats:
    route_file: str
    loaded: int
    inserted: int
    arrived: int
    waiting_to_insert: int
    running_at_end: int
    teleports: int
    mean_waiting_time: float
    mean_travel_time: float
    mean_time_loss: float

    @property
    def teleport_ratio(self) -> float:
        return self.teleports / self.inserted if self.inserted else 0.0


def parse_statistics(xml_text: str, route_file: str = "") -> RunStats:
    """Parse SUMO's ``--statistic-output`` file."""
    root = ET.fromstring(xml_text)
    vehicles = root.find("vehicles")
    teleports = root.find("teleports")
    trips = root.find("vehicleTripStatistics")
    if vehicles is None or teleports is None or trips is None:
        raise ValueError("Unexpected statistic-output format")
    return RunStats(
        route_file=route_file,
        loaded=int(vehicles.get("loaded", 0)),
        inserted=int(vehicles.get("inserted", 0)),
        arrived=int(trips.get("count", 0)),
        waiting_to_insert=int(vehicles.get("waiting", 0)),
        running_at_end=int(vehicles.get("running", 0)),
        teleports=int(teleports.get("total", 0)),
        mean_waiting_time=float(trips.get("waitingTime", 0.0)),
        mean_travel_time=float(trips.get("duration", 0.0)),
        mean_time_loss=float(trips.get("timeLoss", 0.0)),
    )


def simulate(manifest: Manifest, route_file: str, *, end: int, seed: int = 0) -> RunStats:
    """Run one route file headless with the network's default TLS programs."""
    with tempfile.TemporaryDirectory() as tmp:
        stats_file = Path(tmp) / "stats.xml"
        cmd = [
            sumo.binary("sumo"),
            "--net-file",
            manifest.path(manifest.net_file),
            "--route-files",
            manifest.path(route_file),
            "--begin",
            str(manifest.begin),
            "--end",
            str(end),
            "--time-to-teleport",
            str(TIME_TO_TELEPORT),
            "--statistic-output",
            stats_file,
            "--no-step-log",
            "--no-warnings",
            "--duration-log.statistics",  # adds <vehicleTripStatistics> to the output
            *sumo_seed_args(seed),
        ]
        if manifest.additional_files:
            cmd += [
                "--additional-files",
                ",".join(str(manifest.path(f)) for f in manifest.additional_files),
            ]
        sumo.run(cmd)
        return parse_statistics(stats_file.read_text(encoding="utf-8"), route_file)


def validate(manifest: Manifest, *, end: int, max_teleport_ratio: float) -> list[dict]:
    """Simulate every route file of the scenario; one result row per file."""
    rows = []
    for profile, splits in manifest.routes.items():
        for split, files in splits.items():
            for rel in files:
                stats = simulate(manifest, rel, end=end)
                row = {
                    "scenario": manifest.name,
                    "profile": profile,
                    "split": split,
                    **asdict(stats),
                }
                row["teleport_ratio"] = round(stats.teleport_ratio, 4)
                # Vehicles never inserted or still running at the time cap mean
                # the demand exceeds what the default program can serve.
                row["passed"] = (
                    stats.teleport_ratio <= max_teleport_ratio
                    and stats.waiting_to_insert == 0
                    and stats.running_at_end == 0
                )
                rows.append(row)
    return rows
