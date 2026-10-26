"""Headless validation of built scenarios.

Each route file is simulated until every vehicle has left the network (or a
time cap is reached) under the scenario's static or actuated programs. A file
passes when SUMO reports no error, every vehicle is inserted and arrives, and
teleports (SUMO's way of clearing a jam: a vehicle stuck for 300 s is moved
ahead) stay below a threshold. Per-profile rules (``validation.profiles`` in
the scenario config) adapt this to demand levels beyond saturation:

* ``gridlock_check``: rerun with teleporting disabled, so that a permanent
  gridlock cannot be cleared by teleports, and require that the network
  drains. The run stops early once no vehicle has arrived for
  ``gridlock_window`` seconds while vehicles remain (gridlock detected).
  With ``gridlock_required: false`` the outcome is recorded but not required
  (LuST, whose real junction geometry deadlocks without teleports above its
  real demand; see docs/decisions.md D15).
* ``report_only``: statistics are recorded but the file cannot fail (stress
  levels used only by the robustness study).

Runs are independent jobs executed in parallel worker processes. Each worker
handles a single job and exits (libsumo supports one simulation per process).
Every finished job is appended to the CSV and the log at once, so an
interrupted validation loses nothing and resumes where it stopped.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trafficmarl.scenarios.spec import Manifest, load_scenario_config, output_dir
from trafficmarl.utils import sumo
from trafficmarl.utils.logging_utils import get_logger
from trafficmarl.utils.seeding import sumo_seed_args

logger = get_logger(__name__)

TIME_TO_TELEPORT = 300  # SUMO default: a vehicle stuck this long (s) is teleported
DEFAULT_RULES: dict[str, Any] = {
    "controller": "static",
    "end": 7200,
    "max_teleport_ratio": 0.01,
    "gridlock_check": False,
    "gridlock_window": 600,
    "gridlock_required": True,
    "report_only": False,
}
CSV_FIELDS = [
    "key", "scenario", "profile", "split", "route_file", "controller", "loaded", "inserted",
    "arrived", "waiting_to_insert", "running_at_end", "teleports", "teleport_ratio",
    "mean_waiting_time", "mean_travel_time", "mean_time_loss", "gridlock_check",
    "drained_without_teleport", "arrived_without_teleport", "gridlock_at", "report_only",
    "passed", "wall_time_s",
]  # fmt: skip


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


@dataclass
class DrainResult:
    """Outcome of a run with teleporting disabled."""

    drained: bool
    loaded: int
    arrived: int
    gridlock_at: float | None  # simulation time at which gridlock was detected


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


def sumo_command(
    manifest: Manifest,
    route_file: str,
    *,
    end: int,
    controller: str,
    time_to_teleport: int,
    seed: int = 0,
) -> list[str]:
    """SUMO command line for one route file with static or actuated programs."""
    if controller not in ("static", "actuated"):
        raise ValueError(f"Unknown controller '{controller}'")
    additional = list(manifest.additional_files)
    if controller == "actuated":
        if not manifest.actuated_file:
            raise ValueError(f"Scenario {manifest.name} has no actuated programs")
        additional.append(manifest.actuated_file)
    cmd = [
        sumo.binary("sumo"),
        "--net-file", str(manifest.path(manifest.net_file)),
        "--route-files", str(manifest.path(route_file)),
        "--begin", str(manifest.begin),
        "--end", str(end),
        "--time-to-teleport", str(time_to_teleport),
        "--no-step-log",
        "--no-warnings",
        *sumo_seed_args(seed),
    ]  # fmt: skip
    if additional:
        cmd += ["--additional-files", ",".join(str(manifest.path(f)) for f in additional)]
    return cmd


def simulate(
    manifest: Manifest,
    route_file: str,
    *,
    end: int,
    controller: str = "static",
    time_to_teleport: int = TIME_TO_TELEPORT,
    seed: int = 0,
) -> RunStats:
    """Run one route file in a SUMO subprocess and return its statistics."""
    cmd = sumo_command(
        manifest, route_file, end=end, controller=controller,
        time_to_teleport=time_to_teleport, seed=seed,
    )  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        stats_file = Path(tmp) / "stats.xml"
        # --duration-log.statistics adds <vehicleTripStatistics> to the output.
        sumo.run([*cmd, "--statistic-output", str(stats_file), "--duration-log.statistics"])
        return parse_statistics(stats_file.read_text(encoding="utf-8"), route_file)


def drain_without_teleport(
    manifest: Manifest, route_file: str, *, end: int, controller: str, window: int
) -> DrainResult:
    """Simulate with teleporting disabled and stop at the first sign of permanent gridlock.

    Gridlock = no vehicle arrived during the last ``window`` seconds while
    vehicles are still in (or waiting to enter) the network. Uses libsumo, so
    it must run in a process that holds no other simulation.
    """
    import libsumo

    cmd = sumo_command(manifest, route_file, end=end, controller=controller, time_to_teleport=-1)
    libsumo.start(cmd)
    try:
        loaded = arrived = 0
        last_arrival = float(manifest.begin)
        while True:
            libsumo.simulationStep()
            now = libsumo.simulation.getTime()
            loaded += libsumo.simulation.getLoadedNumber()
            new_arrivals = libsumo.simulation.getArrivedNumber()
            if new_arrivals:
                arrived += new_arrivals
                last_arrival = now
            if libsumo.simulation.getMinExpectedNumber() == 0:
                return DrainResult(True, loaded, arrived, None)
            if now - last_arrival >= window:
                return DrainResult(False, loaded, arrived, now)
            if now >= end:
                return DrainResult(False, loaded, arrived, None)
    finally:
        libsumo.close()


def profile_rules(validation_cfg: dict, profile: str) -> dict:
    """Validation rules for one demand profile: defaults < scenario-wide < per-profile."""
    rules = dict(DEFAULT_RULES)
    rules.update({k: v for k, v in validation_cfg.items() if k != "profiles"})
    rules.update(validation_cfg.get("profiles", {}).get(profile, {}))
    return rules


@dataclass(frozen=True)
class Job:
    scenario: str
    root: str
    profile: str
    split: str
    route_file: str
    rules_json: str
    key: str

    @property
    def rules(self) -> dict:
        return json.loads(self.rules_json)


def _digest(*parts: bytes) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part)
    return h.hexdigest()[:16]


def plan_jobs(names: Iterable[str], profiles: Iterable[str] | None = None) -> list[Job]:
    """One job per route file. The key changes whenever the network, the route
    file or the rules change, so stale results are never reused on resume."""
    wanted = set(profiles) if profiles else None
    jobs = []
    for name in names:
        cfg = load_scenario_config(name)
        manifest = Manifest.load(output_dir(cfg))
        net_bytes = manifest.path(manifest.net_file).read_bytes()
        for profile, splits in manifest.routes.items():
            if wanted is not None and profile not in wanted:
                continue
            rules_json = json.dumps(profile_rules(cfg["validation"], profile), sort_keys=True)
            for split, files in splits.items():
                for rel in files:
                    key = _digest(
                        name.encode(),
                        net_bytes,
                        manifest.path(rel).read_bytes(),
                        rules_json.encode(),
                    )
                    jobs.append(Job(name, str(manifest.root), profile, split, rel, rules_json, key))
    return jobs


def run_job(job: Job) -> dict:
    """Validate one route file (executed in a fresh worker process)."""
    started = time.monotonic()
    rules = job.rules
    manifest = Manifest.load(job.root)
    stats = simulate(manifest, job.route_file, end=rules["end"], controller=rules["controller"])
    row: dict[str, Any] = {
        "key": job.key,
        "scenario": job.scenario,
        "profile": job.profile,
        "split": job.split,
        "controller": rules["controller"],
        **asdict(stats),
        "teleport_ratio": round(stats.teleport_ratio, 4),
        "gridlock_check": rules["gridlock_check"],
        "report_only": rules["report_only"],
    }
    passed = stats.waiting_to_insert == 0 and stats.running_at_end == 0
    if rules["max_teleport_ratio"] is not None:
        passed = passed and stats.teleport_ratio <= rules["max_teleport_ratio"]
    if rules["gridlock_check"]:
        drain = drain_without_teleport(
            manifest, job.route_file, end=rules["end"], controller=rules["controller"],
            window=rules["gridlock_window"],
        )  # fmt: skip
        row["drained_without_teleport"] = drain.drained
        row["arrived_without_teleport"] = drain.arrived
        row["gridlock_at"] = drain.gridlock_at
        if rules["gridlock_required"]:
            passed = passed and drain.drained
    row["passed"] = True if rules["report_only"] else bool(passed)
    row["wall_time_s"] = round(time.monotonic() - started, 1)
    return row


def completed_keys(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {row["key"] for row in csv.DictReader(fh)}


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def run_validation(
    jobs: list[Job], csv_path: Path, *, workers: int | None = None, fresh: bool = False
) -> int:
    """Run pending jobs in parallel; append each row to ``csv_path`` as soon as it is ready.

    Returns the number of jobs executed (jobs already in the CSV are skipped).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = csv_path.with_suffix(".log")
    if fresh:
        csv_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
    done = completed_keys(csv_path)
    pending = [job for job in jobs if job.key not in done]
    logger.info(
        "%d jobs, %d already recorded, %d to run", len(jobs), len(jobs) - len(pending), len(pending)
    )
    if not pending:
        return 0
    new_file = not csv_path.is_file()
    # A fresh process per job: libsumo cannot host two simulations in one process.
    ctx = multiprocessing.get_context("spawn")
    with (
        csv_path.open("a", newline="", encoding="utf-8") as csv_fh,
        log_path.open("a", encoding="utf-8") as log_fh,
        ctx.Pool(processes=workers or default_workers(), maxtasksperchild=1) as pool,
    ):
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_job, pending), start=1):
            writer.writerow(row)
            csv_fh.flush()
            line = (
                f"[{i}/{len(pending)}] {row['scenario']} {row['profile']} {row['split']} "
                f"{Path(row['route_file']).name}: passed={row['passed']} "
                f"teleports={row['teleports']} wait={row['mean_waiting_time']:.1f}s "
                f"drained_without_teleport={row.get('drained_without_teleport', '-')} "
                f"({row['wall_time_s']}s)"
            )
            log_fh.write(line + "\n")
            log_fh.flush()
            logger.info(line)
    return len(pending)


def read_results(csv_path: Path, jobs: list[Job]) -> list[dict]:
    """Rows of ``csv_path`` matching ``jobs`` (latest row per key)."""
    keys = {job.key for job in jobs}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = {row["key"]: row for row in csv.DictReader(fh) if row["key"] in keys}
    return list(rows.values())
