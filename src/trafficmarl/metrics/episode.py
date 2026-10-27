"""Per-episode metrics computed from SUMO's own output files, identically for every method.

Sources (all written by SUMO during the evaluation episode):

* ``tripinfo`` with ``write-unfinished``: one record per inserted vehicle,
  including those still in the network when the episode ends, with waiting
  time, time loss, insertion delay (``departDelay``) and CO2 emissions;
* the route file: vehicles that should have departed but were never inserted
  do not appear in tripinfo, so their intended departure times are read here;
* ``summary``: network-wide halting vehicles at every second;
* ``SaveTLSStates``: every signal state at every second, from which green
  durations and the longest red of each phase are measured.

Delay definitions (docs/decisions.md D26):

* waiting time: tripinfo ``waitingTime`` of **arrived** vehicles (the classic
  metric, which ignores vehicles that never finished);
* insertion delay: ``departDelay`` for inserted vehicles, ``end - intended
  departure`` for vehicles never inserted;
* total delay: insertion delay + ``timeLoss`` (time lost w.r.t. free flow) for
  arrived and unfinished vehicles, ``end - intended departure`` for never
  inserted ones, averaged over every vehicle due to depart before the end.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def output_args(directory: Path) -> tuple[list[str], dict[str, Path]]:
    """SUMO arguments writing the evaluation outputs into ``directory``."""
    files = {"tripinfo": directory / "tripinfo.xml", "summary": directory / "summary.xml"}
    args = [
        "--tripinfo-output", str(files["tripinfo"]),
        "--tripinfo-output.write-unfinished",
        "--device.emissions.probability", "1",
        "--summary-output", str(files["summary"]),
    ]  # fmt: skip
    return args, files


def tls_states_additional(tls_ids: list[str], directory: Path) -> tuple[Path, Path]:
    """Additional file making SUMO write every signal's state each second (ground truth)."""
    output = directory / "tls_states.xml"
    events = "".join(
        f'<timedEvent type="SaveTLSStates" source="{tls}" dest="{output}"/>' for tls in tls_ids
    )
    additional = directory / "tls_states.add.xml"
    additional.write_text(f"<additional>{events}</additional>", encoding="utf-8")
    return additional, output


def _iter_elements(path: Path, tag: str):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        for _, elem in ET.iterparse(fh):
            if elem.tag == tag:
                yield elem
                elem.clear()


def intended_departures(route_file: Path) -> dict[str, float]:
    """Vehicle id -> scheduled departure time, from a (possibly gzipped) route file."""
    return {v.get("id"): float(v.get("depart")) for v in _iter_elements(route_file, "vehicle")}


def trip_records(tripinfo: Path) -> list[dict[str, float]]:
    records = []
    for trip in _iter_elements(tripinfo, "tripinfo"):
        emissions = trip.find("emissions")
        records.append(
            {
                "id": trip.get("id"),
                "arrived": float(trip.get("arrival")) >= 0,
                "depart_delay": float(trip.get("departDelay")),
                "waiting": float(trip.get("waitingTime")),
                "duration": float(trip.get("duration")),
                "time_loss": float(trip.get("timeLoss")),
                "co2_mg": float(emissions.get("CO2_abs")) if emissions is not None else 0.0,
            }
        )
    return records


def network_queue(summary: Path) -> float:
    """Time-averaged number of halting vehicles in the whole network."""
    halting = [float(step.get("halting")) for step in _iter_elements(summary, "step")]
    return float(np.mean(halting)) if halting else 0.0


def signal_timing(
    intervals: dict[str, list[tuple[str, float, int, bool]]],
    green_states: dict[str, list[str]],
    end: float,
    begin: float = 0.0,
) -> dict[str, Any]:
    """Green-duration extremes and longest red per phase, from per-second signal states.

    The red of a phase is the time between two of its green intervals (or
    between the episode start / end and its first / last green).
    """
    longest_red: dict[str, float] = {}
    greens: list[int] = []
    n_runs: list[int] = []
    for tls, states in green_states.items():
        runs = intervals.get(tls, [])
        n_runs.append(len(runs))
        greens += [length for _, _, length, completed in runs if completed]
        phase_red = []
        for state in set(states):
            own = sorted((start, start + length) for s, start, length, _ in runs if s == state)
            if not own:
                phase_red.append(end - begin)
                continue
            gaps = [own[0][0] - begin, end - own[-1][1]]
            gaps += [nxt[0] - cur[1] for cur, nxt in zip(own, own[1:], strict=False)]
            phase_red.append(max(gaps))
        longest_red[tls] = max(phase_red) if phase_red else 0.0
    reds = list(longest_red.values())
    return {
        "green_min_s": float(min(greens)) if greens else float("nan"),
        "green_max_s": float(max(greens)) if greens else float("nan"),
        "longest_red_max_s": float(max(reds)) if reds else 0.0,
        "longest_red_mean_s": float(np.mean(reds)) if reds else 0.0,
        "longest_red_per_signal": longest_red,
        # Phase changes measured from SUMO's states, so programs and agents are comparable.
        "phase_changes_per_signal": float(np.mean([max(n - 1, 0) for n in n_runs]))
        if n_runs
        else 0.0,
    }


def delay_metrics(records: list[dict], departures: dict[str, float], end: float) -> dict[str, Any]:
    """Waiting, insertion delay, total delay, throughput and unfinished vehicles (D26)."""
    inserted = {r["id"] for r in records}
    due = {vid: t for vid, t in departures.items() if t < end}
    never_inserted = [end - t for vid, t in due.items() if vid not in inserted]
    arrived = [r for r in records if r["arrived"]]
    in_network = [r for r in records if not r["arrived"]]
    insertion = [r["depart_delay"] for r in records] + never_inserted
    total = [r["depart_delay"] + r["time_loss"] for r in records] + never_inserted

    def mean(values) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    return {
        "vehicles_due": len(due),
        "inserted": len(records),
        "arrived": len(arrived),
        "throughput": len(arrived),
        "unfinished_in_network": len(in_network),
        "never_inserted": len(never_inserted),
        "unfinished_total": len(in_network) + len(never_inserted),
        "mean_waiting_time": mean([r["waiting"] for r in arrived]),
        # Starvation seen by vehicles (arrived and unfinished): longest and 95th-percentile wait.
        "max_waiting_time": float(max((r["waiting"] for r in records), default=0.0)),
        "p95_waiting_time": (
            float(np.percentile([r["waiting"] for r in records], 95)) if records else 0.0
        ),
        "mean_travel_time": mean([r["duration"] for r in arrived]),
        "mean_time_loss": mean([r["time_loss"] for r in arrived]),
        "mean_insertion_delay": mean(insertion),
        "mean_total_delay": mean(total),
        # Tail of the total delay over every vehicle due (never-inserted ones included).
        "p95_total_delay": float(np.percentile(total, 95)) if total else 0.0,
        "total_delay_h": float(np.sum(total)) / 3600.0,
        "co2_total_kg": sum(r["co2_mg"] for r in records) / 1e6,
        "co2_per_vehicle_g": mean([r["co2_mg"] for r in records]) / 1e3,
    }
