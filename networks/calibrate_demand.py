"""Demand calibration sweep for the synthetic scenarios.

For each candidate entry rate, generate one constant-demand route file and
simulate it with SUMO's default fixed-time program. The knee of the mean
waiting time (and the first teleports) locates saturation; ``entry_rate`` in
``configs/scenarios/<name>.yaml`` is chosen so that ``high`` (x1.5) sits just
below it.

Usage (after ``python -m trafficmarl scenarios build <name>``):
    python networks/calibrate_demand.py grid3x3 450 600 750 900 1100 1300
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import sumolib

from trafficmarl.scenarios import demand
from trafficmarl.scenarios.spec import Manifest, load_scenario_config, output_dir
from trafficmarl.scenarios.validate import simulate

CALIBRATION_SEED = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario")
    parser.add_argument("rates", nargs="+", type=float, help="entry rates to test (veh/h)")
    args = parser.parse_args()

    cfg = load_scenario_config(args.scenario)
    built = Manifest.load(output_dir(cfg))
    net = sumolib.net.readNet(str(built.path(built.net_file)))
    dem = cfg["demand"]
    print("rate_veh_h,inserted,mean_waiting_s,mean_travel_s,teleports,running_at_end")
    with tempfile.TemporaryDirectory() as tmp:
        for rate in args.rates:
            vehicles = demand.generate_vehicles(
                net,
                entry_rate=rate,
                profile=[1.0],
                duration=dem["duration"],
                turn_probabilities=dem["turn_probabilities"],
                seed=CALIBRATION_SEED,
                entry_weights=dem.get("entry_weights"),
            )
            route = Path(tmp) / f"rate_{int(rate)}.rou.xml"
            demand.write_route_file(vehicles, route)
            stats = simulate(built, str(route), end=cfg["validation"]["end"])
            print(
                f"{rate:.0f},{stats.inserted},{stats.mean_waiting_time:.1f},"
                f"{stats.mean_travel_time:.1f},{stats.teleports},{stats.running_at_end}"
            )


if __name__ == "__main__":
    main()
