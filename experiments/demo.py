"""Watch a controller in the SUMO GUI, for the demo video (fixed-time vs best MARL).

Runs one deterministic episode on the first held-out route file of a profile, with the
same settings as the evaluation. SUMO's GUI needs the socket API, not libsumo:

    unset LIBSUMO_AS_TRACI
    python experiments/demo.py --scenario grid3x3 --profile peak --method fixed_time
    python experiments/demo.py --scenario grid3x3 --profile peak --method ippo_extended --seed 1

Learned methods load ``results/marl_<scenario>/<method>/<seed>/checkpoint``. Set the GUI
delay (top bar) to about 50 ms to make the video readable; record with any screen recorder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heatmap import _spec_and_config  # noqa: E402
from trafficmarl.utils.config import deep_merge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="grid3x3")
    parser.add_argument("--profile", default="peak")
    parser.add_argument("--method", default="fixed_time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gui", action="store_true", help="headless check of the setup")
    args = parser.parse_args()
    if os.environ.get("LIBSUMO_AS_TRACI") and not args.no_gui:
        sys.exit("libsumo has no GUI: run `unset LIBSUMO_AS_TRACI` first")

    from trafficmarl.envs.factory import make_env
    from trafficmarl.metrics.evaluate import _make_policy, _resolve_additional
    from trafficmarl.scenarios.spec import Manifest, output_dir

    spec, config = _spec_and_config(args.scenario, args.profile, args.method, args.seed)
    overrides = {"simulation": {"gui": not args.no_gui}, "env": spec.get("env", {})}
    config = deep_merge(config, overrides)
    manifest = Manifest.load(output_dir(config["scenario"]))
    env = make_env(
        config,
        split="eval",
        seed=args.seed,
        extra_additional_files=_resolve_additional(spec, manifest, args.profile),
        control=spec["control"],
    )
    obs, _ = env.reset(seed=args.seed, options={"route_index": 0, "sumo_seed": args.seed})
    policy = _make_policy(spec, env, args.seed) if spec["control"] == "agents" else None
    while env.agents:
        obs, *_ = env.step(policy(obs, env) if policy else None)
    env.close()


if __name__ == "__main__":
    main()
