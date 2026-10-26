"""Build environments from a resolved experiment config."""

from __future__ import annotations

from typing import Any

from trafficmarl.envs.traffic_env import TrafficSignalEnv
from trafficmarl.scenarios.spec import Manifest, load_scenario_config, output_dir
from trafficmarl.utils import paths
from trafficmarl.utils.config import deep_merge, load_config


def experiment_config(scenario: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """``configs/base.yaml`` + one scenario (+ optional overrides), as an experiment would."""
    config = load_config(paths.CONFIGS_DIR / "base.yaml")
    config["scenario"] = load_scenario_config(scenario)
    return deep_merge(config, overrides or {})


def make_env(
    config: dict[str, Any],
    *,
    split: str = "train",
    seed: int = 0,
    extra_additional_files: list[str] | None = None,
) -> TrafficSignalEnv:
    """Environment for ``config['scenario']`` with the demand profile ``config['env']['profile']``.

    ``split='train'`` samples a training route file at each reset; ``'eval'``
    cycles through the held-out evaluation files in a fixed order.
    """
    scenario = config["scenario"]
    manifest = Manifest.load(output_dir(scenario))
    return TrafficSignalEnv(
        manifest,
        profile=config["env"]["profile"],
        split=split,
        env_cfg=config["env"],
        sim_cfg=config["simulation"],
        yellow_time=int(scenario["signals"]["yellow_time"]),
        seed=seed,
        extra_additional_files=extra_additional_files,
        reward_scale=scenario.get("reward_scale"),
    )
