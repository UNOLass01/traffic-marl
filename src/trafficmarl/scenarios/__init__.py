"""SUMO scenarios: network generation, demand, LuST cropping and validation."""

from __future__ import annotations

from trafficmarl.scenarios.spec import Manifest, load_scenario_config


def build_scenario(name: str) -> Manifest:
    """Build one scenario from ``configs/scenarios/<name>.yaml``."""
    cfg = load_scenario_config(name)
    if cfg["kind"] == "synthetic":
        from trafficmarl.scenarios import synthetic

        return synthetic.build(cfg)
    if cfg["kind"] == "lust":
        from trafficmarl.scenarios import lust

        return lust.build(cfg)
    raise ValueError(f"Unknown scenario kind '{cfg['kind']}' in {name}")
