"""Scenario configs and the manifest describing a built scenario.

A scenario is defined by ``configs/scenarios/<name>.yaml`` (top-level key
``scenario``). Building it writes SUMO files into ``scenario.output_dir`` plus a
``manifest.yaml`` that lists every file with paths relative to that directory.
Everything downstream (env factory, baselines, validation) reads the manifest
rather than guessing file names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trafficmarl.utils import paths
from trafficmarl.utils.config import load_config

SCENARIO_CONFIG_DIR = paths.CONFIGS_DIR / "scenarios"
MANIFEST_NAME = "manifest.yaml"
SPLITS = ("train", "eval")


def available_scenarios() -> list[str]:
    """Scenario names; files starting with "_" hold shared defaults only."""
    return sorted(p.stem for p in SCENARIO_CONFIG_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def load_scenario_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load the ``scenario`` section of a scenario config, by name or by path."""
    candidate = Path(name_or_path)
    if candidate.suffix not in (".yaml", ".yml"):
        candidate = SCENARIO_CONFIG_DIR / f"{name_or_path}.yaml"
    config = load_config(candidate)
    if "scenario" not in config:
        raise ValueError(f"{candidate} has no top-level 'scenario' section")
    return config["scenario"]


def output_dir(scenario_cfg: dict[str, Any]) -> Path:
    return paths.resolve(scenario_cfg["output_dir"])


def route_file_name(profile: str, split: str, index: int) -> str:
    """Route files are gzip-compressed (read natively by SUMO) to keep the repository small."""
    return f"routes/{profile}_{split}_{index}.rou.xml.gz"


@dataclass
class Manifest:
    """Files and facts of a built scenario (paths relative to ``root``)."""

    name: str
    root: Path
    net_file: str
    tls_ids: list[str]
    begin: int
    duration: int
    routes: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    vehicles: dict[str, int] = field(default_factory=dict)
    additional_files: list[str] = field(default_factory=list)
    sumocfg: str | None = None
    actuated_file: str | None = None  # additional file switching every TLS to actuated
    sumo_version: str = ""
    # Demand levels used only by the robustness study, never in the main comparison tables.
    robustness_only_profiles: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def path(self, relative: str) -> Path:
        return self.root / relative

    def route_files(self, profile: str, split: str) -> list[Path]:
        return [self.path(r) for r in self.routes.get(profile, {}).get(split, [])]

    def save(self) -> Path:
        data = {k: v for k, v in self.__dict__.items() if k != "root"}
        target = self.root / MANIFEST_NAME
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        return target

    @classmethod
    def load(cls, root: str | Path) -> Manifest:
        root = Path(root)
        target = root / MANIFEST_NAME
        if not target.is_file():
            raise FileNotFoundError(
                f"{target} not found; build the scenario first "
                "(python -m trafficmarl scenarios build <name>)"
            )
        with target.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(root=root, **data)


def write_sumocfg(manifest: Manifest, route: str, end: int) -> str:
    """A ``.sumocfg`` for opening one route file in ``sumo-gui`` (demo, inspection)."""
    additional = ""
    if manifest.additional_files:
        additional = f'        <additional-files value="{",".join(manifest.additional_files)}"/>\n'
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<configuration>\n    <input>\n'
        f'        <net-file value="{manifest.net_file}"/>\n'
        f'        <route-files value="{route}"/>\n{additional}'
        "    </input>\n    <time>\n"
        f'        <begin value="{manifest.begin}"/>\n        <end value="{end}"/>\n'
        "    </time>\n</configuration>\n"
    )
    name = f"{manifest.name}.sumocfg"
    (manifest.root / name).write_text(text, encoding="utf-8")
    return name
