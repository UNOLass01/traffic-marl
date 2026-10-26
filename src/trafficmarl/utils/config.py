"""YAML configuration loading.

An experiment is fully described by one YAML file. To avoid copy-pasting
scenario and algorithm settings, a config may list other configs under the
reserved key ``defaults``; they are merged first (in order) and the file's own
keys override them. Command-line overrides (``key.sub=value``) are applied last.
The fully resolved config is what gets saved next to each run's results.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from trafficmarl.utils import paths

DEFAULTS_KEY = "defaults"


class ConfigError(ValueError):
    """Raised for malformed configuration files or overrides."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested mappings are merged key by key; any other value (including lists)
    in ``override`` replaces the one in ``base``.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _resolve_default(entry: str, parent: Path) -> Path:
    """``defaults`` entries are relative to the including file, else to ``configs/``."""
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate
    local = parent / candidate
    return local if local.is_file() else paths.CONFIGS_DIR / candidate


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ConfigError(f"Circular 'defaults' include: {chain}")
    data = _read_yaml(path)
    includes = data.pop(DEFAULTS_KEY, []) or []
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f"'{DEFAULTS_KEY}' in {path} must be a list of paths")

    merged: dict[str, Any] = {}
    for entry in includes:
        included = _load_recursive(_resolve_default(str(entry), path.parent), (*stack, path))
        merged = deep_merge(merged, included)
    return deep_merge(merged, data)


def parse_override(override: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value``; the value is interpreted as YAML (``3``, ``true``, ``[1, 2]``)."""
    if "=" not in override:
        raise ConfigError(f"Override must look like key.path=value, got '{override}'")
    key, raw_value = override.split("=", 1)
    keys = [k for k in key.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"Empty key in override '{override}'")
    return keys, yaml.safe_load(raw_value)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``config`` with dotted-path overrides applied."""
    result = copy.deepcopy(dict(config))
    for override in overrides:
        keys, value = parse_override(override)
        node = result
        for k in keys[:-1]:
            child = node.setdefault(k, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Cannot override '{override}': '{k}' is not a mapping")
            node = child
        node[keys[-1]] = value
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving ``defaults`` includes and applying overrides."""
    config = _load_recursive(paths.resolve(path), stack=())
    return apply_overrides(config, overrides)


def get_by_path(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch ``config['a']['b']`` via ``'a.b'``; return ``default`` if any level is missing."""
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write the resolved config as YAML (used to make every run self-describing)."""
    target = paths.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(config), fh, sort_keys=False)
    return target
