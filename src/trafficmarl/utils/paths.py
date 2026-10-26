"""Canonical project paths.

Every module resolves files through this module instead of hard-coding paths,
so the project works from any working directory and on any machine.
"""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path

ROOT_ENV_VAR = "TRAFFICMARL_ROOT"


def _find_project_root() -> Path:
    """Locate the repository root.

    ``TRAFFICMARL_ROOT`` takes precedence; otherwise walk up from this file
    until a directory containing ``pyproject.toml`` is found. This works both
    for editable installs and for running from the source tree.
    """
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        f"Could not locate the project root from {here}; set {ROOT_ENV_VAR} explicitly."
    )


PROJECT_ROOT: Path = _find_project_root()
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
NETWORKS_DIR: Path = PROJECT_ROOT / "networks"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
EXTERNAL_DIR: Path = PROJECT_ROOT / "external"
DOCS_DIR: Path = PROJECT_ROOT / "docs"


def sumo_home() -> Path:
    """Return ``$SUMO_HOME``, failing loudly if SUMO is not configured."""
    value = os.environ.get("SUMO_HOME")
    if not value:
        raise OSError("SUMO_HOME is not set; see README.md for setup instructions.")
    return Path(value)


def sumo_tools_dir() -> Path:
    """Directory holding SUMO helper scripts (randomTrips.py, routeSampler.py, ...)."""
    return sumo_home() / "tools"


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def run_dir(experiment: str, method: str, seed: int, *, create: bool = True) -> Path:
    """Output directory for one run: ``results/<experiment>/<method>/<seed>/``."""
    path = RESULTS_DIR / experiment / method / str(seed)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def open_output(path: str | Path) -> io.BufferedIOBase:
    """Binary writer for ``path``, gzip-compressed if it ends with ``.gz``.

    The gzip header's timestamp and file name are fixed so that rebuilding a
    scenario yields byte-identical files (no spurious git diffs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    if path.suffix != ".gz":
        return raw
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9)
    compressed.myfileobj = raw  # let close() also close the underlying file, as gzip.open does
    return compressed


def open_input(path: str | Path) -> io.BufferedIOBase:
    """Binary reader for ``path``, transparently decompressing ``.gz`` files."""
    path = Path(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
