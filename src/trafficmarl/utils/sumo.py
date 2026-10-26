"""Thin wrappers around SUMO executables and the Python tools shipped in ``$SUMO_HOME/tools``."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from trafficmarl.utils import paths
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SumoCommandError(RuntimeError):
    """A SUMO binary or tool exited with a non-zero status."""


def binary(name: str) -> str:
    """Absolute path of a SUMO binary (``sumo``, ``netconvert``, ``netgenerate``...)."""
    found = shutil.which(name)
    if found:
        return found
    candidate = paths.sumo_home() / "bin" / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"SUMO binary '{name}' not found on PATH or in $SUMO_HOME/bin")


def tool(relative_path: str) -> list[str]:
    """Command prefix that runs a SUMO Python tool with the current interpreter."""
    script = paths.sumo_tools_dir() / relative_path
    if not script.is_file():
        raise FileNotFoundError(f"SUMO tool not found: {script}")
    return [sys.executable, str(script)]


def run(cmd: Sequence[str | Path], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, logging it, and raise with its output if it fails."""
    args = [str(c) for c in cmd]
    logger.debug("running: %s", " ".join(args))
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SumoCommandError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    return result
