"""Provenance written next to every run's results (resolved config, git commit)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from trafficmarl.utils import paths
from trafficmarl.utils.config import save_config


def git_commit() -> str:
    """Current commit hash, suffixed with ``-dirty`` when tracked files are modified."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=paths.PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = git("rev-parse", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def write_run_info(directory: Path, config: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_config(config, directory / "config.yaml")
    (directory / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
