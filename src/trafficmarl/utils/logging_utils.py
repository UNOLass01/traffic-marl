"""Project-wide logging setup (console + optional log file per run)."""

from __future__ import annotations

import logging
from pathlib import Path

PACKAGE_LOGGER = "trafficmarl"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the package logger.

    Only the ``trafficmarl`` logger is configured, so third-party libraries
    (Ray, SUMO-RL) keep their own verbosity. Calling it again replaces the
    previous handlers instead of duplicating output.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger, e.g. ``get_logger(__name__)``."""
    if name == PACKAGE_LOGGER or name.startswith(PACKAGE_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")
