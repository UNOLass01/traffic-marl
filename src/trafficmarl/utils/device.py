"""Torch device selection from the ``device`` config key."""

from __future__ import annotations

VALID_DEVICES = ("auto", "cpu", "cuda")


def resolve_device(requested: str = "auto") -> str:
    """Map ``auto | cpu | cuda`` to a concrete device string.

    The resolved value (never ``auto``) is what each run stores in its saved
    config, so results always state where training actually ran.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got '{requested}'")
    if requested == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("device: cuda requested but CUDA is not available")
    return "cpu"
