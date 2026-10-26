"""Global seeding for reproducible runs.

Seeds Python, NumPy and (if installed) PyTorch. SUMO is a separate process, so
its seed must be passed on the command line: use :func:`sumo_seed_args`.
"""

from __future__ import annotations

import os
import random

import numpy as np

MAX_SEED = 2**32 - 1


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> int:
    """Seed every random number generator used in the project.

    ``deterministic_torch`` forces deterministic cuDNN kernels; it makes GPU
    runs bit-reproducible at a speed cost, so it is opt-in.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be in [0, {MAX_SEED}], got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def derive_seeds(base_seed: int, n: int) -> list[int]:
    """Derive ``n`` independent, reproducible seeds from one base seed.

    Used to give each worker / route file / evaluation episode its own seed
    without the correlated streams that ``base_seed + i`` can produce.
    """
    seq = np.random.SeedSequence(base_seed)
    return [int(s.generate_state(1)[0]) for s in seq.spawn(n)]


def sumo_seed_args(seed: int) -> list[str]:
    """SUMO command-line arguments that fix its internal RNG."""
    return ["--seed", str(seed)]
