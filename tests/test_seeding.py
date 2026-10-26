import random

import numpy as np
import pytest
import torch

from trafficmarl.utils.seeding import derive_seeds, set_global_seed, sumo_seed_args


def draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_same_numbers():
    set_global_seed(123)
    first = draw()
    set_global_seed(123)
    assert draw() == first


def test_different_seed_different_numbers():
    set_global_seed(1)
    first = draw()
    set_global_seed(2)
    assert draw() != first


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_out_of_range_seed_rejected(seed):
    with pytest.raises(ValueError):
        set_global_seed(seed)


def test_derive_seeds_reproducible_and_distinct():
    seeds = derive_seeds(42, 5)
    assert seeds == derive_seeds(42, 5)
    assert len(set(seeds)) == 5
    assert seeds != derive_seeds(43, 5)


def test_sumo_seed_args():
    assert sumo_seed_args(7) == ["--seed", "7"]
