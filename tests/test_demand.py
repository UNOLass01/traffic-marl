import numpy as np
import pytest

from trafficmarl.scenarios.demand import (
    entry_weight,
    poisson_departures,
    rate_multiplier,
)
from trafficmarl.scenarios.synthetic import profile_seed


def test_rate_multiplier_constant_and_interpolated():
    assert rate_multiplier([1.5], 123.0, 3600) == 1.5
    profile = [0.5, 1.5, 0.5]
    assert rate_multiplier(profile, 0, 3600) == pytest.approx(0.5)
    assert rate_multiplier(profile, 1800, 3600) == pytest.approx(1.5)
    assert rate_multiplier(profile, 900, 3600) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rate_multiplier([], 0, 3600)


def test_poisson_count_matches_rate():
    rng = np.random.default_rng(0)
    counts = [len(poisson_departures(600, [1.0], 3600, rng)) for _ in range(200)]
    # E = 600, std of the mean over 200 draws ~ 1.7
    assert np.mean(counts) == pytest.approx(600, abs=10)


def test_poisson_departures_sorted_in_range_and_seeded():
    a = poisson_departures(900, [1.0], 3600, np.random.default_rng(1))
    b = poisson_departures(900, [1.0], 3600, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(a) >= 0)
    assert a.min() >= 0 and a.max() < 3600


def test_time_varying_profile_concentrates_departures_at_peak():
    rng = np.random.default_rng(2)
    t = np.concatenate([poisson_departures(1200, [0.2, 2.0, 0.2], 3600, rng) for _ in range(20)])
    middle = np.sum((t > 1200) & (t < 2400))
    edges = np.sum(t < 1200) + np.sum(t > 2400)
    assert middle > edges


def test_zero_rate_gives_no_vehicle():
    assert len(poisson_departures(0, [1.0], 3600, np.random.default_rng(0))) == 0


def test_entry_weight_longest_prefix_wins():
    weights = {"left": 3.0, "left0": 2.0}
    assert entry_weight("left0A0", weights) == 2.0
    assert entry_weight("left1A1", weights) == 3.0
    assert entry_weight("top0A0", weights) == 1.0


def test_profile_seed_differs_per_profile_and_split():
    seeds = {profile_seed(s, p) for s in (1001, 2001) for p in ("low", "medium", "high")}
    assert len(seeds) == 6
    assert profile_seed(1001, "low") == profile_seed(1001, "low")
