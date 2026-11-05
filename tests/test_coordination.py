"""Communication volume of the coordination schemes."""

import pytest

from trafficmarl.coordination.communication import floats_per_step

# A line of 3 lights: A-B-C has 4 directed neighbour links (A>B, B>A, B>C, C>B).
LINE = {"n_agents": 3, "n_links": 4, "message_dim": 6}


def test_independent_agents_exchange_nothing():
    assert floats_per_step(**LINE, extended=False, reward_type="local") == {
        "execution": 0,
        "training": 0,
    }


def test_shared_rewards_cost_only_during_training():
    neighborhood = floats_per_step(**LINE, extended=False, reward_type="neighborhood")
    assert neighborhood == {"execution": 0, "training": 4}  # one float per directed link
    global_ = floats_per_step(**LINE, extended=False, reward_type="global")
    assert global_ == {"execution": 0, "training": 6}  # send + receive through an aggregator


def test_neighbour_observation_costs_at_execution_too():
    out = floats_per_step(**LINE, extended=True, reward_type="neighborhood")
    assert out == {"execution": 4 * 6, "training": 4 * 6 + 4}


def test_unknown_reward_type_rejected():
    with pytest.raises(ValueError):
        floats_per_step(**LINE, extended=False, reward_type="team")
