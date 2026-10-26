"""Environment building blocks that need no running simulation."""

import numpy as np
import pytest

from trafficmarl.envs import features
from trafficmarl.envs.constraints import InvalidActionError, PhaseConstraints
from trafficmarl.envs.measure import ApproachStats
from trafficmarl.envs.policies import make_policy
from trafficmarl.envs.topology import bearing_slot, build_topology
from trafficmarl.scenarios.spec import Manifest, load_scenario_config, output_dir

DELTA, YELLOW, MIN_G, MAX_G = 5, 3, 5, 50


def constraints(n_green=4, n_actions=4, max_red=None):
    pc = PhaseConstraints(
        {"a": n_green}, n_actions, delta_time=DELTA, yellow_time=YELLOW,
        min_green=MIN_G, max_green=MAX_G, max_red=max_red,
    )  # fmt: skip
    pc.reset()
    return pc


# ----------------------------------------------------------------- constraints
def test_min_green_allows_only_keep():
    pc = constraints()
    # SUMO-RL refuses a change before yellow + min_green seconds.
    assert pc.mask("a", now=0, time_since_change=0).tolist() == [1, 0, 0, 0]
    assert pc.mask("a", now=5, time_since_change=5).tolist() == [1, 0, 0, 0]
    assert pc.mask("a", now=10, time_since_change=10).tolist() == [1, 1, 1, 1]


def test_max_green_masks_keep():
    pc = constraints()
    # Green began at t=0: keeping at t=45 would reach 50 s (allowed), at t=50 55 s (not).
    assert pc.mask("a", now=45, time_since_change=45)[0] == 1
    assert pc.mask("a", now=50, time_since_change=50).tolist() == [0, 1, 1, 1]


def test_padded_phases_always_masked():
    pc = constraints(n_green=2, n_actions=4)
    assert pc.mask("a", now=20, time_since_change=20).tolist() == [1, 1, 0, 0]


def test_masked_action_raises_instead_of_being_ignored():
    pc = constraints()
    with pytest.raises(InvalidActionError):
        pc.apply("a", 2, now=0, time_since_change=0)
    with pytest.raises(InvalidActionError):
        pc.apply("a", 0, now=50, time_since_change=50)


def test_switch_bookkeeping_and_counters():
    pc = constraints()
    assert pc.apply("a", 0, now=0, time_since_change=0) is False
    assert pc.apply("a", 2, now=10, time_since_change=10) is True
    st = pc.state["a"]
    assert (st.green_phase, st.green_start) == (2, 10 + YELLOW)
    # Phase 2 was red from 0 until its green at 13 s.
    assert pc.longest_red("a", now=13)[2] == 13
    # Phase 0 is red since the end of its yellow (13 s).
    assert pc.longest_red("a", now=40)[0] == 27
    counters = pc.counters("a")
    assert counters["decisions"] == 2 and counters["switches"] == 1
    assert counters["min_green_active"] == 1


def test_forced_switch_counted():
    pc = constraints()
    pc.apply("a", 1, now=50, time_since_change=50)
    assert pc.counters("a")["max_green_forced"] == 1


def test_optional_max_red_forces_starved_phase():
    pc = constraints(max_red=30)
    mask = pc.mask("a", now=25, time_since_change=25)
    # Phases 1-3 have been red since 0; 25 + 5 + 3 > 30 -> only they are valid.
    assert mask.tolist() == [0, 1, 1, 1]


def test_constraint_config_validated():
    with pytest.raises(ValueError):
        PhaseConstraints({"a": 2}, 2, delta_time=5, yellow_time=3, min_green=5, max_green=8)
    with pytest.raises(ValueError):
        PhaseConstraints({"a": 2}, 2, delta_time=5, yellow_time=5, min_green=5, max_green=50)


# ---------------------------------------------------------------- policies
def test_heuristic_policies_respect_mask():
    obs = {"a": {"action_mask": np.array([0, 1, 1, 0], dtype=np.int8)}}
    for name in ("random", "keep", "switch"):
        action = make_policy(name)(obs, {"a": 0})["a"]
        assert obs["a"]["action_mask"][action]
    keep_obs = {"a": {"action_mask": np.array([1, 0, 0, 0], dtype=np.int8)}}
    assert make_policy("switch")(keep_obs, {"a": 0})["a"] == 0


# ------------------------------------------------------------------ rewards
def _stats(waiting=0.0, halting=0.0, count=(0.0,), outgoing=0.0):
    count = np.asarray(count)
    return ApproachStats(
        queue=count, count=count, speed=np.full(len(count), np.nan),
        max_speed=np.full(len(count), 13.89), waiting=waiting, halting_total=halting,
        outgoing=outgoing,
    )  # fmt: skip


def _spec(**kw):
    defaults = dict(
        agents=["a"], topology={}, n_green={"a": 4}, max_lanes=12, max_phases=4,
        max_distance=200.0, max_green=50, extended=False, agent_id=False,
    )  # fmt: skip
    defaults.update(kw)
    return features.FeatureSpec(**defaults)


def test_waiting_reward_sign_and_scale():
    spec = _spec()
    value, ref = features.traffic_term(spec, _stats(waiting=300.0), "waiting", 100.0, 100.0)
    assert value == pytest.approx(-2.0) and ref == 300.0  # more waiting -> negative
    value, _ = features.traffic_term(spec, _stats(waiting=50.0), "waiting", 100.0, 300.0)
    assert value > 0  # queue cleared -> positive


def test_queue_and_pressure_rewards_non_positive():
    spec = _spec()
    q, _ = features.traffic_term(spec, _stats(halting=10.0), "queue", 100.0, 0.0)
    p, _ = features.traffic_term(spec, _stats(count=(5.0, 5.0), outgoing=2.0), "pressure", 100, 0)
    assert q == pytest.approx(-10.0 / spec.capacity) and p == pytest.approx(-8.0 / spec.capacity)
    with pytest.raises(ValueError):
        features.traffic_term(spec, _stats(), "speed", 100.0, 0.0)


def test_reward_mixing_formulas():
    traffic = {"a": -1.0, "b": -3.0, "c": 2.0}
    penalty = {"a": 0.1, "b": 0.0, "c": 0.0}
    nbrs = {"a": ["b", "c"], "b": ["a"], "c": []}
    local = features.mix_rewards(traffic, penalty, "local", nbrs, 0.5)
    assert local == pytest.approx({"a": -1.1, "b": -3.0, "c": 2.0})
    hood = features.mix_rewards(traffic, penalty, "neighborhood", nbrs, 0.5)
    assert hood["a"] == pytest.approx(-1.0 + 0.5 * (-0.5) - 0.1)
    assert hood["c"] == pytest.approx(2.0)  # no neighbours -> local term only
    glob = features.mix_rewards(traffic, penalty, "global", nbrs, 0.5)
    assert glob == pytest.approx({"a": -2 / 3 - 0.1, "b": -2 / 3, "c": -2 / 3})
    with pytest.raises(ValueError):
        features.mix_rewards(traffic, penalty, "selfish", nbrs, 0.5)


def test_lane_features_normalised_to_unit_interval():
    spec = _spec()
    stats = ApproachStats(
        queue=np.array([0.0, 40.0]), count=np.array([0.0, 60.0]), speed=np.array([np.nan, 20.0]),
        max_speed=np.array([13.89, 13.89]), waiting=0, halting_total=0, outgoing=0,
    )  # fmt: skip
    queue, density, speed = features.lane_features(spec, stats)
    assert queue.tolist() == [0.0, 1.0] and density.tolist() == [0.0, 1.0]  # clipped
    assert speed.tolist() == [1.0, 1.0]  # empty lane = free flow


def test_observation_dimensions():
    spec = _spec(agents=["a", "b"])
    assert spec.local_dim == 3 * 12 + 4 + 2 + 4 + 1
    assert _spec(extended=True).obs_dim == spec.local_dim + 4 * (2 + 4 + 1)
    assert _spec(agent_id=True, agents=["a", "b"]).local_dim == spec.local_dim + 2


# ----------------------------------------------------------------- topology
def _topology(name, max_distance=200.0):
    manifest = Manifest.load(output_dir(load_scenario_config(name)))
    return build_topology(str(manifest.path(manifest.net_file)), max_distance)


@pytest.mark.parametrize(
    ("origin", "target", "slot"),
    [
        ((0, 0), (0, 10), "N"),
        ((0, 0), (10, 1), "E"),
        ((0, 0), (-1, -10), "S"),
        ((0, 0), (-10, 0), "W"),
    ],
)
def test_bearing_slot(origin, target, slot):
    assert bearing_slot(origin, target) == slot


def test_grid_neighbours_and_slots():
    topo = _topology("grid3x3")
    counts = sorted(len(t.neighbours) for t in topo.values())
    assert counts == [2, 2, 2, 2, 3, 3, 3, 3, 4]  # corners, edges, centre
    centre = topo["B1"]
    assert {slot: ids for slot, ids in centre.slots.items()} == {
        "N": ["B2"], "E": ["C1"], "S": ["B0"], "W": ["A1"],
    }  # fmt: skip


@pytest.mark.parametrize("name", ["grid3x3", "lust"])
def test_approaches_capped_and_weights_consistent(name):
    topo = _topology(name)
    for t in topo.values():
        assert all(0 < length <= 200.0 + 1e-6 for length in t.approach_length.values())
        # Every upstream lane part is shared out in full among the stop lanes it feeds.
        totals: dict = {}
        for segments in t.approach.values():
            for seg in segments:
                totals[(seg.lane, seg.start)] = totals.get((seg.lane, seg.start), 0.0) + seg.weight
        assert all(v == pytest.approx(1.0) for v in totals.values())


def test_neighbour_graph_symmetric_on_lust():
    topo = _topology("lust")
    for tls, t in topo.items():
        for other in t.neighbours:
            assert tls in topo[other].neighbours


def test_shorter_max_distance_shortens_approaches():
    short = _topology("grid3x3", max_distance=50.0)
    assert all(v <= 50.0 + 1e-6 for t in short.values() for v in t.approach_length.values())


@pytest.mark.parametrize("name", ["single", "corridor", "grid3x3", "grid5x5", "lust"])
def test_reward_scale_recorded_for_every_measure(name):
    scale = load_scenario_config(name)["reward_scale"]
    assert set(scale) == set(features.REWARD_MEASURES)
    assert all(v > 0 for v in scale.values())
