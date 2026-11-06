"""CTDE methods: validation on small known cooperative tasks (D2) and their building blocks."""

import multiprocessing
import shutil

import numpy as np
import pytest
import torch

from trafficmarl.agents.maddpg import MADDPGLearner
from trafficmarl.agents.offpolicy import exploration_level
from trafficmarl.agents.qmix import MonotonicMixer, QMIXLearner
from trafficmarl.agents.replay import ReplayBuffer
from trafficmarl.coordination.communication import centralized_floats
from trafficmarl.envs.duration import duration_to_phase, target_green

requires_sumo = pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")

QMIX_CFG = {
    "hidden": [32], "mixer_embed": 16, "mixer_hyper": 32, "lr": 1e-3, "gamma": 0.99,
    "grad_clip": 10.0, "target_update_every": 50, "epsilon_start": 1.0, "epsilon_end": 1.0,
    "exploration_steps": 1, "reward_scale": 1.0,
}  # fmt: skip
MADDPG_CFG = {
    "hidden": [64, 64], "actor_lr": 1e-3, "critic_lr": 1e-3, "gamma": 0.0, "tau": 0.01,
    "grad_clip": 1.0, "agents_per_update": 2, "noise_start": 0.2, "noise_end": 0.2,
    "exploration_steps": 1, "learning_starts": 0, "reward_scale": 1.0,
}  # fmt: skip


def test_mixer_is_monotonic_in_every_utility():
    mixer = MonotonicMixer(n_agents=3, state_dim=5)
    state = torch.rand(8, 5)
    utilities = torch.rand(8, 3, requires_grad=True)
    mixer(utilities, state).sum().backward()
    assert (utilities.grad >= 0).all()


def _two_step_game(learner: QMIXLearner) -> float:
    """The two-step game of Rashid et al. (2018, section 6.1); returns the greedy team reward.

    In state 0, agent 0's action selects state 1 (every joint action pays 7) or state 2
    (pays [[0, 1], [1, 8]]). The optimum, 8, needs both agents to coordinate in state 2,
    and additive value decompositions (VDN) settle for 7.
    """
    payoff = {1: np.full((2, 2), 7.0), 2: np.array([[0.0, 1.0], [1.0, 8.0]])}
    states = np.eye(3, dtype=np.float32)
    joint = lambda s: np.stack([states[s], states[s]])  # noqa: E731
    all_valid = np.ones((2, 2), dtype=bool)
    buffer = ReplayBuffer(2000, 2, 3, 2, seed=0)
    for _ in range(1000):
        a = learner.act(joint(0)[None], all_valid[None], 0)[0]  # epsilon = 1: uniform behaviour
        next_state = 1 if a[0] == 0 else 2
        buffer.add(joint(0), all_valid, a, np.zeros(2), joint(next_state), all_valid, False)
        b = learner.act(joint(next_state)[None], all_valid[None], 0)[0]
        reward = np.full(2, payoff[next_state][b[0], b[1]], dtype=np.float32)
        buffer.add(joint(next_state), all_valid, b, reward, joint(0), all_valid, True)
    for _ in range(3000):
        learner.update(buffer.sample(64))

    def greedy(state: int) -> np.ndarray:
        with torch.no_grad():
            return learner.agent(torch.as_tensor(joint(state))).argmax(-1).numpy()

    chosen = 1 if greedy(0)[0] == 0 else 2
    a = greedy(chosen)
    return float(payoff[chosen][a[0], a[1]])


def test_qmix_solves_the_two_step_game():
    torch.manual_seed(0)
    learner = QMIXLearner(2, 3, 2, QMIX_CFG, seed=0)
    assert _two_step_game(learner) == 8.0


def test_qmix_respects_the_mask():
    learner = QMIXLearner(2, 3, 4, {**QMIX_CFG, "epsilon_start": 0.5, "epsilon_end": 0.5}, seed=0)
    mask = np.zeros((64, 2, 4), dtype=bool)
    mask[:, :, [1, 3]] = True
    actions = learner.act(np.random.rand(64, 2, 3).astype(np.float32), mask, 0)
    assert set(np.unique(actions)) <= {1, 3}


def test_maddpg_solves_a_continuous_coordination_game():
    """Both agents see a target and are paid -(a0 + a1 - target)^2: the sum must match it."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    learner = MADDPGLearner(2, 1, 1, MADDPG_CFG, seed=0)
    buffer = ReplayBuffer(2000, 2, 1, 1, action_dtype=np.float32, seed=0)
    # Closed loop: MADDPG's actor answers the partner's actions found in replay, so these
    # must come from the partner's current behaviour, not from a fixed random policy.
    mask = np.ones((2, 1), bool)
    for step in range(6000):
        target = rng.uniform(-1, 1)
        obs = np.full((2, 1), (target + 1) / 2, dtype=np.float32)
        action = learner.act(obs[None], mask[None], step)[0]
        reward = np.full(2, -((action.sum() - target) ** 2), dtype=np.float32)
        buffer.add(obs, mask, action, reward, obs, mask, True)
        if step > 200:
            learner.update(buffer.sample(64))
    targets = np.linspace(-1, 1, 21, dtype=np.float32)
    obs = torch.as_tensor((targets[:, None] + 1) / 2)
    with torch.no_grad():
        sums = 2 * learner.policy(obs).numpy()  # shared actor: both agents output the same
    assert np.abs(sums - targets).mean() < 0.15


def test_learners_restore_their_state():
    source = QMIXLearner(2, 3, 2, QMIX_CFG, seed=0)
    copy = QMIXLearner(2, 3, 2, QMIX_CFG, seed=1)
    copy.load_state_dict(source.state_dict())
    for a, b in zip(source.agent.parameters(), copy.agent.parameters(), strict=True):
        assert torch.equal(a, b)


def test_replay_wraps_and_stores_float16_observations():
    buffer = ReplayBuffer(3, 2, 2, 2, seed=0)
    for i in range(5):
        obs = np.full((2, 2), i / 10, dtype=np.float32)
        buffer.add(
            obs, np.ones((2, 2), bool), [0, 1], [1.0, 2.0], obs, np.ones((2, 2), bool), False
        )
    assert buffer.size == 3
    assert buffer.obs.dtype == np.float16
    assert buffer.sample(16)["obs"].dtype == np.float32
    assert sorted(np.round(buffer.obs[:, 0, 0].astype(float), 2)) == [0.2, 0.3, 0.4]


def test_exploration_level_anneals_then_stays():
    assert exploration_level(0, 1.0, 0.1, 100) == 1.0
    assert exploration_level(50, 1.0, 0.0, 100) == 0.5
    assert exploration_level(500, 1.0, 0.1, 100) == pytest.approx(0.1)


def test_target_green_covers_the_allowed_range():
    assert target_green(-1.0, 5, 50) == 5
    assert target_green(1.0, 5, 50) == 50
    assert target_green(3.0, 5, 50) == 50  # clipped


@pytest.mark.parametrize(
    ("action", "age", "mask", "expected"),
    [
        (1.0, 10, [1, 1, 1, 1], 0),  # target 50 s not reached: keep
        (-1.0, 10, [1, 1, 1, 1], 1),  # target 5 s passed: next phase in the cycle
        (-1.0, 3, [1, 0, 0, 0], 0),  # minimum green: only "keep" is valid
        (1.0, 46, [0, 1, 1, 1], 1),  # maximum green: "keep" is masked
        (1.0, 10, [0, 0, 0, 1], 3),  # maximum red: the overdue phase, whatever the target
        (-1.0, 10, [1, 0, 1, 0], 2),  # skips masked phases (e.g. padded ones)
    ],
)
def test_duration_action_obeys_the_mask(action, age, mask, expected):
    phase = duration_to_phase(
        action, current=0, mask=np.array(mask), green_age=age, min_green=5, max_green=50
    )
    assert phase == expected


def test_centralized_floats_by_method():
    size = {"n_agents": 4, "obs_dim": 47, "message_dim": 6}
    assert centralized_floats("ippo", **size) == {"execution": 0, "training": 0}
    assert centralized_floats("mappo", **size) == {"execution": 0, "training": 48}
    assert centralized_floats("qmix", **size) == {"execution": 0, "training": 196}
    assert centralized_floats("centralized_ppo", **size)["execution"] == 192


def _summaries_match_messages() -> bool:
    from trafficmarl.agents.mappo import summarize_observation
    from trafficmarl.envs import features
    from trafficmarl.envs.factory import experiment_config, make_env

    env = make_env(experiment_config("corridor", {"env": {"profile": "low"}}), seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        actions = {a: int(np.flatnonzero(o["action_mask"])[0]) for a, o in obs.items()}
        obs, *_ = env.step(actions)
    ok = True
    for agent in env.agents:
        stats = env.meter.stats(env.sumo_env.sumo, agent)
        expected = features.message(env.spec, stats, env.green_phases()[agent])
        got = summarize_observation(env.spec, agent, obs[agent]["observations"])
        ok = ok and bool(np.allclose(expected, got, atol=1e-5))
    env.close()
    return ok


@requires_sumo
@pytest.mark.sumo
def test_mappo_state_summary_equals_the_neighbour_message():
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        assert pool.apply(_summaries_match_messages)
