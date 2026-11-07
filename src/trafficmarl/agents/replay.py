"""Uniform replay buffer of joint transitions for the off-policy MARL methods."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Ring buffer of ``(obs, mask, action, reward, next_obs, next_mask, terminated)``.

    Observations of all agents are kept together because the CTDE critics and the
    QMIX mixer need the joint state. They are stored as ``float16``: all features are
    in [0, 1], where this precision is enough, and it halves the memory, which
    matters at 25 agents. ``terminated`` excludes time-limit truncations, so that
    targets still bootstrap there.
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        action_dtype: type = np.int64,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.size = 0
        self._next = 0
        self._rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float16)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float16)
        self.mask = np.zeros((capacity, n_agents, n_actions), dtype=bool)
        self.next_mask = np.zeros((capacity, n_agents, n_actions), dtype=bool)
        self.action = np.zeros((capacity, n_agents), dtype=action_dtype)
        self.reward = np.zeros((capacity, n_agents), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)

    def add(self, obs, mask, action, reward, next_obs, next_mask, terminated: bool) -> None:
        i = self._next
        self.obs[i], self.mask[i], self.action[i] = obs, mask, action
        self.reward[i], self.next_obs[i], self.next_mask[i] = reward, next_obs, next_mask
        self.terminated[i] = terminated
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self._rng.integers(self.size, size=batch_size)
        return {
            "obs": self.obs[idx].astype(np.float32),
            "mask": self.mask[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx].astype(np.float32),
            "next_mask": self.next_mask[idx],
            "terminated": self.terminated[idx].astype(np.float32),
        }
