"""Centralized single-agent control: one PPO agent drives every traffic light.

The multi-agent environment is wrapped as a single Gymnasium environment:

* observation: all agents' observations concatenated (N x 47);
* action: a factored joint action, ``MultiDiscrete([P] * N)`` (one phase per
  intersection), with the agents' action masks concatenated so that the same
  minimum/maximum-green constraints apply (sb3-contrib MaskablePPO);
* reward: the mean of the agents' rewards (local traffic terms minus switch
  penalties, gridlock penalty included), i.e. a network-wide objective.

Its input and output sizes grow linearly with the number of intersections,
which is exactly what the scalability study measures (training time, memory,
convergence).
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from trafficmarl.envs.factory import make_env


class CentralizedEnv(gym.Env):
    """Single-agent view of :class:`~trafficmarl.envs.traffic_env.TrafficSignalEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, split: str = "train", seed: int = 0):
        self.env = make_env(config, split=split, seed=seed)
        self.agents = list(self.env.possible_agents)
        n, p, d = len(self.agents), self.env.spec.max_phases, self.env.spec.obs_dim
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n * d,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([p] * n)
        self._obs: dict = {}
        self.gridlocks = 0

    def _flat(self, obs: dict) -> np.ndarray:
        return np.concatenate([obs[a]["observations"] for a in self.agents]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        return np.concatenate([self._obs[a]["action_mask"] for a in self.agents]).astype(bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._obs, _ = self.env.reset(seed=seed, options=options)
        return self._flat(self._obs), {}

    def step(self, action):
        actions = {a: int(action[i]) for i, a in enumerate(self.agents)}
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._obs = obs
        info = next(iter(infos.values()))
        if info.get("gridlock"):
            self.gridlocks += 1
        reward = float(np.mean(list(rewards.values())))
        terminated = any(terms.values())
        truncated = any(truncs.values()) and not terminated
        return (
            self._flat(obs),
            reward,
            terminated,
            truncated,
            {"episode_summary": info.get("episode")},
        )

    def close(self) -> None:
        self.env.close()


def _make_env_fn(config: dict, seed: int, rank: int):
    def _init():
        env = CentralizedEnv(config, split="train", seed=seed * 1000 + rank)
        env.reset(seed=seed * 1000 + rank)
        return env

    return _init


def train(config: dict, seed: int, run_dir: Path) -> dict[str, Any]:
    """Train MaskablePPO with ``config['algorithm']`` hyperparameters; save model and stats."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from trafficmarl.utils.device import resolve_device
    from trafficmarl.utils.run_info import write_run_info
    from trafficmarl.utils.seeding import set_global_seed

    algo = config["algorithm"]
    set_global_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(run_dir, {**config, "device_resolved": device, "seed": seed})
    # One SUMO (libsumo) instance per subprocess.
    vec = SubprocVecEnv(
        [_make_env_fn(config, seed, r) for r in range(algo["n_envs"])], start_method="spawn"
    )
    vec = VecMonitor(vec, filename=str(run_dir / "monitor"))
    model = MaskablePPO(
        "MlpPolicy",
        vec,
        learning_rate=algo["learning_rate"],
        n_steps=algo["n_steps"],
        batch_size=algo["batch_size"],
        n_epochs=algo["n_epochs"],
        gamma=algo["gamma"],
        gae_lambda=algo["gae_lambda"],
        clip_range=algo["clip_range"],
        ent_coef=algo["ent_coef"],
        policy_kwargs={"net_arch": list(algo["net_arch"])},
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=seed,
        device=device,
        verbose=0,
    )
    started = time.monotonic()
    model.learn(total_timesteps=int(algo["total_timesteps"]), progress_bar=False)
    wall = time.monotonic() - started
    model.save(run_dir / "model.zip")
    gridlocks = sum(vec.get_attr("gridlocks"))
    vec.close()
    stats = {
        "seed": seed,
        "timesteps": int(model.num_timesteps),
        "wall_time_s": round(wall, 1),
        "steps_per_s": round(model.num_timesteps / wall, 1),
        "peak_rss_mb_main": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "peak_rss_mb_children": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1
        ),
        "obs_dim": int(vec.observation_space.shape[0]),
        "action_heads": len(vec.action_space.nvec),
        "parameters": int(sum(p.numel() for p in model.policy.parameters())),
        "training_gridlocks": int(gridlocks),
        "device": device,
    }
    (run_dir / "training_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class CentralizedPolicy:
    """Deterministic MaskablePPO policy as a ``(observations, env) -> actions`` callable."""

    def __init__(self, checkpoint: str, env):
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(checkpoint, device="cpu")
        self.agents = list(env.possible_agents)

    def __call__(self, observations: dict, env) -> dict[str, int]:
        flat = np.concatenate([observations[a]["observations"] for a in self.agents])
        mask = np.concatenate([observations[a]["action_mask"] for a in self.agents]).astype(bool)
        action, _ = self.model.predict(flat, deterministic=True, action_masks=mask)
        return {a: int(action[i]) for i, a in enumerate(self.agents)}


def make_centralized(spec: dict, env) -> CentralizedPolicy:
    return CentralizedPolicy(spec["checkpoint"], env)
