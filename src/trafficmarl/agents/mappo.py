"""MAPPO (Yu et al., NeurIPS 2022): PPO with a centralized critic, on RLlib's new API stack.

The actor is the same as IPPO's: a shared, action-masked MLP on the local
observation. Only the critic differs: during training it also receives a global
state, so it can value the whole network's situation; it is dropped at execution.

The global state is "agent-specific" (Yu et al., section 4.1): the agent's own
observation, plus a compressed summary of every intersection, the message of the
extended observation (mean queue, mean density, phase one-hot). Compression keeps
the batch memory of a 25-intersection network within this machine's limits; the
full concatenation would hold 8 times more floats per agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from torch import nn

from trafficmarl.agents import ippo
from trafficmarl.envs.features import FeatureSpec

GLOBAL_STATE = "global_state"


def summarize_observation(spec: FeatureSpec, tls: str, obs: np.ndarray) -> np.ndarray:
    """``features.message`` recomputed from a local observation vector, without SUMO calls."""
    lanes = spec.max_lanes
    n_lanes = len(spec.topology[tls].stop_lanes)
    queue = obs[:lanes].sum() / n_lanes
    density = obs[lanes : 2 * lanes].sum() / n_lanes
    phase = obs[3 * lanes : 3 * lanes + spec.max_phases]
    return np.concatenate([[queue, density], phase]).astype(np.float32)


class MAPPOEnv(ippo.RLlibTrafficEnv):
    """The IPPO environment, with the global state added to every observation."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        spec = self.env.spec
        dim = len(self.possible_agents) * spec.message_dim
        state_space = gym.spaces.Box(0.0, 1.0, (dim,), dtype=np.float32)
        for agent in self.possible_agents:
            space = self.observation_spaces[agent]
            self.observation_spaces[agent] = gym.spaces.Dict(
                {**space.spaces, GLOBAL_STATE: state_space}
            )
        self.observation_space = gym.spaces.Dict(self.observation_spaces)

    def _with_state(self, obs: dict) -> dict:
        spec = self.env.spec
        state = np.concatenate(
            [summarize_observation(spec, a, obs[a]["observations"]) for a in self.possible_agents]
        )
        return {a: {**o, GLOBAL_STATE: state} for a, o in obs.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = super().reset(seed=seed, options=options)
        return self._with_state(obs), infos

    def step(self, action_dict: dict[str, int]):
        obs, rewards, terms, truncs, infos = super().step(action_dict)
        return self._with_state(obs), rewards, terms, truncs, infos


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in hidden:
        layers += [nn.Linear(in_dim, size), nn.Tanh()]
        in_dim = size
    return nn.Sequential(*layers, nn.Linear(in_dim, out_dim))


class MAPPOModule(TorchRLModule, ValueFunctionAPI):
    """Masked actor on the local observation; critic on local observation + global state."""

    def setup(self) -> None:
        spaces = self.observation_space.spaces
        obs_dim = spaces["observations"].shape[0]
        state_dim = spaces[GLOBAL_STATE].shape[0]
        hidden = list(self.model_config["fcnet_hiddens"])
        self.actor = _mlp(obs_dim, hidden, self.action_space.n)
        self.critic = _mlp(obs_dim + state_dim, hidden, 1)

    def _forward(self, batch: dict, **kwargs) -> dict:
        obs = batch[Columns.OBS]
        logits = self.actor(obs["observations"])
        # Forbidden phases get the lowest logit of the float type, as in ActionMaskingTorchRLModule.
        masked = logits.masked_fill(obs["action_mask"] <= 0, torch.finfo(logits.dtype).min)
        return {Columns.ACTION_DIST_INPUTS: masked}

    def compute_values(self, batch: dict, embeddings=None) -> torch.Tensor:
        obs = batch[Columns.OBS]
        return self.critic(torch.cat([obs["observations"], obs[GLOBAL_STATE]], dim=-1)).squeeze(-1)

    def get_inference_action_dist_cls(self):
        return TorchCategorical

    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    def get_train_action_dist_cls(self):
        return TorchCategorical


def _env_creator(env_config):
    return MAPPOEnv(env_config)


def build_algorithm_config(config: dict[str, Any], seed: int, run_dir: Path, device: str):
    return ippo.build_algorithm_config(
        config, seed, run_dir, device, env_creator=_env_creator, module_class=MAPPOModule
    )


def train(config: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    return ippo.train(config, seed, run_dir, builder=build_algorithm_config)
