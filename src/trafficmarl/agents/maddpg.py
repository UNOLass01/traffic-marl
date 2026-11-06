"""MADDPG (Lowe et al., NeurIPS 2017), in PyTorch, for the phase-duration action (decision D2).

Each agent has a deterministic actor ``a_i = mu(o_i)`` in [-1, 1], read as a target
green duration (:mod:`trafficmarl.envs.duration`), trained through a critic
``Q(x, a_1..a_N, i)`` that sees all observations and all actions. The critic exists
only during training: execution uses the actor alone, on local observations.

The actor and the critic are shared by all agents, as in every other MARL method of
the study; a one-hot agent index tells the critic whose reward it predicts. Since
the critic input is joint, each update trains on a random subset of agents
(``agents_per_update``) to keep its cost from growing with N squared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from trafficmarl.agents.offpolicy import CHECKPOINT_FILE, exploration_level, train_offpolicy
from trafficmarl.agents.qmix import mlp
from trafficmarl.envs.duration import to_phases


def _polyak(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters(), strict=True):
            t.mul_(1.0 - tau).add_(tau * s)


class MADDPGLearner:
    """Shared actor and centralized critic with their targets. Arrays are [batch, agents, ...]."""

    action_dtype = np.float32

    def __init__(
        self, n_agents: int, obs_dim: int, n_actions: int, cfg: dict, seed: int, device: str = "cpu"
    ):
        self.cfg, self.device = cfg, device
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.rng = np.random.default_rng(seed)
        hidden = list(cfg["hidden"])
        make_actor = lambda: mlp([obs_dim, *hidden, 1]).to(device)  # noqa: E731
        critic_in = n_agents * obs_dim + n_agents + n_agents
        make_critic = lambda: mlp([critic_in, *hidden, 1]).to(device)  # noqa: E731
        self.actor, self.target_actor = make_actor(), make_actor()
        self.critic, self.target_critic = make_critic(), make_critic()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg["actor_lr"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(cfg["critic_lr"]))
        self.updates = 0

    def noise(self, steps: int) -> float:
        c = self.cfg
        return exploration_level(
            steps, float(c["noise_start"]), float(c["noise_end"]), int(c["exploration_steps"])
        )

    def policy(self, obs: torch.Tensor, actor: nn.Module | None = None) -> torch.Tensor:
        return torch.tanh((actor or self.actor)(obs)).squeeze(-1)

    def act(self, obs: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
        """Gaussian exploration; uniformly random actions until learning starts."""
        if steps < int(self.cfg["learning_starts"]):
            return self.rng.uniform(-1.0, 1.0, obs.shape[:2]).astype(np.float32)
        with torch.no_grad():
            action = self.policy(torch.as_tensor(obs, device=self.device)).cpu().numpy()
        noisy = action + self.rng.normal(0.0, self.noise(steps), action.shape)
        return np.clip(noisy, -1.0, 1.0).astype(np.float32)

    def _q(self, critic, joint_obs, joint_action, agent: int) -> torch.Tensor:
        index = torch.zeros(joint_obs.shape[0], self.n_agents, device=self.device)
        index[:, agent] = 1.0
        return critic(torch.cat([joint_obs, joint_action, index], dim=-1)).squeeze(-1)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        obs, next_obs, action = tensors["obs"], tensors["next_obs"], tensors["action"]
        size = obs.shape[0]
        joint_obs, joint_next = obs.view(size, -1), next_obs.view(size, -1)
        count = min(int(self.cfg["agents_per_update"]), self.n_agents)
        chosen = self.rng.choice(self.n_agents, size=count, replace=False)
        reward = tensors["reward"] * float(self.cfg["reward_scale"])
        keep = self.cfg["gamma"] * (1.0 - tensors["terminated"])

        with torch.no_grad():
            next_action = self.policy(next_obs, self.target_actor)
        critic_loss = torch.zeros((), device=self.device)
        for agent in chosen:
            with torch.no_grad():
                bootstrap = self._q(self.target_critic, joint_next, next_action, int(agent))
                target = reward[:, agent] + keep * bootstrap
            critic_loss = critic_loss + F.mse_loss(
                self._q(self.critic, joint_obs, action, int(agent)), target
            )
        critic_loss = critic_loss / count
        self.critic_opt.zero_grad()
        critic_loss.backward()
        clip = float(self.cfg["grad_clip"])
        nn.utils.clip_grad_norm_(self.critic.parameters(), clip)
        self.critic_opt.step()

        # Each chosen agent's action comes from the current policy, the others' from replay.
        mine = self.policy(obs)
        actor_loss = torch.zeros((), device=self.device)
        for agent in chosen:
            joint = action.clone()
            joint[:, agent] = mine[:, agent]
            actor_loss = actor_loss - self._q(self.critic, joint_obs, joint, int(agent)).mean()
        actor_loss = actor_loss / count
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), clip)
        self.actor_opt.step()

        tau = float(self.cfg["tau"])
        _polyak(self.target_actor, self.actor, tau)
        _polyak(self.target_critic, self.critic, tau)
        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "dims": {"obs_dim": self.obs_dim},
            "hidden": list(self.cfg["hidden"]),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("actor", "critic", "target_actor", "target_critic", "actor_opt", "critic_opt"):
            getattr(self, name).load_state_dict(state[name])
        self.updates = state["updates"]


def train(config: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    return train_offpolicy(config, seed, run_dir, MADDPGLearner, action_kind="duration")


class MADDPGPolicy:
    """Decentralized execution: the shared actor, without noise, on local observations."""

    def __init__(self, checkpoint: str):
        state = torch.load(Path(checkpoint) / CHECKPOINT_FILE, map_location="cpu",
                           weights_only=False)["learner"]  # fmt: skip
        self.actor = mlp([state["dims"]["obs_dim"], *state["hidden"], 1])
        self.actor.load_state_dict(state["actor"])
        self.actor.eval()

    def __call__(self, observations: dict, env) -> dict[str, int]:
        agents = list(env.possible_agents)
        obs = np.stack([observations[a]["observations"] for a in agents])
        mask = np.stack([observations[a]["action_mask"] for a in agents]).astype(bool)
        with torch.no_grad():
            actions = torch.tanh(self.actor(torch.as_tensor(obs))).squeeze(-1).numpy()
        return to_phases(actions, env, mask)


def make_maddpg(spec: dict, env) -> MADDPGPolicy:
    return MADDPGPolicy(spec["checkpoint"])
