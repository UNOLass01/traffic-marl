"""QMIX (Rashid et al., ICML 2018), in PyTorch, with action masking (decision D2).

One agent network, shared by all intersections, gives per-agent utilities
``Q_i(o_i, a_i)``. A mixing network combines the chosen utilities into
``Q_tot(s, a)``; its weights come from hypernetworks of the global state ``s`` and
are made non-negative, so ``Q_tot`` is monotonic in every ``Q_i``. The joint greedy
action is then each agent's own greedy action, which keeps execution decentralized.

Choices that differ from the paper, recorded in docs/decisions.md:

* feed-forward agent networks instead of GRUs: the observation holds the current
  phase and the time since the last change, so little depends on history, and
  transitions can be replayed one at a time instead of as whole episodes;
* double Q-learning targets (as in the authors' PyMARL code);
* targets respect the action mask of the next state.

The team reward is the mean of the agents' rewards, so its scale does not depend
on the network size (like the ``global`` scheme, D21). The global state is the
concatenation of all observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from trafficmarl.agents.offpolicy import CHECKPOINT_FILE, exploration_level, train_offpolicy


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class MonotonicMixer(nn.Module):
    """Two-layer mixing network whose weights are absolute values of hypernetwork outputs."""

    def __init__(self, n_agents: int, state_dim: int, embed: int = 32, hyper: int = 64):
        super().__init__()
        self.n_agents, self.embed = n_agents, embed
        self.hyper_w1 = mlp([state_dim, hyper, n_agents * embed])
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = mlp([state_dim, hyper, embed])
        self.hyper_b2 = mlp([state_dim, embed, 1])

    def forward(self, utilities: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``utilities`` [B, N], ``state`` [B, S] -> ``Q_tot`` [B]."""
        w1 = self.hyper_w1(state).abs().view(-1, self.n_agents, self.embed)
        b1 = self.hyper_b1(state).view(-1, 1, self.embed)
        hidden = F.elu(torch.bmm(utilities.unsqueeze(1), w1) + b1)
        w2 = self.hyper_w2(state).abs().view(-1, self.embed, 1)
        b2 = self.hyper_b2(state).view(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(-1)


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~mask, -torch.inf).argmax(dim=-1)


class QMIXLearner:
    """Agent network, mixer and their targets. Arrays are [batch, agents, ...]."""

    action_dtype = np.int64

    def __init__(
        self, n_agents: int, obs_dim: int, n_actions: int, cfg: dict, seed: int, device: str = "cpu"
    ):
        self.cfg, self.device = cfg, device
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.rng = np.random.default_rng(seed)
        hidden = list(cfg["hidden"])
        state_dim = n_agents * obs_dim
        embed, hyper = int(cfg["mixer_embed"]), int(cfg["mixer_hyper"])
        make_agent = lambda: mlp([obs_dim, *hidden, n_actions]).to(device)  # noqa: E731
        make_mixer = lambda: MonotonicMixer(n_agents, state_dim, embed, hyper).to(device)  # noqa: E731
        self.agent, self.target_agent = make_agent(), make_agent()
        self.mixer, self.target_mixer = make_mixer(), make_mixer()
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            [*self.agent.parameters(), *self.mixer.parameters()], lr=float(cfg["lr"])
        )
        self.updates = 0

    def epsilon(self, steps: int) -> float:
        c = self.cfg
        return exploration_level(
            steps, float(c["epsilon_start"]), float(c["epsilon_end"]), int(c["exploration_steps"])
        )

    def act(self, obs: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
        """Epsilon-greedy among the allowed phases, each agent exploring on its own."""
        with torch.no_grad():
            q = self.agent(torch.as_tensor(obs, device=self.device))
            greedy = masked_argmax(q, torch.as_tensor(mask, device=self.device)).cpu().numpy()
        noise = self.rng.random(mask.shape)  # uniform among allowed phases: argmax of random keys
        random_actions = np.where(mask, noise, -1.0).argmax(axis=-1)
        explore = self.rng.random(greedy.shape) < self.epsilon(steps)
        return np.where(explore, random_actions, greedy)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        obs, next_obs = tensors["obs"], tensors["next_obs"]
        size = obs.shape[0]
        team_reward = tensors["reward"].mean(dim=1) * float(self.cfg["reward_scale"])
        chosen = self.agent(obs).gather(-1, tensors["action"].unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen, obs.view(size, -1))
        with torch.no_grad():
            best = masked_argmax(self.agent(next_obs), tensors["next_mask"]).unsqueeze(-1)
            target_q = self.target_agent(next_obs).gather(-1, best).squeeze(-1)
            target_tot = self.target_mixer(target_q, next_obs.view(size, -1))
            targets = team_reward + self.cfg["gamma"] * (1.0 - tensors["terminated"]) * target_tot
        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        params = [*self.agent.parameters(), *self.mixer.parameters()]
        grad_norm = nn.utils.clip_grad_norm_(params, float(self.cfg["grad_clip"]))
        self.optimizer.step()
        self.updates += 1
        if self.updates % int(self.cfg["target_update_every"]) == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return {
            "td_loss": float(loss.detach()),
            "q_tot_mean": float(q_tot.detach().mean()),
            "grad_norm": float(grad_norm),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "dims": {"obs_dim": self.obs_dim, "n_actions": self.n_actions},
            "hidden": list(self.cfg["hidden"]),
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("agent", "mixer", "target_agent", "target_mixer", "optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.updates = state["updates"]


def train(config: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    return train_offpolicy(config, seed, run_dir, QMIXLearner, action_kind="phase")


class QMIXPolicy:
    """Decentralized execution: every agent takes the allowed phase with the highest utility."""

    def __init__(self, checkpoint: str):
        state = torch.load(Path(checkpoint) / CHECKPOINT_FILE, map_location="cpu",
                           weights_only=False)["learner"]  # fmt: skip
        dims = state["dims"]
        self.agent = mlp([dims["obs_dim"], *state["hidden"], dims["n_actions"]])
        self.agent.load_state_dict(state["agent"])
        self.agent.eval()

    def __call__(self, observations: dict, env) -> dict[str, int]:
        agents = list(observations)
        obs = np.stack([observations[a]["observations"] for a in agents])
        mask = np.stack([observations[a]["action_mask"] for a in agents]).astype(bool)
        with torch.no_grad():
            actions = masked_argmax(self.agent(torch.as_tensor(obs)), torch.as_tensor(mask))
        return {a: int(actions[i]) for i, a in enumerate(agents)}


def make_qmix(spec: dict, env) -> QMIXPolicy:
    return QMIXPolicy(spec["checkpoint"])
