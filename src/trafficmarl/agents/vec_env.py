"""Environment workers of the off-policy trainers: one SUMO (libsumo) per process.

libsumo hosts one simulation per process, so each worker is a spawned process
stepped in lockstep through a pipe. Workers restart their episode by themselves
and write the same per-episode log as the RLlib runners.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from trafficmarl.agents.training_logs import append_episode


def _stack(obs: dict, agents: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([obs[a]["observations"] for a in agents]),
        np.stack([obs[a]["action_mask"] for a in agents]).astype(bool),
    )


def _worker(conn, config: dict, seed: int, log_path: str | None, action_kind: str) -> None:
    from trafficmarl.envs.duration import to_phases
    from trafficmarl.envs.factory import make_env

    env = make_env(config, split="train", seed=seed)
    agents = list(env.possible_agents)
    path = Path(log_path) if log_path else None
    obs, _ = env.reset(seed=seed)
    current, mask = _stack(obs, agents)
    returns, steps, episode = {a: 0.0 for a in agents}, 0, 0
    conn.send(
        {
            "n_agents": len(agents),
            "obs_dim": env.spec.obs_dim,
            "n_actions": env.spec.max_phases,
            "obs": current,
            "mask": mask,
        }
    )
    while True:
        message = conn.recv()
        if message is None:
            env.close()
            conn.close()
            return
        if action_kind == "duration":
            actions = to_phases(message, env, mask)
        else:
            actions = {a: int(message[i]) for i, a in enumerate(agents)}
        obs, rewards, terms, _, infos = env.step(actions)
        steps += 1
        for a in agents:
            returns[a] += rewards[a]
        next_obs, next_mask = _stack(obs, agents)
        out: dict[str, Any] = {
            "next_obs": next_obs,
            "next_mask": next_mask,
            "reward": np.array([rewards[a] for a in agents], dtype=np.float32),
            "terminated": any(terms.values()),
            "episode_return": None,
            "episode_length": None,
        }
        mask = next_mask
        if not env.agents:
            episode += 1
            summary = next(iter(infos.values()))["episode"]
            append_episode(path, episode, steps, returns, summary)
            out["episode_return"] = float(np.mean(list(returns.values())))
            out["episode_length"] = steps
            obs, _ = env.reset()
            current, mask = _stack(obs, agents)
            out["reset_obs"], out["reset_mask"] = current, mask
            returns, steps = {a: 0.0 for a in agents}, 0
        conn.send(out)


class VecEnv:
    """``n`` environments stepped in lockstep, in as many processes."""

    def __init__(self, config: dict, seed: int, n: int, run_dir: Path, action_kind: str):
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            log = run_dir / "episodes" / f"runner_{i + 1}_0.jsonl"
            # Same seed layout as the RLlib runners (one seed per worker).
            worker_seed = seed * 10_000 + (i + 1) * 100
            proc = ctx.Process(
                target=_worker,
                args=(child, config, worker_seed, str(log), action_kind),
                daemon=True,
            )
            proc.start()
            self._conns.append(parent)
            self._procs.append(proc)
        first = [c.recv() for c in self._conns]
        self.n_agents = first[0]["n_agents"]
        self.obs_dim = first[0]["obs_dim"]
        self.n_actions = first[0]["n_actions"]
        self.obs = np.stack([f["obs"] for f in first])
        self.mask = np.stack([f["mask"] for f in first])

    def step(self, actions: np.ndarray) -> list[dict[str, Any]]:
        """Send one joint action per environment; ``self.obs``/``mask`` move to the next state."""
        for conn, action in zip(self._conns, actions, strict=True):
            conn.send(action)
        outs = [c.recv() for c in self._conns]
        for i, out in enumerate(outs):
            self.obs[i] = out.get("reset_obs", out["next_obs"])
            self.mask[i] = out.get("reset_mask", out["next_mask"])
        return outs

    def close(self) -> None:
        for conn in self._conns:
            conn.send(None)
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
