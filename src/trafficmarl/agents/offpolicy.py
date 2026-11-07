"""Training loop shared by the in-house off-policy methods (QMIX, MADDPG).

A learner object provides ``act``, ``update``, ``state_dict`` and
``load_state_dict``; this module runs the environments, the replay buffer, the
logs and the checkpoints, with the same run layout as the RLlib trainers:
``progress.csv``, ``episodes/``, TensorBoard, ``checkpoint/`` and
``training_stats.json``. An "iteration" is ``iteration_steps`` environment steps.
The replay buffer is not checkpointed: a restarted run refills it for
``learning_starts`` steps while its networks keep training only after that.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from trafficmarl.agents.replay import ReplayBuffer
from trafficmarl.agents.training_logs import rollback_logs, training_stats
from trafficmarl.agents.vec_env import VecEnv
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)

CHECKPOINT_FILE = "state.pt"


class Learner(Protocol):
    action_dtype: type

    def act(self, obs: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray: ...
    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def exploration_level(steps: int, start: float, end: float, anneal_steps: int) -> float:
    """Linear annealing from ``start`` to ``end`` over ``anneal_steps`` environment steps."""
    fraction = min(steps / anneal_steps, 1.0)
    return start + fraction * (end - start)


def train_offpolicy(
    config: dict[str, Any],
    seed: int,
    run_dir: Path,
    build_learner: Callable[[int, int, int, dict, int, str], Learner],
    action_kind: str,
) -> dict[str, Any]:
    """Train for ``algorithm.total_env_steps``; write progress, checkpoints and stats."""
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from trafficmarl.utils.device import resolve_device
    from trafficmarl.utils.run_info import write_run_info
    from trafficmarl.utils.seeding import set_global_seed

    algo = config["algorithm"]
    set_global_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    torch.set_num_threads(int(algo.get("torch_threads", 4)))
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(run_dir, {**config, "device_resolved": device, "seed": seed})

    envs = VecEnv(config, seed, int(algo["num_env_runners"]), run_dir, action_kind)
    learner = build_learner(envs.n_agents, envs.obs_dim, envs.n_actions, algo, seed, device)
    replay = ReplayBuffer(
        int(algo["replay_capacity"]), envs.n_agents, envs.obs_dim, envs.n_actions,
        learner.action_dtype, seed,
    )  # fmt: skip
    checkpoint = run_dir / "checkpoint" / CHECKPOINT_FILE
    rows: list[dict[str, Any]] = []
    steps, iteration, updates, wall_before = 0, 0, 0, 0.0
    if checkpoint.is_file() and (run_dir / "progress.csv").is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        learner.load_state_dict(state["learner"])
        steps, iteration, updates = state["steps"], state["iteration"], state["updates"]
        rows = rollback_logs(run_dir, iteration, CHECKPOINT_FILE)
        wall_before = float(rows[-1]["wall_time_s"]) if rows else 0.0
        logger.info("resuming %s at iteration %d, %d env steps", run_dir, iteration, steps)

    budget = int(algo["total_env_steps"])
    per_iteration = int(algo["iteration_steps"])
    every = int(algo["checkpoint_every"])
    starts = steps + int(algo["learning_starts"])
    update_every = int(algo["update_every"])
    batch_size = int(algo["batch_size"])
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    returns: deque[float] = deque(maxlen=20)
    lengths: deque[int] = deque(maxlen=20)
    episodes = int(rows[-1]["episodes"]) if rows else 0
    losses: dict[str, float] = {}
    started = time.monotonic()
    since_update = 0
    try:
        while steps < budget:
            actions = learner.act(envs.obs, envs.mask, steps)
            obs, mask = envs.obs.copy(), envs.mask.copy()
            outs = envs.step(actions)
            for i, out in enumerate(outs):
                replay.add(obs[i], mask[i], actions[i], out["reward"], out["next_obs"],
                           out["next_mask"], out["terminated"])  # fmt: skip
                if out["episode_return"] is not None:
                    returns.append(out["episode_return"])
                    lengths.append(out["episode_length"])
                    episodes += 1
            steps += len(outs)
            since_update += len(outs)
            if steps >= starts and replay.size >= batch_size:
                while since_update >= update_every:
                    losses = learner.update(replay.sample(batch_size))
                    updates += 1
                    since_update -= update_every
            if steps // per_iteration > iteration:
                iteration = steps // per_iteration
                row = {
                    "iteration": iteration,
                    "env_steps": steps,
                    "agent_steps": steps * envs.n_agents,
                    "episodes": episodes,
                    "episode_return_mean": float(np.mean(returns)) if returns else np.nan,
                    "episode_len_mean": float(np.mean(lengths)) if lengths else np.nan,
                    "updates": updates,
                    **losses,
                    "wall_time_s": wall_before + time.monotonic() - started,
                }
                rows.append(row)
                for key, value in row.items():
                    if key not in ("iteration", "env_steps") and value == value:
                        writer.add_scalar(key, float(value), steps)
                pd.DataFrame(rows).to_csv(run_dir / "progress.csv", index=False)
                if iteration % every == 0:
                    _save(checkpoint, learner, steps, iteration, updates)
        _save(checkpoint, learner, steps, iteration, updates)
    finally:
        wall = wall_before + time.monotonic() - started
        writer.close()
        envs.close()
    return training_stats(run_dir, seed, steps, iteration, wall, device, rows)


def _save(path: Path, learner: Learner, steps: int, iteration: int, updates: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "learner": learner.state_dict(),
        "steps": steps,
        "iteration": iteration,
        "updates": updates,
    }
    # Written then renamed: an interruption during the save keeps the previous checkpoint.
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
