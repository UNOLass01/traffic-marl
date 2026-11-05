"""IPPO: independent PPO agents with a shared policy, on RLlib's new API stack.

Every traffic light is an agent that acts on its own observation (no
communication, no centralized critic). All agents map to one policy
(``shared``): parameter sharing is valid because observations are padded to
the same size and carry intersection-type features (D20), and the action mask
removes padded and constraint-violating phases (D13). The policy is RLlib's
``ActionMaskingTorchRLModule``, which reads the ``observations`` and
``action_mask`` keys of our observation dictionary and sets the logits of
masked actions to the float minimum before sampling.

Each RLlib environment runner is a separate process holding one SUMO
(libsumo) simulation. Training budgets are counted in environment steps, one
step being one decision of every agent (the same unit as the centralized
baseline's joint decisions, D29).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from trafficmarl.envs.factory import make_env
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)

POLICY_ID = "shared"
ENV_NAME = "trafficmarl"


class RLlibTrafficEnv(MultiAgentEnv):
    """:class:`~trafficmarl.envs.traffic_env.TrafficSignalEnv` as an RLlib multi-agent env.

    ``env_config`` holds the resolved experiment ``config``, the run ``seed`` and
    the ``run_dir``; each finished episode is appended to
    ``<run_dir>/episodes/runner_<i>.jsonl`` (end reason, gridlock, teleports...).
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()
        worker = int(getattr(env_config, "worker_index", 0))
        vector = int(getattr(env_config, "vector_index", 0))
        self._seed = int(env_config.get("seed", 0)) * 10_000 + worker * 100 + vector
        self.env = make_env(env_config["config"], split="train", seed=self._seed)
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.possible_agents)
        self.observation_spaces = {a: self.env.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.env.action_space(a) for a in self.possible_agents}
        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)
        run_dir = env_config.get("run_dir")
        self._log_path = (
            Path(run_dir) / "episodes" / f"runner_{worker}_{vector}.jsonl" if run_dir else None
        )
        self._episode = 0
        self._returns: dict[str, float] = {}
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # The first reset seeds this runner's route-file and SUMO-seed sequence.
        obs, infos = self.env.reset(
            seed=self._seed if self._episode == 0 else None, options=options
        )
        self.agents = list(self.env.agents)
        self._returns = {a: 0.0 for a in self.agents}
        self._steps = 0
        return obs, infos

    def step(self, action_dict: dict[str, int]):
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        self._steps += 1
        for a, r in rewards.items():
            self._returns[a] += r
        done = not self.env.agents
        terms["__all__"] = done and any(terms.values())
        truncs["__all__"] = done and not terms["__all__"]
        if done:
            self.agents = []
            self._log_episode(next(iter(infos.values()))["episode"])
        return obs, rewards, terms, truncs, infos

    def _log_episode(self, summary: dict[str, Any]) -> None:
        self._episode += 1
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "wall_time": time.time(),
            "episode": self._episode,
            "env_steps": self._steps,
            "return_mean": float(np.mean(list(self._returns.values()))),
            **{k: summary[k] for k in ("route_file", "end_reason", "sim_time", "departed",
                                       "arrived", "teleports", "teleport_rate",
                                       "approach_queue_per_signal", "gridlock_penalty")},
        }  # fmt: skip
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.env.close()


def _env_creator(env_config):
    return RLlibTrafficEnv(env_config)


def build_algorithm_config(config: dict[str, Any], seed: int, run_dir: Path, device: str):
    """PPOConfig for IPPO with one shared, action-masked policy."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
        ActionMaskingTorchRLModule,
    )
    from ray.tune.registry import register_env

    algo = config["algorithm"]
    register_env(ENV_NAME, _env_creator)
    env_config = {"config": config, "seed": seed, "run_dir": str(run_dir)}
    return (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_config, disable_env_checking=True)
        .framework("torch")
        .env_runners(
            num_env_runners=int(algo["num_env_runners"]),
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto",
            num_cpus_per_env_runner=1,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=1 if device == "cuda" else 0,
        )
        .multi_agent(
            policies={POLICY_ID},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: POLICY_ID,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    POLICY_ID: RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config={
                            "fcnet_hiddens": list(algo["fcnet_hiddens"]),
                            "fcnet_activation": algo["fcnet_activation"],
                            "head_fcnet_hiddens": [],
                            "vf_share_layers": False,
                        },
                    )
                }
            )
        )
        .training(
            lr=float(algo["lr"]),
            gamma=float(algo["gamma"]),
            lambda_=float(algo["gae_lambda"]),
            clip_param=float(algo["clip_param"]),
            entropy_coeff=float(algo["entropy_coeff"]),
            vf_loss_coeff=float(algo["vf_loss_coeff"]),
            vf_clip_param=float(algo["vf_clip_param"]),
            grad_clip=float(algo["grad_clip"]),
            train_batch_size_per_learner=int(algo["train_batch_size"]),
            minibatch_size=int(algo["minibatch_size"]),
            num_epochs=int(algo["num_epochs"]),
        )
        .debugging(seed=seed, log_level="WARN")
    )


def _metric(result: dict, *path: str, default=np.nan):
    node: Any = result
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def train(config: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    """Train IPPO for ``algorithm.total_env_steps``; write progress, checkpoints and stats."""
    import ray
    from torch.utils.tensorboard import SummaryWriter

    from trafficmarl.utils.device import resolve_device
    from trafficmarl.utils.run_info import write_run_info
    from trafficmarl.utils.seeding import set_global_seed

    algo_cfg = config["algorithm"]
    set_global_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(run_dir, {**config, "device_resolved": device, "seed": seed})
    ray.init(
        num_cpus=int(algo_cfg["num_env_runners"]) + 1,
        num_gpus=1 if device == "cuda" else 0,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"LIBSUMO_AS_TRACI": os.environ.get("LIBSUMO_AS_TRACI", "1")}},
    )
    # Workers are processes: the checkpoint path must be absolute.
    run_dir = run_dir.resolve()
    algo = build_algorithm_config(config, seed, run_dir, device).build_algo()
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    budget = int(algo_cfg["total_env_steps"])
    every = int(algo_cfg["checkpoint_every"])
    rows, started, steps, iteration, wall_before = [], time.monotonic(), 0, 0, 0.0
    if (run_dir / "checkpoint").is_dir() and (run_dir / "progress.csv").is_file():
        # Extension of a finished run to a larger budget (D38), or restart of an
        # interrupted one: continue from the checkpoint.
        algo.restore_from_path(str((run_dir / "checkpoint").resolve()))
        rows = rollback_logs(run_dir, int(algo.iteration))
        if rows:
            iteration, steps = int(rows[-1]["iteration"]), int(rows[-1]["env_steps"])
            wall_before = float(rows[-1]["wall_time_s"])
        logger.info("resuming %s at iteration %d, %d env steps", run_dir, iteration, steps)
    try:
        while steps < budget:
            result = algo.train()
            iteration += 1
            steps = int(_metric(result, "num_env_steps_sampled_lifetime", default=0))
            row = {
                "iteration": iteration,
                "env_steps": steps,
                "agent_steps": _metric(result, "num_agent_steps_sampled_lifetime"),
                "episodes": _metric(result, "env_runners", "num_episodes_lifetime"),
                "episode_return_mean": _metric(result, "env_runners", "episode_return_mean"),
                "episode_len_mean": _metric(result, "env_runners", "episode_len_mean"),
                "policy_loss": _metric(result, "learners", POLICY_ID, "policy_loss"),
                "vf_loss": _metric(result, "learners", POLICY_ID, "vf_loss"),
                "entropy": _metric(result, "learners", POLICY_ID, "entropy"),
                "wall_time_s": wall_before + time.monotonic() - started,
            }
            rows.append(row)
            for key, value in row.items():
                if key not in ("iteration", "env_steps") and value == value:
                    writer.add_scalar(key, float(value), steps)
            pd.DataFrame(rows).to_csv(run_dir / "progress.csv", index=False)
            if iteration % every == 0:
                algo.save_to_path(str(run_dir / "checkpoint"))
        algo.save_to_path(str(run_dir / "checkpoint"))
    finally:
        wall = wall_before + time.monotonic() - started
        writer.close()
        algo.stop()
        ray.shutdown()
    episodes = load_episodes(run_dir)
    stats = {
        "seed": seed,
        "env_steps": steps,
        "iterations": iteration,
        "wall_time_s": round(wall, 1),
        "env_steps_per_s": round(steps / wall, 1) if wall else 0.0,
        "episodes": len(episodes),
        "training_gridlocks": int((episodes.end_reason == "gridlock").sum())
        if len(episodes)
        else 0,
        "device": device,
        "plateau": plateau(pd.DataFrame(rows)),
    }
    (run_dir / "training_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def rollback_logs(run_dir: Path, iteration: int) -> list[dict[str, Any]]:
    """Cut the run's logs back to the checkpoint at ``iteration``; return the kept progress rows.

    A run interrupted between two checkpoints has logged iterations (and
    episodes) whose training is lost on restore. Keeping them would count
    their environment steps twice on the learning curve, so progress rows
    after ``iteration`` and episodes that ended after the checkpoint was
    written are removed. After a completed run this changes nothing.
    """
    progress = pd.read_csv(run_dir / "progress.csv")
    kept = progress[progress.iteration <= iteration]
    if len(kept) < len(progress):
        kept.to_csv(run_dir / "progress.csv", index=False)
        logger.info("dropped %d iterations logged after the checkpoint", len(progress) - len(kept))
    saved_at = (run_dir / "checkpoint" / "algorithm_state.pkl").stat().st_mtime
    for path in (run_dir / "episodes").glob("*.jsonl"):
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x]
        keep = [x for x in lines if json.loads(x)["wall_time"] <= saved_at]
        if len(keep) < len(lines):
            path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
    return kept.to_dict("records")


def load_episodes(run_dir: Path) -> pd.DataFrame:
    files = sorted((run_dir / "episodes").glob("*.jsonl"))
    records = [json.loads(line) for f in files for line in f.read_text().splitlines() if line]
    return pd.DataFrame(records).sort_values("wall_time") if records else pd.DataFrame()


def plateau(progress: pd.DataFrame, tolerance: float = 0.05) -> dict[str, Any]:
    """Has the learning curve flattened? (decisions.md D38)

    The curve (episode return per iteration) is cut into five equal parts by
    environment steps. It has plateaued when the gain between the last two
    fifths is below ``tolerance`` of the gain between the first and last fifth.
    """
    curve = progress.dropna(subset=["episode_return_mean"])
    if len(curve) < 10:
        return {"plateaued": False, "reason": "too few iterations"}
    parts = np.array_split(curve.episode_return_mean.to_numpy(), 5)
    first, fourth, last = parts[0].mean(), parts[3].mean(), parts[4].mean()
    total_gain = last - first
    recent_gain = last - fourth
    flat = total_gain <= 0 or recent_gain <= tolerance * abs(total_gain)
    return {
        "plateaued": bool(flat),
        "first_fifth": round(float(first), 2),
        "fourth_fifth": round(float(fourth), 2),
        "last_fifth": round(float(last), 2),
        "recent_gain_share": round(float(recent_gain / total_gain), 3) if total_gain else None,
    }


class IPPOPolicy:
    """Deterministic shared policy: the most likely phase among the allowed ones."""

    def __init__(self, checkpoint: str):
        import torch
        from ray.rllib.core.rl_module.rl_module import RLModule

        # RLlib reads checkpoints through pyarrow, which needs an absolute path.
        module_dir = Path(checkpoint).resolve() / "learner_group" / "learner" / "rl_module"
        module_dir = module_dir / POLICY_ID
        self.module = RLModule.from_checkpoint(str(module_dir))
        self.module.eval()
        self._torch = torch

    def __call__(self, observations: dict, env) -> dict[str, int]:
        torch = self._torch
        agents = list(observations)
        batch = {
            "obs": {
                "observations": torch.as_tensor(
                    np.stack([observations[a]["observations"] for a in agents])
                ),
                "action_mask": torch.as_tensor(
                    np.stack([observations[a]["action_mask"] for a in agents]).astype(np.float32)
                ),
            }
        }
        with torch.no_grad():
            logits = self.module.forward_inference(batch)["action_dist_inputs"]
        actions = logits.argmax(dim=-1).numpy()
        return {a: int(actions[i]) for i, a in enumerate(agents)}


def make_ippo(spec: dict, env) -> IPPOPolicy:
    return IPPOPolicy(spec["checkpoint"])
