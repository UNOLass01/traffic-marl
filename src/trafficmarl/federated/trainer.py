"""Independent and federated PPO: one local policy per traffic light (Phase 9).

Each intersection trains its own PPO policy on its own experience. With
``algorithm.round_every: null`` nothing is ever shared (independent learners);
otherwise, every ``round_every`` PPO iterations, the clients send their parameters
to the server (FedAvg, :mod:`trafficmarl.federated.server`) and restart from the
global model. Optimizer states stay local. Both are compared with the parameter
sharing of IPPO, where one policy trains on everybody's raw trajectories.

RLlib runs the clients in one process (one module per client); client and server
only exchange parameters, through :class:`~trafficmarl.federated.client.FederatedClient`.
A run's last step is an aggregation, so the federated result is one global model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trafficmarl.agents import ippo
from trafficmarl.agents.training_logs import rollback_logs, training_stats
from trafficmarl.federated.client import FederatedClient
from trafficmarl.federated.server import FedAvgServer, parameter_count
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)

SERVER_FILE = "federated_server.json"


def _agent_policy(agent_id, *args, **kwargs):
    return agent_id


def build_algorithm_config(config: dict[str, Any], seed: int, run_dir: Path, device: str, agents):
    """The IPPO configuration with one module per traffic light."""
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

    algo = ippo.build_algorithm_config(config, seed, run_dir, device)
    shared_spec = algo.rl_module_spec.rl_module_specs[ippo.POLICY_ID]
    return algo.multi_agent(
        policies=set(agents), policy_mapping_fn=_agent_policy
    ).rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs={a: shared_spec for a in agents})
    )  # fmt: skip


def _sync_runners(algo, agents: list[str]) -> None:
    """Send the learners' weights to the environment runners before the next sampling."""
    algo.env_runner_group.sync_weights(
        policies=agents, from_worker_or_learner_group=algo.learner_group, inference_only=True
    )


def _federation_round(algo, clients, server, agents) -> None:
    models = [c.parameters() for c in clients]
    model = server.aggregate(models, [c.samples for c in clients])
    for client in clients:
        client.load(model)
    _sync_runners(algo, agents)


def _agent_ids(config: dict[str, Any]) -> list[str]:
    from trafficmarl.scenarios.spec import Manifest, output_dir

    return sorted(str(t) for t in Manifest.load(output_dir(config["scenario"])).tls_ids)


def train(config: dict[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    """Train for ``algorithm.total_env_steps``; write progress, checkpoints and stats."""
    import ray
    from torch.utils.tensorboard import SummaryWriter

    from trafficmarl.utils.device import resolve_device
    from trafficmarl.utils.run_info import write_run_info
    from trafficmarl.utils.seeding import set_global_seed

    cfg = config["algorithm"]
    set_global_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_info(run_dir, {**config, "device_resolved": device, "seed": seed})
    ray.init(
        num_cpus=int(cfg["num_env_runners"]) + 1,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"LIBSUMO_AS_TRACI": os.environ.get("LIBSUMO_AS_TRACI", "1")}},
    )
    agents = _agent_ids(config)
    algo = build_algorithm_config(config, seed, run_dir, device, agents).build_algo()
    round_every = cfg.get("round_every")
    server = FedAvgServer(
        clip_norm=cfg.get("clip_norm"), noise_multiplier=float(cfg.get("noise_multiplier", 0.0)),
        seed=seed,
    )  # fmt: skip
    group = algo.learner_group
    clients = [
        FederatedClient(
            a,
            read=lambda m: group.get_weights([m])[m],
            write=lambda m, w: group.set_weights({m: w}),
        )
        for a in agents
    ]
    budget, every = int(cfg["total_env_steps"]), int(cfg["checkpoint_every"])
    batch = int(cfg["train_batch_size"])
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    rows, started, steps, iteration, wall_before = [], time.monotonic(), 0, 0, 0.0
    server_file = run_dir / "checkpoint" / SERVER_FILE
    if (run_dir / "checkpoint").is_dir() and (run_dir / "progress.csv").is_file():
        algo.restore_from_path(str(run_dir / "checkpoint"))
        rows = rollback_logs(run_dir, int(algo.iteration))
        if rows:
            iteration, steps = int(rows[-1]["iteration"]), int(rows[-1]["env_steps"])
            wall_before = float(rows[-1]["wall_time_s"])
        if server_file.is_file():
            saved = json.loads(server_file.read_text())
            server.rounds, server.bytes_up = saved["rounds"], saved["bytes_up"]
            server.bytes_down = saved["bytes_down"]
    if round_every and iteration == 0:
        # Every client starts from the same model, as in FedAvg.
        first = clients[0].parameters()
        for client in clients:
            client.load(first)
        _sync_runners(algo, agents)
    try:
        while steps < budget:
            result = algo.train()
            iteration += 1
            steps = int(ippo._metric(result, "num_env_steps_sampled_lifetime", default=0))
            for client in clients:
                client.record_samples(batch)
            if round_every and iteration % int(round_every) == 0:
                _federation_round(algo, clients, server, agents)
            row = {
                "iteration": iteration,
                "env_steps": steps,
                "agent_steps": ippo._metric(result, "num_agent_steps_sampled_lifetime"),
                "episodes": ippo._metric(result, "env_runners", "num_episodes_lifetime"),
                "episode_return_mean": ippo._metric(result, "env_runners", "episode_return_mean"),
                "episode_len_mean": ippo._metric(result, "env_runners", "episode_len_mean"),
                "federation_rounds": server.rounds,
                "federation_bytes": server.bytes_up + server.bytes_down,
                "wall_time_s": wall_before + time.monotonic() - started,
            }
            rows.append(row)
            for key, value in row.items():
                if key not in ("iteration", "env_steps") and value == value:
                    writer.add_scalar(key, float(value), steps)
            pd.DataFrame(rows).to_csv(run_dir / "progress.csv", index=False)
            if iteration % every == 0:
                algo.save_to_path(str(run_dir / "checkpoint"))
                server_file.write_text(json.dumps(_server_state(server)), encoding="utf-8")
        if round_every and iteration % int(round_every) != 0:
            _federation_round(algo, clients, server, agents)  # the result is one global model
        algo.save_to_path(str(run_dir / "checkpoint"))
        server_file.write_text(json.dumps(_server_state(server)), encoding="utf-8")
        params = parameter_count(clients[0].parameters())
    finally:
        wall = wall_before + time.monotonic() - started
        writer.close()
        algo.stop()
        ray.shutdown()
    stats = training_stats(run_dir, seed, steps, iteration, wall, device, rows)
    stats["federation"] = {
        **_server_state(server),
        "round_every": round_every,
        "clients": len(agents),
        "parameters_per_client": params,
        "bytes_per_round": (server.bytes_up + server.bytes_down) // max(server.rounds, 1),
        "noise_multiplier": server.noise_multiplier,
    }
    (run_dir / "training_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _server_state(server: FedAvgServer) -> dict[str, int]:
    return {"rounds": server.rounds, "bytes_up": server.bytes_up, "bytes_down": server.bytes_down}


class LocalPoliciesPolicy:
    """Each intersection acts with its own module (identical after a final FedAvg round)."""

    def __init__(self, checkpoint: str, agents: list[str]):
        import torch
        from ray.rllib.core.rl_module.rl_module import RLModule

        root = Path(checkpoint).resolve() / "learner_group" / "learner" / "rl_module"
        self.modules = {a: RLModule.from_checkpoint(str(root / a)) for a in agents}
        for module in self.modules.values():
            module.eval()
        self._torch = torch

    def __call__(self, observations: dict, env) -> dict[str, int]:
        torch = self._torch
        actions = {}
        for agent, obs in observations.items():
            batch = {
                "obs": {
                    "observations": torch.as_tensor(obs["observations"][None]),
                    "action_mask": torch.as_tensor(obs["action_mask"][None].astype(np.float32)),
                }
            }
            with torch.no_grad():
                logits = self.modules[agent].forward_inference(batch)["action_dist_inputs"]
            actions[agent] = int(logits.argmax(dim=-1)[0])
        return actions


def make_local_policies(spec: dict, env) -> LocalPoliciesPolicy:
    return LocalPoliciesPolicy(spec["checkpoint"], list(env.possible_agents))
