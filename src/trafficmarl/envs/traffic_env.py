"""Multi-agent traffic-signal environment (PettingZoo ``ParallelEnv``) built on SUMO-RL.

SUMO-RL runs the simulation and the yellow transitions; this module adds what
the project's MDP needs (docs/mdp.md):

* observations and local rewards through SUMO-RL's official hooks
  (``observation_class`` and ``reward_fn``), measured on the full approach;
* action masking for minimum / maximum green and padded phases (D13), with the
  mask in every observation (``{"observations", "action_mask"}``) and in ``info``;
* cooperative reward schemes (local / neighbourhood / global) and the switch
  penalty;
* identical episode rules for every method (D14): fixed time limit
  (``truncated``), early end when the network is empty or gridlocked
  (``terminated``), and a terminal penalty on gridlock so that jamming the
  network can never pay off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sumolib
from gymnasium import spaces
from pettingzoo import ParallelEnv
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction

from trafficmarl.envs import features
from trafficmarl.envs.constraints import PhaseConstraints
from trafficmarl.envs.measure import ApproachMeter
from trafficmarl.envs.topology import build_topology
from trafficmarl.scenarios.spec import Manifest
from trafficmarl.utils.logging_utils import get_logger

logger = get_logger(__name__)

END_TIME_LIMIT = "time_limit"
END_EMPTY = "network_empty"
END_GRIDLOCK = "gridlock"


class CountingSumoEnvironment(SumoEnvironment):
    """SUMO-RL environment that also counts arrivals and teleports every simulated second.

    Only ``_sumo_step`` is extended; everything else is SUMO-RL's own code.
    """

    def reset_counters(self) -> None:
        self.arrived = 0
        self.departed = 0
        self.teleports = 0
        self.last_arrival_time = float(self.begin_time)

    def _sumo_step(self) -> None:
        super()._sumo_step()
        sim = self.sumo.simulation
        arrived = sim.getArrivedNumber()
        if arrived:
            self.arrived += arrived
            self.last_arrival_time = sim.getTime()
        self.departed += sim.getDepartedNumber()
        self.teleports += sim.getStartingTeleportNumber()


def count_green_phases(net_file: str) -> dict[str, int]:
    """Green phases per traffic light, with the same rule SUMO-RL uses to build its actions."""
    net = sumolib.net.readNet(net_file, withPrograms=True)
    counts = {}
    for tl in net.getTrafficLights():
        program = next(iter(tl.getPrograms().values()))
        counts[tl.getID()] = sum(
            "y" not in p.state and (p.state.count("r") + p.state.count("s") != len(p.state))
            for p in program.getPhases()
        )
    return counts


@dataclass
class EpisodeLog:
    """Per-agent reward components, kept for reward-scale diagnostics."""

    traffic: dict[str, list[float]] = field(default_factory=dict)
    penalty: dict[str, list[float]] = field(default_factory=dict)
    reward: dict[str, list[float]] = field(default_factory=dict)


class TrafficSignalEnv(ParallelEnv):
    """One agent per traffic light; see the module docstring."""

    metadata = {"name": "trafficmarl_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(
        self,
        manifest: Manifest,
        *,
        profile: str,
        split: str,
        env_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
        yellow_time: int,
        seed: int = 0,
        extra_additional_files: list[str] | None = None,
        reward_scale: dict[str, float] | None = None,
    ):
        self.manifest = manifest
        self.route_files = manifest.route_files(profile, split)
        if not self.route_files:
            raise ValueError(f"{manifest.name} has no route files for {profile}/{split}")
        self.split = split
        self.env_cfg = env_cfg
        self.reward_cfg = env_cfg["reward"]
        self.sim_cfg = sim_cfg
        self.delta_time = int(env_cfg["delta_time"])
        self.time_limit = manifest.duration + int(sim_cfg["drain_time"])
        self._rng = np.random.default_rng(seed)
        self._episode = 0

        net_file = str(manifest.path(manifest.net_file))
        obs_cfg = env_cfg["observation"]
        topology = build_topology(net_file, obs_cfg["max_distance"])
        n_green = count_green_phases(net_file)
        self.spec = features.FeatureSpec(
            agents=sorted(topology),
            topology=topology,
            n_green=n_green,
            max_lanes=max(len(t.stop_lanes) for t in topology.values()),
            max_phases=max(n_green.values()),
            max_distance=float(obs_cfg["max_distance"]),
            max_green=int(env_cfg["max_green"]),
            extended=obs_cfg["type"] == "extended",
            agent_id=bool(obs_cfg.get("agent_id", False)),
        )
        self.meter = ApproachMeter(topology)
        self.constraints = PhaseConstraints(
            n_green,
            self.spec.max_phases,
            delta_time=self.delta_time,
            yellow_time=yellow_time,
            min_green=int(env_cfg["min_green"]),
            max_green=int(env_cfg["max_green"]),
            max_red=env_cfg.get("max_red"),
        )
        self._penalty_bound = self._per_step_reward_bound()
        self._penalty_cap = self._gridlock_penalty_cap(reward_scale or {})
        self.gridlock_count = 0

        additional = [str(manifest.path(f)) for f in manifest.additional_files]
        additional += list(extra_additional_files or [])
        self.sumo_env = CountingSumoEnvironment(
            net_file=net_file,
            route_file=str(self.route_files[0]),
            use_gui=False,
            num_seconds=self.time_limit,
            delta_time=self.delta_time,
            yellow_time=yellow_time,
            min_green=int(env_cfg["min_green"]),
            max_green=int(env_cfg["max_green"]),
            time_to_teleport=int(sim_cfg["time_to_teleport"]),
            observation_class=self._observation_class(),
            reward_fn=self._reward_fn(),
            add_system_info=False,
            add_per_agent_info=False,
            sumo_warnings=False,
            additional_sumo_cmd=f"--additional-files {','.join(additional)}"
            if additional
            else None,
        )
        if sorted(self.sumo_env.ts_ids) != self.spec.agents:
            raise RuntimeError("SUMO-RL and the topology disagree on the traffic lights")
        self.possible_agents = list(self.spec.agents)
        self.agents: list[str] = []
        self._obs_space = spaces.Dict(
            {
                "observations": spaces.Box(0.0, 1.0, (self.spec.obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (self.spec.max_phases,), dtype=np.int8),
            }
        )
        self._act_space = spaces.Discrete(self.spec.max_phases)
        self.log = EpisodeLog()

    # ------------------------------------------------------------------ hooks
    def _observation_class(self) -> type[ObservationFunction]:
        env = self

        class ApproachObservation(ObservationFunction):
            """SUMO-RL observation hook delegating to :func:`features.observe`."""

            def __call__(self) -> np.ndarray:
                ts = self.ts
                return features.observe(
                    env.spec,
                    env.meter,
                    ts.sumo,
                    ts.id,
                    env.green_phases(),
                    ts.time_since_last_phase_change,
                    env._keep_only(ts.id),
                )

            def observation_space(self) -> spaces.Box:
                return spaces.Box(0.0, 1.0, (env.spec.obs_dim,), dtype=np.float32)

        return ApproachObservation

    def _reward_fn(self):
        env = self

        def local_traffic_reward(ts) -> float:
            """SUMO-RL reward hook: the local traffic term of :func:`features.traffic_term`."""
            value, ts.last_measure = features.traffic_term(
                env.spec,
                env.meter.stats(ts.sumo, ts.id),
                env.reward_cfg["measure"],
                float(env.reward_cfg["waiting_scale"]),
                ts.last_measure,
            )
            return value

        return local_traffic_reward

    # --------------------------------------------------------------- helpers
    def _signals(self):
        return self.sumo_env.traffic_signals

    def green_phases(self) -> dict[str, int]:
        return {tls: ts.green_phase for tls, ts in self._signals().items()}

    def _now(self) -> float:
        return float(self.sumo_env.sim_step)

    def _mask(self, tls: str) -> np.ndarray:
        ts = self._signals()[tls]
        return self.constraints.mask(tls, self._now(), ts.time_since_last_phase_change)

    def _keep_only(self, tls: str) -> bool:
        if not self.constraints.state:
            return True
        mask = self._mask(tls)
        return bool(mask.sum() == 1 and mask[self._signals()[tls].green_phase])

    def _wrap_obs(self, raw: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        return {
            a: {"observations": raw[a].astype(np.float32), "action_mask": self._mask(a)}
            for a in self.agents
        }

    def _per_step_reward_bound(self) -> float:
        """Upper bound of |traffic term| any agent can receive in one decision step.

        Waiting: every vehicle that fits on the approach waits the whole step.
        Queue / pressure: normalised counts, bounded by approach length / max_distance.
        """
        measure = self.reward_cfg["measure"]
        bounds = []
        for topo in self.spec.topology.values():
            length = sum(seg.length for seg in topo.all_segments)
            if measure == "waiting":
                vehicles = length / features.VEHICLE_SPACING
                bounds.append(vehicles * self.delta_time / float(self.reward_cfg["waiting_scale"]))
            else:
                bounds.append(2.0 * length / self.spec.max_distance)
        factor = (
            1.0 + float(self.reward_cfg["alpha"])
            if self.reward_cfg["type"] == "neighborhood"
            else 1.0
        )
        return max(bounds) * factor

    def _gridlock_penalty_cap(self, reward_scale: dict[str, float]) -> float | None:
        """Cap of the ``auto`` penalty: ``gridlock_penalty_cap_std`` x the per-step std of
        the traffic term, measured on the scenario's random-policy rollout (D24)."""
        k = self.reward_cfg.get("gridlock_penalty_cap_std")
        if k is None:
            return None
        measure = self.reward_cfg["measure"]
        if measure not in reward_scale:
            raise ValueError(
                f"{self.manifest.name}: no measured reward scale for '{measure}'; run "
                "`python -m trafficmarl env reward-scale` and record it in the scenario config "
                "(scenario.reward_scale), or set env.reward.gridlock_penalty_cap_std: null"
            )
        return float(k) * float(reward_scale[measure])

    def gridlock_penalty(self) -> float:
        """Terminal penalty when an episode ends in gridlock.

        ``auto``: the per-step bound times the decision steps left before the
        time limit, so ending the episode by gridlock costs at least as much as
        any reward the remaining steps could have produced; capped at
        ``gridlock_penalty_cap_std`` per-step standard deviations (D24) so that
        a single terminal value does not dwarf every other reward.
        """
        value = self.reward_cfg.get("gridlock_penalty", "auto")
        if value != "auto":
            return float(value)
        remaining = math.ceil((self.time_limit - self._now()) / self.delta_time)
        penalty = self._penalty_bound * max(remaining, 1)
        return min(penalty, self._penalty_cap) if self._penalty_cap is not None else penalty

    # ------------------------------------------------------------ PettingZoo
    def observation_space(self, agent: str) -> spaces.Dict:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.split == "train":
            route = self.route_files[self._rng.integers(len(self.route_files))]
        else:
            route = self.route_files[self._episode % len(self.route_files)]
        self._episode += 1
        self.sumo_env._route = str(route)  # SUMO-RL reads the route file at each reset
        self.sumo_env.sumo_seed = int(self._rng.integers(2**31 - 1))
        self.route_file = Path(route).name
        self.meter.reset()
        self.constraints.state = {}
        raw = self.sumo_env.reset()
        self.sumo_env.reset_counters()
        self.constraints.reset(begin=self._now())
        self.agents = list(self.possible_agents)
        # Observations at t=0 were computed before the constraint state existed.
        raw = {a: self._signals()[a].compute_observation() for a in self.agents}
        self.log = EpisodeLog(
            traffic={a: [] for a in self.agents},
            penalty={a: [] for a in self.agents},
            reward={a: [] for a in self.agents},
        )
        obs = self._wrap_obs(raw)
        infos = {a: {"action_mask": obs[a]["action_mask"]} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset()")
        now = self._now()
        penalty = {}
        for agent in self.agents:
            ts = self._signals()[agent]
            switched = self.constraints.apply(
                agent, int(actions[agent]), now, ts.time_since_last_phase_change
            )
            penalty[agent] = float(self.reward_cfg["switch_penalty"]) if switched else 0.0
        raw_obs, traffic, dones, _ = self.sumo_env.step({a: int(actions[a]) for a in self.agents})
        for agent in self.agents:
            if self._signals()[agent].green_phase != self.constraints.state[agent].green_phase:
                raise RuntimeError(f"phase bookkeeping out of sync for {agent}")

        rewards = features.mix_rewards(
            traffic,
            penalty,
            self.reward_cfg["type"],
            {a: self.spec.topology[a].neighbours for a in self.agents},
            float(self.reward_cfg["alpha"]),
        )
        end_reason = self._end_reason(dones["__all__"])
        terminal = 0.0
        if end_reason == END_GRIDLOCK:
            terminal = self.gridlock_penalty()
            rewards = {a: r - terminal for a, r in rewards.items()}
            self.gridlock_count += 1
            logger.warning(
                "gridlock: scenario=%s route=%s t=%.0fs episode=%d penalty=%.1f (gridlocks: %d)",
                self.manifest.name, self.route_file, self._now(), self._episode, terminal,
                self.gridlock_count,
            )  # fmt: skip
        for a in self.agents:
            self.log.traffic[a].append(traffic[a])
            self.log.penalty[a].append(penalty[a])
            self.log.reward[a].append(rewards[a])

        obs = self._wrap_obs(raw_obs)
        terminated = end_reason in (END_EMPTY, END_GRIDLOCK)
        truncated = end_reason == END_TIME_LIMIT
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {
            a: {
                "action_mask": obs[a]["action_mask"],
                "traffic_reward": traffic[a],
                "switch_penalty": penalty[a],
                "gridlock": end_reason == END_GRIDLOCK,
            }
            for a in self.agents
        }
        if end_reason is not None:
            summary = self.episode_summary(end_reason)
            summary["gridlock_penalty"] = terminal
            for a in self.agents:
                infos[a]["episode"] = summary
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def _end_reason(self, time_limit_reached: bool) -> str | None:
        """Gridlock and empty network are real ends (terminated); the time limit truncates."""
        sumo = self.sumo_env
        now = self._now()
        vehicles_left = sumo.sumo.simulation.getMinExpectedNumber()
        if vehicles_left > 0 and now - sumo.last_arrival_time >= self.sim_cfg["gridlock_window"]:
            return END_GRIDLOCK
        if vehicles_left == 0 and now >= self.manifest.begin + self.manifest.duration:
            return END_EMPTY
        if time_limit_reached:
            return END_TIME_LIMIT
        return None

    def episode_summary(self, end_reason: str) -> dict[str, Any]:
        now = self._now()
        departed = self.sumo_env.departed
        return {
            "end_reason": end_reason,
            "sim_time": now,
            "route_file": self.route_file,
            "departed": departed,
            "arrived": self.sumo_env.arrived,
            "teleports": self.sumo_env.teleports,
            "teleport_rate": self.sumo_env.teleports / departed if departed else 0.0,
            "constraints": {a: self.constraints.counters(a) for a in self.possible_agents},
            "longest_red": {a: self.constraints.longest_red(a, now) for a in self.possible_agents},
        }

    def close(self) -> None:
        self.sumo_env.close()

    def render(self) -> None:
        return None
