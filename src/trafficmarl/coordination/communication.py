"""Communication volume of the coordination schemes, in floats per decision step.

Two phases are counted separately because they have different costs in a
deployment:

* **Execution** (the trained controllers on the street): only the extended
  observation needs messages. Each directed neighbour link carries one
  message of ``message_dim`` floats per decision step.
* **Training**: the observation messages, plus what the reward scheme needs.
  ``neighborhood`` sends each agent's traffic term along every directed
  neighbour link (one float per link). ``global`` is counted through an
  aggregator: every agent sends its traffic term and receives the network mean
  (two floats per agent), which is cheaper than all-to-all exchange.
  ``local`` needs nothing.

Rewards are not used once the policy is trained, so reward sharing costs
nothing at execution time.

Centralized methods add what their central component must collect
(:func:`centralized_floats`): CTDE critics only during training, the centralized
single agent at execution as well.
"""

from __future__ import annotations


def floats_per_step(
    *,
    n_agents: int,
    n_links: int,
    message_dim: int,
    extended: bool,
    reward_type: str,
) -> dict[str, int]:
    """Floats exchanged per decision step, at execution and during training.

    ``n_links`` counts directed neighbour links (sum over agents of their
    number of neighbours).
    """
    observation = n_links * message_dim if extended else 0
    rewards = {"local": 0, "neighborhood": n_links, "global": 2 * n_agents}
    if reward_type not in rewards:
        raise ValueError(f"Unknown reward type '{reward_type}'")
    return {"execution": observation, "training": observation + rewards[reward_type]}


def centralized_floats(
    method: str, *, n_agents: int, obs_dim: int, message_dim: int
) -> dict[str, int]:
    """Floats per decision step that a method's central component adds to ``floats_per_step``.

    * ``mappo``: the critic's global state, one summary per agent, gathered and
      redistributed (``2 * n_agents * message_dim``), during training only;
    * ``qmix`` / ``maddpg``: the mixer or critic is trained at one place from every
      agent's observation, action and reward (``n_agents * (obs_dim + 2)``);
    * ``centralized_ppo``: the same upstream at training, plus the action sent back;
      at execution, every observation goes up and every action comes down.
    """
    if method == "mappo":
        return {"execution": 0, "training": 2 * n_agents * message_dim}
    if method in ("qmix", "maddpg"):
        return {"execution": 0, "training": n_agents * (obs_dim + 2)}
    if method == "centralized_ppo":
        return {"execution": n_agents * (obs_dim + 1), "training": n_agents * (obs_dim + 3)}
    return {"execution": 0, "training": 0}
