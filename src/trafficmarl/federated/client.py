"""Federated client: one traffic light's local policy, seen from the outside.

The client exposes only its parameters and how many samples produced them. Its
observations and trajectories stay inside its own RLlib module's training data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from trafficmarl.federated.server import Weights


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        read: Callable[[str], Weights],
        write: Callable[[str, Weights], None],
    ):
        self.client_id = client_id
        self._read, self._write = read, write
        self.samples = 0

    def record_samples(self, count: int) -> None:
        self.samples += count

    def parameters(self) -> Weights:
        return {k: np.asarray(v) for k, v in self._read(self.client_id).items()}

    def load(self, model: Weights) -> None:
        self._write(self.client_id, model)
        self.samples = 0
