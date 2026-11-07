"""Federated aggregation server. It only ever receives model parameters.

FedAvg (McMahan et al., AISTATS 2017): the new global model is the mean of the
clients' models weighted by their number of local samples. Optionally, updates
are protected the way DP-FedAvg does (McMahan et al., ICLR 2018): each client's
update is clipped to a maximum L2 norm and Gaussian noise is added to the mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Weights = dict[str, np.ndarray]


def fedavg(models: list[Weights], samples: list[int]) -> Weights:
    """Sample-weighted mean of the clients' parameters."""
    if not models or len(models) != len(samples):
        raise ValueError("fedavg needs one sample count per client model")
    if min(samples) < 0 or sum(samples) == 0:
        raise ValueError("sample counts must be non-negative with a positive total")
    keys = models[0].keys()
    if any(m.keys() != keys for m in models):
        raise ValueError("client models have different parameters")
    shares = np.asarray(samples, dtype=np.float64) / sum(samples)
    return {
        k: sum(share * m[k] for share, m in zip(shares, models, strict=True)).astype(
            models[0][k].dtype
        )
        for k in keys
    }


def clip_update(update: Weights, max_norm: float) -> Weights:
    """Scale ``update`` down so that its global L2 norm is at most ``max_norm``."""
    norm = float(np.sqrt(sum(np.sum(np.square(v)) for v in update.values())))
    factor = min(1.0, max_norm / norm) if norm > 0 else 1.0
    return {k: v * factor for k, v in update.items()}


def parameter_count(model: Weights) -> int:
    return int(sum(v.size for v in model.values()))


@dataclass
class FedAvgServer:
    """Holds the global model and counts what crosses the network.

    ``noise_multiplier`` > 0 with ``clip_norm`` turns on the differential-privacy
    variant: the noise std on the mean is ``noise_multiplier * clip_norm / n_clients``.
    """

    clip_norm: float | None = None
    noise_multiplier: float = 0.0
    seed: int = 0
    global_model: Weights | None = None
    rounds: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.noise_multiplier > 0 and self.clip_norm is None:
            raise ValueError("differential privacy needs clip_norm")

    def aggregate(self, models: list[Weights], samples: list[int]) -> Weights:
        """One round: receive every client's model, return the new global model."""
        if self.clip_norm is not None and self.global_model is not None:
            reference = self.global_model
            models = [
                {k: reference[k] + v for k, v in clip_update(
                    {k: m[k] - reference[k] for k in m}, self.clip_norm).items()}
                for m in models
            ]  # fmt: skip
        new = fedavg(models, samples)
        if self.noise_multiplier > 0 and self.global_model is not None:
            std = self.noise_multiplier * self.clip_norm / len(models)
            new = {
                k: (v + self._rng.normal(0.0, std, v.shape)).astype(v.dtype) for k, v in new.items()
            }
        self.global_model = new
        self.rounds += 1
        size = sum(int(v.size) * v.dtype.itemsize for v in new.values())
        self.bytes_up += size * len(models)
        self.bytes_down += size * len(models)
        return new
