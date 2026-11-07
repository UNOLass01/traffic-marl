"""FedAvg aggregation math and the server's accounting (D46)."""

import numpy as np
import pytest

from trafficmarl.federated.client import FederatedClient
from trafficmarl.federated.server import FedAvgServer, clip_update, fedavg, parameter_count


def _model(value: float) -> dict:
    return {"w": np.full((2, 2), value, np.float32), "b": np.full(3, value, np.float32)}


def test_fedavg_is_the_sample_weighted_mean():
    out = fedavg([_model(1.0), _model(3.0)], [3, 1])
    assert np.allclose(out["w"], 1.5)
    assert np.allclose(out["b"], 1.5)
    assert out["w"].dtype == np.float32


def test_fedavg_of_identical_models_changes_nothing():
    out = fedavg([_model(2.5)] * 4, [1, 2, 3, 4])
    assert np.allclose(out["w"], 2.5)


def test_fedavg_ignores_clients_without_samples():
    assert np.allclose(fedavg([_model(1.0), _model(9.0)], [5, 0])["w"], 1.0)


@pytest.mark.parametrize(
    ("models", "samples"),
    [([], []), ([_model(1.0)], [1, 2]), ([_model(1.0)], [0]), ([_model(1.0)], [-1])],
)
def test_fedavg_rejects_bad_inputs(models, samples):
    with pytest.raises(ValueError):
        fedavg(models, samples)


def test_fedavg_rejects_different_architectures():
    with pytest.raises(ValueError):
        fedavg([_model(1.0), {"w": np.zeros(1)}], [1, 1])


def test_clip_bounds_the_global_norm_and_leaves_small_updates():
    update = {"a": np.array([3.0]), "b": np.array([4.0])}  # norm 5
    clipped = clip_update(update, 1.0)
    assert np.isclose(np.sqrt(clipped["a"] ** 2 + clipped["b"] ** 2).item(), 1.0)
    assert clip_update(update, 10.0)["a"].item() == 3.0


def test_server_counts_bytes_both_ways():
    server = FedAvgServer()
    server.aggregate([_model(1.0), _model(2.0), _model(3.0)], [1, 1, 1])
    assert parameter_count(_model(0.0)) == 7
    assert server.rounds == 1
    assert server.bytes_up == 3 * 7 * 4
    assert server.bytes_down == 3 * 7 * 4


def test_noise_is_seeded_and_scaled_by_clients():
    def round_with(seed):
        server = FedAvgServer(clip_norm=1.0, noise_multiplier=2.0, seed=seed)
        server.aggregate([_model(0.0)] * 2, [1, 1])  # first round sets the reference
        return server.aggregate([_model(0.0)] * 2, [1, 1])

    assert np.array_equal(round_with(0)["w"], round_with(0)["w"])
    assert not np.array_equal(round_with(0)["w"], round_with(1)["w"])
    server = FedAvgServer(clip_norm=1.0, noise_multiplier=2.0, seed=0)
    big = {"w": np.zeros(100_000, np.float32)}
    server.aggregate([big] * 4, [1] * 4)
    noisy = server.aggregate([big] * 4, [1] * 4)
    assert noisy["w"].std() == pytest.approx(2.0 * 1.0 / 4, rel=0.05)


def test_privacy_needs_a_clip_norm():
    with pytest.raises(ValueError):
        FedAvgServer(noise_multiplier=1.0)


def test_client_exposes_only_parameters_and_resets_its_count():
    store = {"tls": _model(1.0)}
    client = FederatedClient(
        "tls", read=lambda c: store[c], write=lambda c, w: store.__setitem__(c, w)
    )
    client.record_samples(10)
    assert client.samples == 10
    assert np.allclose(client.parameters()["w"], 1.0)
    client.load(_model(5.0))
    assert np.allclose(store["tls"]["w"], 5.0)
    assert client.samples == 0
