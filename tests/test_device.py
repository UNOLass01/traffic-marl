import pytest
import torch

from trafficmarl.utils.device import resolve_device


def test_cpu_always_available():
    assert resolve_device("cpu") == "cpu"


def test_auto_matches_cuda_availability():
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


def test_cuda_without_gpu_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")
