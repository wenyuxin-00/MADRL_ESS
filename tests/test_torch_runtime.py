import numpy as np
import torch

from scripts.utils.torch_runtime import (
    PERFORMANCE_RUNTIME_MODE,
    STRICT_REPRO_RUNTIME_MODE,
    configure_torch_runtime,
    derive_worker_seed,
)
from configs.experiment_config import ExperimentConfig


def test_configure_torch_runtime_updates_config_and_returns_state():
    cfg = ExperimentConfig()
    cfg.runtime.device = torch.device("cpu")
    cfg.runtime.seed = 123
    cfg.runtime.execution_mode = PERFORMANCE_RUNTIME_MODE

    state = configure_torch_runtime(cfg, seed=cfg.runtime.seed)

    assert state.device.type == "cpu"
    assert state.mode == PERFORMANCE_RUNTIME_MODE
    assert state.seed == 123
    assert cfg.runtime.device.type == "cpu"
    assert cfg.runtime.seed == 123
    assert cfg.runtime.pin_memory is False
    assert cfg.runtime.non_blocking_transfers is False


def test_strict_runtime_mode_enables_deterministic_switches():
    cfg = ExperimentConfig()
    cfg.runtime.device = torch.device("cpu")
    cfg.runtime.seed = 7
    cfg.runtime.execution_mode = STRICT_REPRO_RUNTIME_MODE

    state = configure_torch_runtime(cfg, seed=cfg.runtime.seed)

    assert state.mode == STRICT_REPRO_RUNTIME_MODE
    assert state.allow_tf32 is False
    assert state.cudnn_deterministic is True
    assert state.deterministic_algorithms is True


def test_derive_worker_seed_is_stable_and_distinct():
    seed_a = derive_worker_seed(42, 0)
    seed_b = derive_worker_seed(42, 1)
    seed_c = derive_worker_seed(42, 0)

    assert isinstance(seed_a, int)
    assert seed_a != seed_b
    assert seed_a == seed_c


def test_runtime_seeding_replays_numpy_and_torch_random_streams():
    cfg = ExperimentConfig()
    cfg.runtime.device = torch.device("cpu")
    cfg.runtime.execution_mode = STRICT_REPRO_RUNTIME_MODE
    cfg.runtime.seed = 99

    configure_torch_runtime(cfg, seed=cfg.runtime.seed)
    first_numpy = np.random.rand(4)
    first_torch = torch.rand(4)

    configure_torch_runtime(cfg, seed=cfg.runtime.seed)
    second_numpy = np.random.rand(4)
    second_torch = torch.rand(4)

    assert np.allclose(first_numpy, second_numpy)
    assert torch.allclose(first_torch, second_torch)


def test_configure_torch_runtime_normalizes_cuda_device_without_index(monkeypatch):
    cfg = ExperimentConfig()
    cfg.runtime.device = torch.device("cuda")
    cfg.runtime.execution_mode = PERFORMANCE_RUNTIME_MODE
    cfg.runtime.seed = 0

    set_device_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: set_device_calls.append(str(device)))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    state = configure_torch_runtime(cfg, seed=cfg.runtime.seed)

    assert str(state.device) == "cuda:2"
    assert str(cfg.runtime.device) == "cuda:2"
    assert set_device_calls == ["cuda:2"]
