from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

PERFORMANCE_RUNTIME_MODE = "performance"
STRICT_REPRO_RUNTIME_MODE = "strict_reproducibility"
VALID_RUNTIME_MODES = {PERFORMANCE_RUNTIME_MODE, STRICT_REPRO_RUNTIME_MODE}


@dataclass(frozen=True)
class TorchRuntimeState:
    device: torch.device
    seed: int | None
    pin_memory: bool
    non_blocking_transfers: bool


def _runtime_cfg(runtime_or_cfg: Any | None):
    if runtime_or_cfg is None or isinstance(runtime_or_cfg, (TorchRuntimeState, str, torch.device)):
        return None
    return getattr(runtime_or_cfg, "runtime", runtime_or_cfg)


def _cfg_value(runtime_cfg, name: str, default):
    value = default if runtime_cfg is None or not hasattr(runtime_cfg, name) else getattr(runtime_cfg, name)
    return default if value is None else value


def resolve_device(device: str | torch.device | TorchRuntimeState | None = None) -> torch.device:
    if isinstance(device, TorchRuntimeState):
        return device.device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_runtime_mode(mode: str | None = None) -> str:
    resolved = PERFORMANCE_RUNTIME_MODE if mode is None else str(mode).strip().lower()
    if resolved not in VALID_RUNTIME_MODES:
        raise ValueError(f"Unknown runtime mode '{mode}', expected one of {sorted(VALID_RUNTIME_MODES)}.")
    return resolved


def derive_worker_seed(base_seed: int | None, *components: int) -> int | None:
    if base_seed is None:
        return None
    seed = int(base_seed) & 4294967295
    for component in components:
        component = int(component)
        if component != 0:
            seed = (seed * 1664525 + 1013904223 + component * 374761393) & 4294967295
    return int(seed)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(runtime_or_cfg: Any | str | torch.device | None = None, *, device: str | torch.device | None = None, require_cuda: bool | None = None, seed: int | None = None, mode: str | None = None, worker_rank: int = 0) -> TorchRuntimeState:
    if isinstance(runtime_or_cfg, TorchRuntimeState):
        return runtime_or_cfg
    runtime_cfg = _runtime_cfg(runtime_or_cfg)
    requested_device = device if device is not None else runtime_or_cfg if isinstance(runtime_or_cfg, (str, torch.device)) else _cfg_value(runtime_cfg, "device", None)
    resolved_device = resolve_device(requested_device)
    resolved_mode = resolve_runtime_mode(mode if mode is not None else _cfg_value(runtime_cfg, "execution_mode", PERFORMANCE_RUNTIME_MODE))
    cuda_available = torch.cuda.is_available()
    resolved_require_cuda = bool(_cfg_value(runtime_cfg, "require_cuda", False) if require_cuda is None else require_cuda)
    if seed is not None:
        resolved_seed = int(seed)
    elif runtime_cfg is not None and hasattr(runtime_cfg, "seed"):
        resolved_seed = int(_cfg_value(runtime_cfg, "seed", 0))
    else:
        resolved_seed = None
    if resolved_require_cuda and (resolved_device.type != "cuda" or not cuda_available):
        raise RuntimeError("CUDA is required for this entry point, but the current PyTorch runtime cannot provide it.")
    if resolved_device.type == "cuda" and not cuda_available:
        raise RuntimeError(f"Requested device '{resolved_device}', but CUDA is not available in the current PyTorch build.")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device(f"cuda:{torch.cuda.current_device() if cuda_available else 0}")
    pin_memory = bool(_cfg_value(runtime_cfg, "pin_memory", resolved_device.type == "cuda"))
    non_blocking_transfers = bool(_cfg_value(runtime_cfg, "non_blocking_transfers", resolved_device.type == "cuda"))
    worker_seed = derive_worker_seed(resolved_seed, worker_rank)
    if worker_seed is not None:
        _seed_everything(worker_seed, resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)
    state = TorchRuntimeState(device=resolved_device, seed=worker_seed, pin_memory=pin_memory, non_blocking_transfers=non_blocking_transfers)
    if runtime_cfg is not None:
        for name, value in {"device": state.device, "execution_mode": resolved_mode, "seed": 0 if resolved_seed is None else int(resolved_seed), "pin_memory": state.pin_memory, "non_blocking_transfers": state.non_blocking_transfers}.items():
            setattr(runtime_cfg, name, value)
        runtime_cfg.require_cuda = resolved_require_cuda
    return state


def describe_device(device: str | torch.device | TorchRuntimeState) -> dict[str, object]:
    resolved = resolve_device(device)
    summary: dict[str, object] = {"device": str(resolved), "cuda_available": bool(torch.cuda.is_available())}
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return summary
    device_index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    summary.update({"name": torch.cuda.get_device_name(device_index), "capability": f"{properties.major}.{properties.minor}", "total_memory_gb": round(properties.total_memory / 1024 ** 3, 2)})
    return summary
