from __future__ import annotations
import os
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
    mode: str
    seed: int | None
    worker_rank: int
    matmul_precision: str
    allow_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    pin_memory: bool
    non_blocking_transfers: bool
    enable_amp: bool
    amp_dtype: str
    enable_compile: bool
    compile_mode: str
    compile_fullgraph: bool
    compile_dynamic: bool

def _unwrap_runtime_config(runtime_or_cfg: Any | None):
    if runtime_or_cfg is None:
        return None, None
    if isinstance(runtime_or_cfg, TorchRuntimeState):
        return None, runtime_or_cfg
    if isinstance(runtime_or_cfg, (str, torch.device)):
        return None, None
    if hasattr(runtime_or_cfg, "runtime"):
        return runtime_or_cfg.runtime, None
    return runtime_or_cfg, None

def _runtime_attr(runtime_cfg, name: str, default):
    if runtime_cfg is None or not hasattr(runtime_cfg, name):
        return default
    value = getattr(runtime_cfg, name)
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
        raise ValueError(
            f"Unknown runtime mode '{mode}', expected one of {sorted(VALID_RUNTIME_MODES)}."
        )
    return resolved

def derive_worker_seed(base_seed: int | None, *components: int) -> int | None:
    if base_seed is None:
        return None

    seed = int(base_seed) & 0xFFFFFFFF
    for component in components:
        component = int(component)
        if component == 0:
            continue
        seed = (seed * 1664525 + 1013904223 + component * 374761393) & 0xFFFFFFFF
    return int(seed)

def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def configure_torch_runtime(
    runtime_or_cfg: Any | str | torch.device | None = None,
    *,
    device: str | torch.device | None = None,
    require_cuda: bool | None = None,
    seed: int | None = None,
    mode: str | None = None,
    worker_rank: int = 0,
) -> TorchRuntimeState:
    runtime_cfg, runtime_state = _unwrap_runtime_config(runtime_or_cfg)
    if runtime_state is not None:
        return runtime_state

    requested_device = device
    if requested_device is None:
        if isinstance(runtime_or_cfg, (str, torch.device)):
            requested_device = runtime_or_cfg
        else:
            requested_device = _runtime_attr(runtime_cfg, "device", None)

    resolved_device = resolve_device(requested_device)
    resolved_mode = resolve_runtime_mode(
        mode if mode is not None else _runtime_attr(runtime_cfg, "execution_mode", PERFORMANCE_RUNTIME_MODE)
    )
    resolved_require_cuda = bool(
        _runtime_attr(runtime_cfg, "require_cuda", False)
        if require_cuda is None
        else require_cuda
    )
    resolved_seed = (
        int(_runtime_attr(runtime_cfg, "seed", 0))
        if seed is None and runtime_cfg is not None and hasattr(runtime_cfg, "seed")
        else (None if seed is None else int(seed))
    )
    strict_mode = resolved_mode == STRICT_REPRO_RUNTIME_MODE
    matmul_precision = str(_runtime_attr(runtime_cfg, "matmul_precision", "high"))
    allow_tf32 = bool(_runtime_attr(runtime_cfg, "allow_tf32", not strict_mode))
    cudnn_deterministic = bool(_runtime_attr(runtime_cfg, "cudnn_deterministic", strict_mode))
    cudnn_benchmark = bool(_runtime_attr(runtime_cfg, "cudnn_benchmark", not strict_mode))
    deterministic_algorithms = bool(
        _runtime_attr(runtime_cfg, "use_deterministic_algorithms", strict_mode)
    )
    pin_memory = bool(_runtime_attr(runtime_cfg, "pin_memory", resolved_device.type == "cuda"))
    non_blocking_transfers = bool(
        _runtime_attr(runtime_cfg, "non_blocking_transfers", resolved_device.type == "cuda")
    )
    enable_amp = bool(
        _runtime_attr(
            runtime_cfg,
            "enable_amp",
            resolved_device.type == "cuda" and resolved_mode == PERFORMANCE_RUNTIME_MODE,
        )
    )
    amp_dtype = str(_runtime_attr(runtime_cfg, "amp_dtype", "bfloat16"))
    enable_compile = bool(
        _runtime_attr(
            runtime_cfg,
            "enable_compile",
            resolved_device.type == "cuda" and resolved_mode == PERFORMANCE_RUNTIME_MODE,
        )
    )
    compile_mode = str(_runtime_attr(runtime_cfg, "compile_mode", "reduce-overhead"))
    compile_fullgraph = bool(_runtime_attr(runtime_cfg, "compile_fullgraph", False))
    compile_dynamic = bool(_runtime_attr(runtime_cfg, "compile_dynamic", False))
    cuda_available = torch.cuda.is_available()
    if resolved_require_cuda and (resolved_device.type != "cuda" or not cuda_available):
        raise RuntimeError(
            "CUDA is required for this entry point, but the current PyTorch runtime cannot provide it."
        )

    if resolved_device.type == "cuda" and not cuda_available:
        raise RuntimeError(
            f"Requested device '{resolved_device}', but CUDA is not available in the current PyTorch build."
        )

    if resolved_device.type == "cuda" and resolved_device.index is None:
        try:
            resolved_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        except (AttributeError, RuntimeError, ValueError):
            resolved_device = torch.device("cuda:0")

    if strict_mode and resolved_seed is not None and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except (AttributeError, RuntimeError, ValueError):
        pass

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
        except (AttributeError, RuntimeError):
            pass

    if cuda_available and hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        except (AttributeError, RuntimeError):
            pass

    if cuda_available and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark and not cudnn_deterministic
        try:
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except (AttributeError, RuntimeError):
            pass

    if resolved_seed is not None:
        worker_seed = derive_worker_seed(resolved_seed, worker_rank)
        if worker_seed is not None:
            _seed_everything(worker_seed, resolved_device)
    else:
        worker_seed = None

    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)

    if enable_compile and hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        try:
            torch._dynamo.config.suppress_errors = True
        except (AttributeError, RuntimeError):
            pass

    state = TorchRuntimeState(
        device=resolved_device,
        mode=resolved_mode,
        seed=worker_seed,
        worker_rank=int(worker_rank),
        matmul_precision=matmul_precision,
        allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark and not cudnn_deterministic,
        cudnn_deterministic=cudnn_deterministic,
        deterministic_algorithms=deterministic_algorithms,
        pin_memory=pin_memory,
        non_blocking_transfers=non_blocking_transfers,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
        enable_compile=enable_compile,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
    )
    if runtime_cfg is not None:
        runtime_cfg.device = state.device
        runtime_cfg.execution_mode = state.mode
        runtime_cfg.seed = int(resolved_seed) if resolved_seed is not None else 0
        runtime_cfg.worker_rank = int(worker_rank)
        runtime_cfg.matmul_precision = state.matmul_precision
        runtime_cfg.allow_tf32 = state.allow_tf32
        runtime_cfg.cudnn_benchmark = state.cudnn_benchmark
        runtime_cfg.cudnn_deterministic = state.cudnn_deterministic
        runtime_cfg.use_deterministic_algorithms = state.deterministic_algorithms
        runtime_cfg.pin_memory = state.pin_memory
        runtime_cfg.non_blocking_transfers = state.non_blocking_transfers
        runtime_cfg.enable_amp = state.enable_amp
        runtime_cfg.amp_dtype = state.amp_dtype
        runtime_cfg.enable_compile = state.enable_compile
        runtime_cfg.compile_mode = state.compile_mode
        runtime_cfg.compile_fullgraph = state.compile_fullgraph
        runtime_cfg.compile_dynamic = state.compile_dynamic
        runtime_cfg.require_cuda = resolved_require_cuda

    return state

def describe_device(device: str | torch.device | TorchRuntimeState) -> dict[str, object]:
    resolved = resolve_device(device)
    summary: dict[str, object] = {
        "device": str(resolved),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return summary

    device_index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    summary.update(
        {
            "name": torch.cuda.get_device_name(device_index),
            "capability": f"{properties.major}.{properties.minor}",
            "total_memory_gb": round(properties.total_memory / (1024 ** 3), 2),
        }
    )
    return summary
