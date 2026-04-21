from __future__ import annotations
import numpy as np
import torch
NestedArray = dict | np.ndarray
def stack_nested(items: list[NestedArray]) -> NestedArray:
    first = items[0]
    if isinstance(first, dict):
        return {key: stack_nested([item[key] for item in items]) for key in first}
    return np.stack([np.asarray(item, dtype=np.float32) for item in items], axis=0)

def add_batch_dim(payload: NestedArray) -> NestedArray:
    if isinstance(payload, dict):
        return {key: add_batch_dim(value) for key, value in payload.items()}
    return np.expand_dims(np.asarray(payload, dtype=np.float32), axis=0)

def index_nested(payload: NestedArray, index: int) -> NestedArray:
    if isinstance(payload, dict):
        return {key: index_nested(value, index) for key, value in payload.items()}
    return np.asarray(payload, dtype=np.float32)[index].copy()

def to_torch_nested(payload: NestedArray, device: torch.device | str) -> NestedArray:
    if isinstance(payload, dict):
        return {key: to_torch_nested(value, device) for key, value in payload.items()}
    return torch.as_tensor(payload, dtype=torch.float32, device=device)
