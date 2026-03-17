"""Small helpers for nested numpy/torch batch structures."""

from __future__ import annotations

import numpy as np
import torch


def stack_nested(items):
    """Stack a list of identically-structured numpy payloads along axis 0."""
    first = items[0]
    if isinstance(first, dict):
        return {key: stack_nested([item[key] for item in items]) for key in first}
    return np.stack([np.asarray(item, dtype=np.float32) for item in items], axis=0)


def add_batch_dim(payload):
    """Add a leading batch dimension to a nested numpy payload."""
    if isinstance(payload, dict):
        return {key: add_batch_dim(value) for key, value in payload.items()}
    return np.expand_dims(np.asarray(payload, dtype=np.float32), axis=0)


def index_nested(payload, index: int):
    """Take one batch element from a nested numpy payload."""
    if isinstance(payload, dict):
        return {key: index_nested(value, index) for key, value in payload.items()}
    return np.asarray(payload, dtype=np.float32)[index].copy()


def to_torch_nested(payload, device):
    """Convert a nested numpy payload to float32 torch tensors."""
    if isinstance(payload, dict):
        return {key: to_torch_nested(value, device) for key, value in payload.items()}
    return torch.as_tensor(payload, dtype=torch.float32, device=device)
