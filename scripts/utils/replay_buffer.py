"""Experience replay buffer utilities."""

from __future__ import annotations

import numpy as np
import torch

from scripts.utils.nested import NestedArray, to_torch_nested


def _allocate_nested_storage(schema: dict[str, tuple[int, ...]], buffer_size: int) -> dict[str, np.ndarray]:
    return {
        key: np.zeros((buffer_size, *tuple(shape)), dtype=np.float32)
        for key, shape in schema.items()
    }


def _store_nested_at(
    storage: dict[str, np.ndarray],
    payload: NestedArray,
    *,
    source_index: int,
    target_index: int,
) -> None:
    assert isinstance(payload, dict)
    for key, value in payload.items():
        storage[key][target_index] = np.asarray(value[source_index], dtype=np.float32)


def _store_nested_range(
    storage: dict[str, np.ndarray],
    payload: NestedArray,
    *,
    source_slice: slice,
    target_slice: slice,
) -> None:
    assert isinstance(payload, dict)
    for key, value in payload.items():
        storage[key][target_slice] = np.asarray(value[source_slice], dtype=np.float32)


def _sample_nested(storage: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value[indices], dtype=np.float32).copy()
        for key, value in storage.items()
    }


def _sample_nested_into(
    storage: dict[str, np.ndarray],
    indices: np.ndarray,
    out: dict[str, np.ndarray],
) -> None:
    for key, value in storage.items():
        np.take(value, indices, axis=0, out=out[key])


def _sample_nested_torch(
    storage: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    *,
    pin_memory: bool,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key, value in storage.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value[indices]))
        if pin_memory and device.type == "cuda":
            tensor = tensor.pin_memory()
        batch[key] = tensor.to(device=device, non_blocking=non_blocking)
    return batch


def _tensor_to_numpy_view(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


def _tensor_nested_to_device(
    payload: dict[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    copy_on_cpu = device.type == "cpu"
    return {
        key: tensor.to(device=device, non_blocking=non_blocking, copy=copy_on_cpu)
        for key, tensor in payload.items()
    }


class ReplayBuffer:
    """Preallocated ring-buffer for canonical transition dictionaries."""

    def __init__(self, cfg: object) -> None:
        if not getattr(cfg.runtime, "observation_schema", None):
            raise ValueError("ReplayBuffer requires cfg.runtime.observation_schema to be populated.")

        self.buffer_size = int(cfg.train.buffer_size)
        self.batch_size = int(cfg.train.batch_size)
        self.num_agents = int(cfg.env.num_agents)
        self.action_dim = int(cfg.runtime.action_dim)
        self.observation_schema = {
            key: tuple(shape)
            for key, shape in dict(cfg.runtime.observation_schema).items()
        }
        self.obs_storage = _allocate_nested_storage(self.observation_schema, self.buffer_size)
        self.next_obs_storage = _allocate_nested_storage(self.observation_schema, self.buffer_size)
        self.action_storage = np.zeros(
            (self.buffer_size, self.num_agents, self.action_dim),
            dtype=np.float32,
        )
        self.reward_storage = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.done_storage = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.position = 0
        self.current_size = 0
        self._torch_staging_cache: dict[tuple[object, ...], dict[str, object]] = {}

    def _sample_indices(self) -> np.ndarray:
        return np.random.choice(
            self.current_size,
            size=self.batch_size,
            replace=self.current_size < self.batch_size,
        )

    def store_transitions_batched(
        self,
        obs: NestedArray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: NestedArray,
        done: np.ndarray,
    ) -> None:
        """Store one batched env rollout step."""
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        num_envs = int(action.shape[0])

        first_block = min(num_envs, self.buffer_size - self.position)
        if first_block > 0:
            target_slice = slice(self.position, self.position + first_block)
            source_slice = slice(0, first_block)
            _store_nested_range(
                self.obs_storage,
                obs,
                source_slice=source_slice,
                target_slice=target_slice,
            )
            _store_nested_range(
                self.next_obs_storage,
                next_obs,
                source_slice=source_slice,
                target_slice=target_slice,
            )
            self.action_storage[target_slice] = action[source_slice]
            self.reward_storage[target_slice] = reward[source_slice]
            self.done_storage[target_slice] = done[source_slice]

        remaining = num_envs - first_block
        if remaining > 0:
            target_slice = slice(0, remaining)
            source_slice = slice(first_block, num_envs)
            _store_nested_range(
                self.obs_storage,
                obs,
                source_slice=source_slice,
                target_slice=target_slice,
            )
            _store_nested_range(
                self.next_obs_storage,
                next_obs,
                source_slice=source_slice,
                target_slice=target_slice,
            )
            self.action_storage[target_slice] = action[source_slice]
            self.reward_storage[target_slice] = reward[source_slice]
            self.done_storage[target_slice] = done[source_slice]

        self.position = (self.position + num_envs) % self.buffer_size
        self.current_size = min(self.current_size + num_envs, self.buffer_size)

    def sample(self) -> dict[str, NestedArray]:
        """Sample a canonical transition batch as numpy arrays."""
        indices = self._sample_indices()
        return {
            "obs": _sample_nested(self.obs_storage, indices),
            "action": np.asarray(self.action_storage[indices], dtype=np.float32).copy(),
            "reward": np.asarray(self.reward_storage[indices], dtype=np.float32).copy(),
            "next_obs": _sample_nested(self.next_obs_storage, indices),
            "done": np.asarray(self.done_storage[indices], dtype=np.float32).copy(),
        }

    def sample_torch(
        self,
        device: torch.device | str,
        *,
        pin_memory: bool = False,
        non_blocking: bool = False,
    ) -> dict[str, NestedArray]:
        """Sample a canonical transition batch directly as torch tensors."""
        resolved_device = torch.device(device)
        indices = self._sample_indices()
        staging = self._get_torch_staging_cache(
            resolved_device,
            pin_memory=pin_memory,
        )
        _sample_nested_into(self.obs_storage, indices, staging["obs_numpy"])
        np.take(self.action_storage, indices, axis=0, out=staging["action_numpy"])
        np.take(self.reward_storage, indices, axis=0, out=staging["reward_numpy"])
        _sample_nested_into(self.next_obs_storage, indices, staging["next_obs_numpy"])
        np.take(self.done_storage, indices, axis=0, out=staging["done_numpy"])
        return {
            "obs": _tensor_nested_to_device(
                staging["obs_tensors"],
                resolved_device,
                non_blocking=non_blocking,
            ),
            "action": staging["action_tensor"].to(
                device=resolved_device,
                non_blocking=non_blocking,
                copy=resolved_device.type == "cpu",
            ),
            "reward": staging["reward_tensor"].to(
                device=resolved_device,
                non_blocking=non_blocking,
                copy=resolved_device.type == "cpu",
            ),
            "next_obs": _tensor_nested_to_device(
                staging["next_obs_tensors"],
                resolved_device,
                non_blocking=non_blocking,
            ),
            "done": staging["done_tensor"].to(
                device=resolved_device,
                non_blocking=non_blocking,
                copy=resolved_device.type == "cpu",
            ),
        }

    def _get_torch_staging_cache(
        self,
        device: torch.device,
        *,
        pin_memory: bool,
    ) -> dict[str, object]:
        use_pinned = bool(pin_memory and device.type == "cuda")
        schema_signature = tuple(sorted((key, tuple(shape)) for key, shape in self.observation_schema.items()))
        cache_key = (
            self.batch_size,
            schema_signature,
            device.type,
            use_pinned,
            self.num_agents,
            self.action_dim,
        )
        cache = self._torch_staging_cache.get(cache_key)
        if cache is not None:
            return cache

        obs_tensors = {
            key: torch.empty(
                (self.batch_size, *shape),
                dtype=torch.float32,
                pin_memory=use_pinned,
            )
            for key, shape in self.observation_schema.items()
        }
        next_obs_tensors = {
            key: torch.empty(
                (self.batch_size, *shape),
                dtype=torch.float32,
                pin_memory=use_pinned,
            )
            for key, shape in self.observation_schema.items()
        }
        action_tensor = torch.empty(
            (self.batch_size, self.num_agents, self.action_dim),
            dtype=torch.float32,
            pin_memory=use_pinned,
        )
        reward_tensor = torch.empty(
            (self.batch_size, self.num_agents, 1),
            dtype=torch.float32,
            pin_memory=use_pinned,
        )
        done_tensor = torch.empty(
            (self.batch_size, self.num_agents, 1),
            dtype=torch.float32,
            pin_memory=use_pinned,
        )

        cache = {
            "obs_tensors": obs_tensors,
            "obs_numpy": {key: _tensor_to_numpy_view(tensor) for key, tensor in obs_tensors.items()},
            "next_obs_tensors": next_obs_tensors,
            "next_obs_numpy": {
                key: _tensor_to_numpy_view(tensor)
                for key, tensor in next_obs_tensors.items()
            },
            "action_tensor": action_tensor,
            "action_numpy": _tensor_to_numpy_view(action_tensor),
            "reward_tensor": reward_tensor,
            "reward_numpy": _tensor_to_numpy_view(reward_tensor),
            "done_tensor": done_tensor,
            "done_numpy": _tensor_to_numpy_view(done_tensor),
        }
        self._torch_staging_cache[cache_key] = cache
        return cache

    @staticmethod
    def _numpy_batch_to_torch(
        payload: np.ndarray,
        device: torch.device,
        *,
        pin_memory: bool,
        non_blocking: bool,
    ) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(payload))
        if pin_memory and device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=non_blocking)


def to_torch_batch(batch: dict[str, NestedArray], device: torch.device | str) -> dict[str, NestedArray]:
    """Convert a sampled canonical batch to torch tensors."""
    return {
        "obs": to_torch_nested(batch["obs"], device),
        "action": to_torch_nested(batch["action"], device),
        "reward": to_torch_nested(batch["reward"], device),
        "next_obs": to_torch_nested(batch["next_obs"], device),
        "done": to_torch_nested(batch["done"], device),
    }
