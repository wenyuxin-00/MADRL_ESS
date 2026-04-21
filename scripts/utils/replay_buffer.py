from __future__ import annotations
import numpy as np
import torch
from scripts.utils.nested import NestedArray, to_torch_nested

def _allocate_nested_storage(schema: dict[str, tuple[int, ...]], size: int) -> dict[str, np.ndarray]:
    return {key: np.zeros((size, *tuple(shape)), dtype=np.float32) for key, shape in schema.items()}

def _store_nested_range(storage: dict[str, np.ndarray], payload: NestedArray, source: slice, target: slice) -> None:
    assert isinstance(payload, dict)
    for key, value in payload.items():
        storage[key][target] = np.asarray(value[source], dtype=np.float32)

def _sample_nested(storage: dict[str, np.ndarray], indices: np.ndarray, out: dict[str, np.ndarray] | None=None) -> dict[str, np.ndarray]:
    if out is None:
        return {key: np.asarray(value[indices], dtype=np.float32).copy() for key, value in storage.items()}
    for key, value in storage.items():
        np.take(value, indices, axis=0, out=out[key])
    return out

def _tensor_nested_to_device(payload: dict[str, torch.Tensor], device: torch.device, *, non_blocking: bool) -> dict[str, torch.Tensor]:
    copy_on_cpu = device.type == 'cpu'
    return {key: tensor.to(device=device, non_blocking=non_blocking, copy=copy_on_cpu) for key, tensor in payload.items()}

class ReplayBuffer:

    def __init__(self, cfg: object) -> None:
        if not getattr(cfg.runtime, 'observation_schema', None):
            raise ValueError('ReplayBuffer requires cfg.runtime.observation_schema to be populated.')
        self.buffer_size = int(cfg.train.buffer_size)
        self.batch_size = int(cfg.train.batch_size)
        self.num_agents = int(cfg.env.num_agents)
        self.action_dim = int(cfg.runtime.action_dim)
        self.observation_schema = {key: tuple(shape) for key, shape in dict(cfg.runtime.observation_schema).items()}
        self.obs_storage = _allocate_nested_storage(self.observation_schema, self.buffer_size)
        self.next_obs_storage = _allocate_nested_storage(self.observation_schema, self.buffer_size)
        shape = (self.buffer_size, self.num_agents)
        self.action_storage = np.zeros((*shape, self.action_dim), dtype=np.float32)
        self.reward_storage = np.zeros((*shape, 1), dtype=np.float32)
        self.done_storage = np.zeros((*shape, 1), dtype=np.float32)
        self.position = 0
        self.current_size = 0
        self._torch_staging_cache: dict[tuple[object, ...], dict[str, object]] = {}

    def _sample_indices(self) -> np.ndarray:
        return np.random.choice(self.current_size, size=self.batch_size, replace=self.current_size < self.batch_size)

    def store_transitions_batched(self, obs: NestedArray, action: np.ndarray, reward: np.ndarray, next_obs: NestedArray, done: np.ndarray) -> None:
        action, reward, done = (np.asarray(value, dtype=np.float32) for value in (action, reward, done))
        num_envs = int(action.shape[0])
        first_block = min(num_envs, self.buffer_size - self.position)
        blocks = [(slice(0, first_block), slice(self.position, self.position + first_block))]
        if num_envs > first_block:
            blocks.append((slice(first_block, num_envs), slice(0, num_envs - first_block)))
        for source, target in blocks:
            if source.stop == source.start:
                continue
            _store_nested_range(self.obs_storage, obs, source, target)
            _store_nested_range(self.next_obs_storage, next_obs, source, target)
            self.action_storage[target], self.reward_storage[target], self.done_storage[target] = (action[source], reward[source], done[source])
        self.position = (self.position + num_envs) % self.buffer_size
        self.current_size = min(self.current_size + num_envs, self.buffer_size)

    def sample(self) -> dict[str, NestedArray]:
        indices = self._sample_indices()
        return {'obs': _sample_nested(self.obs_storage, indices), 'action': np.asarray(self.action_storage[indices], dtype=np.float32).copy(), 'reward': np.asarray(self.reward_storage[indices], dtype=np.float32).copy(), 'next_obs': _sample_nested(self.next_obs_storage, indices), 'done': np.asarray(self.done_storage[indices], dtype=np.float32).copy()}

    def sample_torch(self, device: torch.device | str, *, pin_memory: bool=False, non_blocking: bool=False) -> dict[str, NestedArray]:
        resolved_device = torch.device(device)
        indices = self._sample_indices()
        staging = self._get_torch_staging_cache(resolved_device, pin_memory=pin_memory)
        _sample_nested(self.obs_storage, indices, staging['obs_numpy'])
        _sample_nested(self.next_obs_storage, indices, staging['next_obs_numpy'])
        for name, storage in (('action', self.action_storage), ('reward', self.reward_storage), ('done', self.done_storage)):
            np.take(storage, indices, axis=0, out=staging[f'{name}_numpy'])
        copy_on_cpu = resolved_device.type == 'cpu'
        return {'obs': _tensor_nested_to_device(staging['obs_tensors'], resolved_device, non_blocking=non_blocking), 'action': staging['action_tensor'].to(device=resolved_device, non_blocking=non_blocking, copy=copy_on_cpu), 'reward': staging['reward_tensor'].to(device=resolved_device, non_blocking=non_blocking, copy=copy_on_cpu), 'next_obs': _tensor_nested_to_device(staging['next_obs_tensors'], resolved_device, non_blocking=non_blocking), 'done': staging['done_tensor'].to(device=resolved_device, non_blocking=non_blocking, copy=copy_on_cpu)}

    def _get_torch_staging_cache(self, device: torch.device, *, pin_memory: bool) -> dict[str, object]:
        use_pinned = bool(pin_memory and device.type == 'cuda')
        schema_signature = tuple(sorted(((key, tuple(shape)) for key, shape in self.observation_schema.items())))
        key = (self.batch_size, schema_signature, device.type, use_pinned, self.num_agents, self.action_dim)
        cache = self._torch_staging_cache.get(key)
        if cache is not None:
            return cache

        def _alloc_nested() -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
            tensors = {name: torch.empty((self.batch_size, *shape), dtype=torch.float32, pin_memory=use_pinned) for name, shape in self.observation_schema.items()}
            return (tensors, {name: tensor.numpy() for name, tensor in tensors.items()})
        obs_tensors, obs_numpy = _alloc_nested()
        next_obs_tensors, next_obs_numpy = _alloc_nested()
        scalars = {name: torch.empty((self.batch_size, self.num_agents, 1 if name != 'action' else self.action_dim), dtype=torch.float32, pin_memory=use_pinned) for name in ('action', 'reward', 'done')}
        cache = {'obs_tensors': obs_tensors, 'obs_numpy': obs_numpy, 'next_obs_tensors': next_obs_tensors, 'next_obs_numpy': next_obs_numpy, 'action_tensor': scalars['action'], 'action_numpy': scalars['action'].numpy(), 'reward_tensor': scalars['reward'], 'reward_numpy': scalars['reward'].numpy(), 'done_tensor': scalars['done'], 'done_numpy': scalars['done'].numpy()}
        self._torch_staging_cache[key] = cache
        return cache

def to_torch_batch(batch: dict[str, NestedArray], device: torch.device | str) -> dict[str, NestedArray]:
    return {name: to_torch_nested(batch[name], device) for name in ('obs', 'action', 'reward', 'next_obs', 'done')}
