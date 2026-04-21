from __future__ import annotations
from dataclasses import dataclass
import numpy as np
def validate_parallel_episode_sampling_mode(mode: str) -> str:
    resolved = str(mode).strip().lower()
    if resolved not in {"unique_active", "per_env_rng"}:
        raise ValueError(
            "train.parallel_episode_sampling must be one of "
            "['unique_active', 'per_env_rng'], "
            f"got {mode!r}."
        )
    return resolved

def validate_wave_done_flags(done_flags: list[bool], *, env_name: str) -> None:
    done_count = int(sum(bool(flag) for flag in done_flags))
    if done_count in {0, len(done_flags)}:
        return
    raise RuntimeError(
        f"{env_name} with parallel_episode_sampling='unique_active' requires synchronized "
        f"episode boundaries, but got {done_count}/{len(done_flags)} envs done in one step."
    )

@dataclass
class ParallelEpisodeSampler:

    num_available_episodes: int
    base_seed: int | None
    num_envs: int
    def __post_init__(self) -> None:
        total = int(self.num_available_episodes)
        if total <= 0:
            raise ValueError("ParallelEpisodeSampler requires num_available_episodes > 0.")
        env_count = int(self.num_envs)
        if env_count <= 0:
            raise ValueError("ParallelEpisodeSampler requires num_envs > 0.")
        self.num_available_episodes = total
        self.num_envs = env_count
        self._rng = np.random.default_rng(None if self.base_seed is None else int(self.base_seed))

    def next_wave(self) -> list[int]:
        indices: list[int] = []
        while len(indices) < self.num_envs:
            permutation = self._rng.permutation(self.num_available_episodes).tolist()
            remaining = self.num_envs - len(indices)
            indices.extend(int(value) for value in permutation[:remaining])
        return indices
