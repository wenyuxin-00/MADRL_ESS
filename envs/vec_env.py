from __future__ import annotations
import numpy as np
from envs.parallel_episode_sampling import ParallelEpisodeSampler, validate_parallel_episode_sampling_mode, validate_wave_done_flags
from scripts.utils.nested import stack_nested
from scripts.utils.torch_runtime import derive_worker_seed

def split_batched_actions(actions_n_batched, env_idx: int, num_agents: int):
    return [np.asarray(actions_n_batched[agent_id][env_idx], dtype=np.float32) for agent_id in range(int(num_agents))]

def stack_step_outputs(obs_list, reward_list, terminated_list, truncated_list, info_list):
    batched_obs = stack_nested(obs_list)
    batched_reward = np.stack(reward_list, axis=0)
    batched_terminated = np.stack(terminated_list, axis=0)
    batched_truncated = np.stack(truncated_list, axis=0)
    return (batched_obs, batched_reward, batched_terminated, batched_truncated, info_list)

class DummyVecEnv:

    def __init__(self, num_envs, env_fn_or_cls, cfg=None, mode: str='train', seed: int | None=None, parallel_episode_sampling: str='unique_active'):
        self.num_envs = int(num_envs)
        self.parallel_episode_sampling = validate_parallel_episode_sampling_mode(parallel_episode_sampling)
        if cfg is None:
            self.envs = [env_fn_or_cls() for _ in range(self.num_envs)]
            self.num_agents = self.envs[0].n
        else:
            self.envs = [env_fn_or_cls(cfg, mode=mode) for _ in range(self.num_envs)]
            self.num_agents = cfg.env.num_agents
        self.base_seed = self._resolve_base_seed(seed)
        self._seeded_envs = [False] * self.num_envs
        self._episode_sampler: ParallelEpisodeSampler | None = None
        self._next_wave_indices: list[int] | None = None
        if self.parallel_episode_sampling == 'unique_active':
            self._initialize_episode_sampler()

    def _resolve_base_seed(self, seed: int | None) -> int | None:
        if seed is not None:
            return int(seed)
        first_env = self.envs[0] if self.envs else None
        runtime = getattr(getattr(first_env, 'cfg', None), 'runtime', None)
        runtime_seed = getattr(runtime, 'seed', None)
        return None if runtime_seed is None else int(runtime_seed)

    def _initialize_episode_sampler(self) -> None:
        first_env = self.envs[0]
        num_available_episodes = int(getattr(first_env, 'num_available_episodes', 0))
        episode_length = int(getattr(first_env, 'episode_length', 0))
        for env_idx, env in enumerate(self.envs[1:], start=1):
            if int(getattr(env, 'num_available_episodes', 0)) != num_available_episodes:
                raise RuntimeError("DummyVecEnv requires matching num_available_episodes across envs in parallel_episode_sampling='unique_active' mode.")
            if int(getattr(env, 'episode_length', 0)) != episode_length:
                raise RuntimeError("DummyVecEnv requires matching episode_length across envs in parallel_episode_sampling='unique_active' mode.")
        self._episode_sampler = ParallelEpisodeSampler(num_available_episodes=num_available_episodes, base_seed=self.base_seed, num_envs=self.num_envs)

    def _consume_initial_seed(self, env_idx: int) -> int | None:
        if self._seeded_envs[env_idx]:
            return None
        self._seeded_envs[env_idx] = True
        return derive_worker_seed(self.base_seed, env_idx) if self.base_seed is not None else None

    @staticmethod
    def _reset_env(env, *, episode_idx: int | None=None, seed: int | None=None):
        kwargs = {}
        if episode_idx is not None:
            kwargs['episode_idx'] = int(episode_idx)
        if seed is not None:
            kwargs['seed'] = int(seed)
        return env.reset(**kwargs)

    def _prime_next_wave_indices(self) -> None:
        if self.parallel_episode_sampling != 'unique_active':
            self._next_wave_indices = None
            return
        if self._episode_sampler is None:
            raise RuntimeError('DummyVecEnv episode sampler is not initialized.')
        self._next_wave_indices = self._episode_sampler.next_wave()

    def reset(self):
        obs_list = []
        info_list = []
        episode_indices: list[int | None]
        if self.parallel_episode_sampling == 'unique_active':
            if self._episode_sampler is None:
                raise RuntimeError('DummyVecEnv episode sampler is not initialized.')
            episode_indices = self._episode_sampler.next_wave()
        else:
            episode_indices = [None] * self.num_envs
        for env_idx, env in enumerate(self.envs):
            obs, info = self._reset_env(env, episode_idx=episode_indices[env_idx], seed=self._consume_initial_seed(env_idx))
            obs_list.append(obs)
            info_list.append(info)
        if self.parallel_episode_sampling == 'unique_active':
            self._prime_next_wave_indices()
        else:
            self._next_wave_indices = None
        return (stack_nested(obs_list), info_list)

    def step(self, actions_n_batched):
        obs_list: list[object] = []
        reward_list: list[np.ndarray] = []
        terminated_list: list[np.ndarray] = []
        truncated_list: list[np.ndarray] = []
        info_list: list[dict] = []
        done_flags: list[bool] = []
        for env_idx, env in enumerate(self.envs):
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            obs, reward, terminated, truncated, info = env.step(action_n)
            episode_done = bool(info.get('episode_done', False) or np.all(np.logical_or(np.asarray(terminated), np.asarray(truncated))))
            obs_list.append(obs)
            reward_list.append(np.asarray(reward, dtype=np.float32).reshape(self.num_agents, 1))
            terminated_list.append(np.asarray(terminated, dtype=np.float32).reshape(self.num_agents, 1))
            truncated_list.append(np.asarray(truncated, dtype=np.float32).reshape(self.num_agents, 1))
            info_list.append(dict(info))
            done_flags.append(bool(episode_done))
        if self.parallel_episode_sampling == 'unique_active':
            validate_wave_done_flags(done_flags, env_name='DummyVecEnv')
            if any(done_flags):
                if self._next_wave_indices is None:
                    raise RuntimeError('DummyVecEnv next episode wave was not primed before step().')
                current_wave_indices = list(self._next_wave_indices)
                for env_idx, env in enumerate(self.envs):
                    reset_obs, reset_info = self._reset_env(env, episode_idx=current_wave_indices[env_idx])
                    info_list[env_idx]['episode_done'] = True
                    info_list[env_idx]['reset_info'] = reset_info
                    obs_list[env_idx] = reset_obs
                self._prime_next_wave_indices()
            else:
                for info in info_list:
                    info['episode_done'] = False
        else:
            for env_idx, env in enumerate(self.envs):
                if not done_flags[env_idx]:
                    info_list[env_idx]['episode_done'] = False
                    continue
                reset_obs, reset_info = self._reset_env(env)
                info_list[env_idx]['episode_done'] = True
                info_list[env_idx]['reset_info'] = reset_info
                obs_list[env_idx] = reset_obs
        return stack_step_outputs(obs_list, reward_list, terminated_list, truncated_list, info_list)

    def close(self):
        for env in self.envs:
            env.close()
