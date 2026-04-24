import numpy as np
import torch

from scripts.builder import build_env
from scripts.utils.torch_runtime import stack_nested
from scripts.utils.replay_buffer import ReplayBuffer, to_torch_batch
from tests.support.helpers import make_case_dir, make_smoke_config


def test_replay_buffer_stores_and_samples_canonical_batch(tmp_path):
    case_dir = make_case_dir(tmp_path, "buffer")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    env = build_env(cfg, mode="train")
    try:
        cfg.runtime.observation_schema = dict(env.observation_schema)
        cfg.runtime.action_dim = int(env.action_space[0].shape[0])

        obs, _ = env.reset(episode_idx=0)
        action_n = [np.zeros((cfg.runtime.action_dim,), dtype=np.float32) for _ in range(cfg.env.num_agents)]
        next_obs, reward, terminated, truncated, _ = env.step(action_n)
        terminated_array = np.asarray(terminated, dtype=np.float32)
        truncated_array = np.asarray(truncated, dtype=np.float32)

        batched_obs = stack_nested([obs, obs])
        batched_next_obs = stack_nested([next_obs, next_obs])
        batched_action = np.zeros((2, cfg.env.num_agents, cfg.runtime.action_dim), dtype=np.float32)
        batched_reward = np.stack(
            [np.asarray(reward, dtype=np.float32).reshape(cfg.env.num_agents, 1)] * 2,
            axis=0,
        )
        batched_terminated = np.stack(
            [terminated_array.reshape(cfg.env.num_agents, 1)] * 2,
            axis=0,
        )
        batched_truncated = np.stack(
            [truncated_array.reshape(cfg.env.num_agents, 1)] * 2,
            axis=0,
        )

        buffer = ReplayBuffer(cfg)
        buffer.store_transitions_batched(
            batched_obs,
            batched_action,
            batched_reward,
            batched_next_obs,
            batched_terminated,
            batched_truncated,
        )

        batch = buffer.sample()
        expected_local_dim = env.observation_schema["local"][1]

        assert set(batch.keys()) == {"obs", "action", "reward", "next_obs", "terminated", "truncated"}
        assert batch["obs"]["local"].shape == (
            cfg.train.batch_size,
            cfg.env.num_agents,
            expected_local_dim,
        )
        assert batch["action"].shape == (cfg.train.batch_size, cfg.env.num_agents, cfg.runtime.action_dim)
        assert batch["reward"].shape == (cfg.train.batch_size, cfg.env.num_agents, 1)
        assert batch["terminated"].shape == (cfg.train.batch_size, cfg.env.num_agents, 1)
        assert batch["truncated"].shape == (cfg.train.batch_size, cfg.env.num_agents, 1)

        torch_batch = to_torch_batch(batch, cfg.runtime.device)
        assert isinstance(torch_batch["obs"]["local"], torch.Tensor)
        assert torch_batch["obs"]["local"].dtype == torch.float32

        direct_torch_batch = buffer.sample_torch(cfg.runtime.device)
        assert isinstance(direct_torch_batch["obs"]["local"], torch.Tensor)
        assert direct_torch_batch["obs"]["local"].shape == torch_batch["obs"]["local"].shape
        assert direct_torch_batch["action"].shape == torch_batch["action"].shape
        assert direct_torch_batch["reward"].dtype == torch.float32

        buffer._sample_indices = lambda: np.array([0, 1], dtype=np.int64)
        direct_torch_batch["action"][0, 0, 0] = 123.0
        repeated_torch_batch = buffer.sample_torch(cfg.runtime.device)
        assert repeated_torch_batch["action"][0, 0, 0].item() != 123.0
    finally:
        env.close()
