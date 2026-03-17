import numpy as np
import torch

from common.nested import stack_nested
from common.replay_buffer import ReplayBuffer, to_torch_batch
from core.builder import build_env
from tests.helpers import make_case_dir, make_smoke_config


def test_replay_buffer_stores_and_samples_canonical_batch(tmp_path):
    case_dir = make_case_dir(tmp_path, "buffer")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    env = build_env(cfg, mode="train")
    try:
        cfg.runtime.observation_schema = dict(env.observation_schema)
        cfg.runtime.action_dim = int(env.action_space[0].shape[0])

        obs = env.reset(episode_idx=0)
        action_n = [np.zeros((1,), dtype=np.float32) for _ in range(cfg.env.num_agents)]
        next_obs, reward, done, _ = env.step(action_n)

        batched_obs = stack_nested([obs, obs])
        batched_next_obs = stack_nested([next_obs, next_obs])
        batched_action = np.zeros((2, cfg.env.num_agents, cfg.runtime.action_dim), dtype=np.float32)
        batched_reward = np.stack(
            [np.asarray(reward, dtype=np.float32).reshape(cfg.env.num_agents, 1)] * 2,
            axis=0,
        )
        batched_done = np.stack(
            [np.asarray(done, dtype=np.float32).reshape(cfg.env.num_agents, 1)] * 2,
            axis=0,
        )

        buffer = ReplayBuffer(cfg)
        buffer.store_transitions_batched(
            batched_obs,
            batched_action,
            batched_reward,
            batched_next_obs,
            batched_done,
        )

        batch = buffer.sample()

        assert set(batch.keys()) == {"obs", "action", "reward", "next_obs", "done"}
        assert batch["obs"]["local"].shape == (cfg.train.batch_size, cfg.env.num_agents, 5)
        assert batch["action"].shape == (cfg.train.batch_size, cfg.env.num_agents, cfg.runtime.action_dim)
        assert batch["reward"].shape == (cfg.train.batch_size, cfg.env.num_agents, 1)

        torch_batch = to_torch_batch(batch, cfg.runtime.device)
        assert isinstance(torch_batch["obs"]["local"], torch.Tensor)
        assert torch_batch["obs"]["local"].dtype == torch.float32
    finally:
        env.close()