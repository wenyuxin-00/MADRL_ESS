import numpy as np
import torch

from data.loaders.registry import build_dataset
from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from scripts.builder import build_env
from tests.support.helpers import make_case_dir, make_smoke_config


def test_dataset_returns_canonical_signals_schema(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_dataset")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    dataset = build_dataset(cfg, mode="train")

    episode = dataset.get_episode(0)

    assert set(episode.keys()) == {"signals", "meta", "history_signals", "history_length", "history_timestamps"}
    assert {"wholesale_price", "load", "pv"}.issubset(set(episode["signals"].keys()))
    assert episode["signals"]["wholesale_price"].shape == (cfg.env.episode_limit,)
    assert episode["signals"]["load"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert episode["signals"]["pv"].shape == (cfg.env.episode_limit, cfg.env.num_agents)


def test_env_returns_structured_observation_schema(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_env")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    env = build_env(cfg, mode="test")

    obs, reset_info = env.reset(episode_idx=0)
    expected_local_dim = env.observation_schema["local"][1]

    assert set(obs.keys()) == {"local", "wholesale_price_seq", "load_seq", "pv_seq", "adjacency", "safety_local"}
    assert obs["local"].shape == (cfg.env.num_agents, expected_local_dim)
    assert obs["wholesale_price_seq"].shape == (cfg.env.future_horizon + 1,)
    assert obs["load_seq"].shape == (cfg.env.num_agents, cfg.env.future_horizon + 1)
    assert obs["pv_seq"].shape == (cfg.env.num_agents, cfg.env.future_horizon + 1)
    assert obs["adjacency"].shape == (cfg.env.num_agents, cfg.env.num_agents)
    assert obs["safety_local"].shape == (cfg.env.num_agents, 5)
    assert env.observation_layout["safety_local"]["scope"] == "per_agent"
    assert env.observation_layout["wholesale_price_seq"]["scope"] == "shared"
    assert env.observation_layout["load_seq"]["scope"] == "per_agent"
    assert env.observation_layout["pv_seq"]["scope"] == "per_agent"
    assert "episode_idx" in reset_info
    env.close()


def test_dummy_vec_env_stacks_structured_observations(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_vec")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    vec_env = DummyVecEnv(2, lambda: build_env(cfg, mode="train"))

    try:
        obs, reset_infos = vec_env.reset()
        expected_local_dim = vec_env.envs[0].observation_schema["local"][1]
        assert obs["local"].shape == (2, cfg.env.num_agents, expected_local_dim)
        assert obs["wholesale_price_seq"].shape == (2, cfg.env.future_horizon + 1)
        assert obs["pv_seq"].shape == (2, cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert obs["safety_local"].shape == (2, cfg.env.num_agents, 5)
        assert len(reset_infos) == 2

        action_list = [np.zeros((2, 2), dtype=np.float32) for _ in range(cfg.env.num_agents)]
        next_obs, reward, terminated, truncated, info_list = vec_env.step(action_list)

        assert next_obs["load_seq"].shape == (2, cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert next_obs["pv_seq"].shape == (2, cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert reward.shape == (2, cfg.env.num_agents, 1)
        assert terminated.shape == (2, cfg.env.num_agents, 1)
        assert truncated.shape == (2, cfg.env.num_agents, 1)
        assert len(info_list) == 2
    finally:
        vec_env.close()


def test_subproc_vec_env_stacks_structured_observations(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_subproc")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    vec_env = SubprocVecEnv(2, cfg, mode="train")

    try:
        obs, reset_infos = vec_env.reset()
        assert obs["local"].shape[:2] == (2, cfg.env.num_agents)
        assert obs["local"].shape[-1] > 0
        assert obs["wholesale_price_seq"].shape == (2, cfg.env.future_horizon + 1)
        assert obs["pv_seq"].shape == (2, cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert obs["safety_local"].shape == (2, cfg.env.num_agents, 5)
        assert len(reset_infos) == 2

        action_list = [np.zeros((2, 2), dtype=np.float32) for _ in range(cfg.env.num_agents)]
        next_obs, reward, terminated, truncated, info_list = vec_env.step(action_list)

        assert next_obs["load_seq"].shape == (2, cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert next_obs["pv_seq"].shape == (2, cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert reward.shape == (2, cfg.env.num_agents, 1)
        assert terminated.shape == (2, cfg.env.num_agents, 1)
        assert truncated.shape == (2, cfg.env.num_agents, 1)
        assert len(info_list) == 2
    finally:
        vec_env.close()


def test_subproc_vec_env_workers_ignore_parent_require_cuda(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_subproc_require_cuda")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.runtime.require_cuda = True
    cfg.runtime.device = torch.device("cpu")
    vec_env = SubprocVecEnv(2, cfg, mode="train")

    try:
        obs, reset_infos = vec_env.reset()
        assert obs["local"].shape[0] == 2
        assert len(reset_infos) == 2
    finally:
        vec_env.close()


def test_env_step_respects_soc_bounds(tmp_path):
    case_dir = make_case_dir(tmp_path, "soc_bounds")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.soc_min = 0.2
    cfg.env.soc_max = 0.8
    cfg.env.init_soc = 0.2
    env = build_env(cfg, mode="test")

    try:
        env.reset(episode_idx=0)
        _, _, _, _, info = env.step(
            [np.array([0.0, 1.0], dtype=np.float32) for _ in range(cfg.env.num_agents)]
        )
        assert np.allclose(info["soc_t"], 0.2)
        assert np.allclose(info["soc_next"], 0.2)
        assert np.allclose(info["e_min"], cfg.env.soc_min * info["battery_capacity_kwh"])
        assert np.all(info["p_lower"] >= -cfg.env.max_charge_rate)
    finally:
        env.close()


def test_dummy_vec_env_uses_distinct_reward_instances(tmp_path):
    case_dir = make_case_dir(tmp_path, "reward_instances")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    vec_env = DummyVecEnv(2, lambda: build_env(cfg, mode="train"))

    try:
        assert vec_env.envs[0].reward_fn is not vec_env.envs[1].reward_fn
    finally:
        vec_env.close()
