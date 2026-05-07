from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from configs.cfg import Cfg
from data.share_data import ShareData
from envs.grid_env import build_env
from envs.vec_env import SubprocVecEnv
from scripts.madrl import OBS_KEYS, ReplayBuffer, SCHEMES, train_madrl_scheme


def _small_cfg() -> Cfg:
    cfg = Cfg()
    return replace(
        cfg,
        env=replace(cfg.env, episode_steps=2, train_window_days=1),
        obs=replace(cfg.obs, sequence_length=2),
        train=replace(cfg.train, train_episodes=1, num_envs=2, batch_size=2, learning_starts=1, actor_learning_starts=1, n_step_return=1),
        runtime=replace(cfg.runtime, device="cpu"),
    )


def _share_data(cfg: Cfg, root: Path) -> ShareData:
    n, steps, horizon = int(cfg.env.num_agents), int(cfg.env.episode_steps), int(cfg.obs.sequence_length)
    timestamps = np.asarray([[f"2020-01-01T00:{15 * step:02d}:00" for step in range(steps)] for _ in range(3)])
    price = np.full((3, steps), 0.20, dtype=np.float32)
    load = np.full((3, steps, n), 3.0, dtype=np.float32)
    pv = np.full((3, steps), 0.4, dtype=np.float32)
    price_seq = np.full((3, steps, horizon), 0.20, dtype=np.float32)
    load_seq = np.full((3, steps, horizon, n), 3.0, dtype=np.float32)
    pv_seq = np.full((3, steps, horizon), 0.4, dtype=np.float32)
    split = {
        "timestamps": timestamps, "price": price, "load": load, "pv": pv,
        "perfect_price_seq": price_seq, "perfect_load_seq": load_seq, "perfect_pv_seq": pv_seq,
        "lstm_price_seq": price_seq, "lstm_load_seq": load_seq, "lstm_pv_seq": pv_seq,
    }
    return ShareData(root=root, manifest={"sequence_length": horizon, "num_agents": n, "episode_steps": steps}, train=split, eval=split)


def test_subproc_vec_env_contract(tmp_path: Path) -> None:
    cfg = _small_cfg(); share_data = _share_data(cfg, tmp_path)
    factory = partial(build_env, cfg, "train", forecast_mode="lstm", share_data=share_data)
    env = SubprocVecEnv(int(cfg.train.num_envs), factory, seed=int(cfg.runtime.seed), parallel_episode_sampling=str(cfg.train.parallel_episode_sampling))
    try:
        obs, _ = env.reset()
        assert set(OBS_KEYS) <= set(obs)
        action = np.zeros((int(cfg.train.num_envs), int(cfg.env.num_agents), int(cfg.model.action_dim)), dtype=np.float32)
        next_obs, reward, done, truncated, infos = env.step(action)
        assert set(OBS_KEYS) <= set(next_obs)
        assert reward.shape == (int(cfg.train.num_envs), int(cfg.env.num_agents))
        assert done.shape == (int(cfg.train.num_envs),)
        assert truncated.shape == (int(cfg.train.num_envs),)
        assert len(infos) == int(cfg.train.num_envs)
    finally:
        env.close()


def test_replay_buffer_sample_contract(tmp_path: Path) -> None:
    cfg = _small_cfg(); share_data = _share_data(cfg, tmp_path)
    env = SubprocVecEnv(1, partial(build_env, cfg, "train", forecast_mode="lstm", share_data=share_data), seed=int(cfg.runtime.seed), parallel_episode_sampling=str(cfg.train.parallel_episode_sampling))
    try:
        buffer = ReplayBuffer(cfg, 1)
        obs, _ = env.reset()
        action = np.zeros((1, int(cfg.env.num_agents), int(cfg.model.action_dim)), dtype=np.float32)
        next_obs, reward, done, _, _ = env.step(action)
        buffer.add_batch(obs, action, reward, next_obs, done)
        batch = buffer.sample(2, np.random.default_rng(0))
        assert set(batch["obs"]) == set(OBS_KEYS)
        assert batch["action"].shape == (2, int(cfg.env.num_agents), int(cfg.model.action_dim))
        assert batch["reward"].shape == (2, int(cfg.env.num_agents))
    finally:
        env.close()


def test_train_madrl_scheme_uses_subproc_vec_env_and_timers(tmp_path: Path) -> None:
    cfg = _small_cfg(); share_data = _share_data(cfg, tmp_path)
    result = train_madrl_scheme(cfg, tmp_path, share_data, SCHEMES[0], episodes=1)
    meta = result["meta"]
    assert meta["vec_env_kind"] == "subproc"
    for key in ("action_sample_s", "env_step_s", "replay_add_s", "sample_update_s", "other_s"):
        assert key in meta
        assert float(meta[key]) >= 0.0
