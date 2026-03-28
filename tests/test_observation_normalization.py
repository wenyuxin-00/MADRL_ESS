from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.builder import build_env
from tests.support.helpers import make_case_dir, make_smoke_config


def test_observation_builder_returns_normalized_obs_and_raw_view(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_normalized_vs_raw")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    env = build_env(cfg, mode="test")

    try:
        obs, _ = env.reset(episode_idx=0)
        raw_obs = env.obs_builder.build_raw(env)

        assert env.obs_builder.normalizer is not None
        assert obs["local"].shape == raw_obs["local"].shape
        assert not np.allclose(obs["price_seq"], raw_obs["price_seq"])
        assert not np.allclose(obs["load_seq"], raw_obs["load_seq"])
        assert not np.allclose(obs["pv_seq"], raw_obs["pv_seq"])
        assert np.max(np.abs(obs["price_seq"])) <= 1.0 + 1e-6
        assert np.max(np.abs(obs["load_seq"])) <= 1.0 + 1e-6
        assert np.min(obs["pv_seq"]) >= -1e-6
        assert np.max(obs["pv_seq"]) <= 1.2 + 1e-6
        assert np.max(np.abs(obs["local"][:, :4])) <= 1.0 + 1e-6
        assert np.max(np.abs(obs["local"][:, 4:])) <= 1.0 + 1e-6
    finally:
        env.close()


def test_test_observation_uses_train_year_price_stats(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_train_frozen_price_stats")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    price_path = case_dir / "data" / "processed" / "prosumer" / "price.csv"
    frame = pd.read_csv(price_path)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Europe/Berlin")
    test_mask = timestamps.dt.year == 2020
    frame.loc[test_mask, "price"] = np.linspace(5.0, 5.5, int(test_mask.sum()), dtype=np.float32)
    frame.to_csv(price_path, index=False)

    env = build_env(cfg, mode="test")
    try:
        obs, _ = env.reset(episode_idx=0)
        raw_obs = env.obs_builder.build_raw(env)
        state = dict(cfg.runtime.observation_normalization_state or {})
        price_state = dict(state["price"])
        q_high = float(np.asarray(price_state["q_high"]).reshape(-1)[0])
        median = float(np.asarray(price_state["median"]).reshape(-1)[0])
        iqr = float(np.asarray(price_state["iqr"]).reshape(-1)[0])
        tanh_scale = float(price_state["tanh_scale"])
        expected = float(np.tanh(((min(float(raw_obs["price_seq"][0]), q_high) - median) / iqr) / tanh_scale))

        assert np.isclose(float(obs["price_seq"][0]), expected)
        assert state["signature"]["train_year"] == 2019
        assert q_high < 1.0
    finally:
        env.close()
