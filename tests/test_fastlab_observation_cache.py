from __future__ import annotations

from copy import deepcopy

import numpy as np

from data.loaders.registry import build_dataset
from envs.fastlab.grid_env_fastlab import GridEnvFastLab
from envs.fastlab.observation_builder_fastlab import CachedObservationBuilderFastLab
from envs.grid.core.grid_types import GridStepResult
from envs.grid_env import GridEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.normalization import build_observation_normalizer
from envs.rewards import NormalReward
from predictors.registry import build_forecaster
from scripts.recorders.episode_recorder import append_step_record, init_episode_record
from scripts.utils.madrl_observation_cache_lab import ObservationCacheStore, build_or_load_observation_cache
from tests.support.helpers import make_smoke_config


class _FakeGridCore:
    def __init__(self, n_agents: int) -> None:
        self.n_agents = int(n_agents)
        self.n_buses = 8
        self.n_lines = 6
        self.n_trafos = 1
        self.last_pf_error = ""

    def reset(self, base_load_kw, base_pv_kw) -> None:
        del base_load_kw, base_pv_kw

    def step(self, p_batt_kw, base_load_kw):
        del base_load_kw
        n = len(p_batt_kw)
        return GridStepResult(
            converged=True,
            vm_pu=np.ones(self.n_buses, dtype=np.float32),
            va_degree=np.zeros(self.n_buses, dtype=np.float32),
            line_loading_pct=np.zeros(self.n_lines, dtype=np.float32),
            trafo_loading_pct=np.zeros(self.n_trafos, dtype=np.float32),
            p_mw_from=np.zeros(self.n_lines, dtype=np.float32),
            agent_vm_pu=np.ones(n, dtype=np.float32),
            v_violation=np.zeros(n, dtype=np.float32),
            line_violation=0.0,
            trafo_violation=0.0,
            l_violation=0.0,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
            n_trafos=self.n_trafos,
            bus_v_excess=np.zeros(self.n_buses, dtype=np.float32),
            line_excess=np.zeros(self.n_lines, dtype=np.float32),
            trafo_excess=np.zeros(self.n_trafos, dtype=np.float32),
            psi_v_raw=0.0,
            psi_line_raw=0.0,
            psi_trafo_raw=0.0,
        )


def _assert_nested_close(lhs, rhs) -> None:
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        left_value = lhs[key]
        right_value = rhs[key]
        if isinstance(left_value, dict):
            _assert_nested_close(left_value, right_value)
            continue
        left_array = np.asarray(left_value, dtype=np.float32)
        right_array = np.asarray(right_value, dtype=np.float32)
        assert left_array.shape == right_array.shape
        assert np.allclose(left_array, right_array)


def _make_lstm_cfg(tmp_path):
    cfg = make_smoke_config(tmp_path, algorithm="MATD3")
    cfg.forecast.type = "lstm"
    cfg.forecast.history_window = 4
    cfg.forecast.lstm_hidden_size = 8
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0
    cfg.forecast.lstm_batch_size = 4
    cfg.forecast.lstm_epochs = 1
    cfg.forecast.lstm_artifact_root = tmp_path / "artifacts" / "forecast" / "lstm"
    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
    cfg.runtime.device = "cpu"
    return cfg


def _build_stepwise_episode_forecast_matrix(
    forecaster,
    history: np.ndarray,
    *,
    signal_name: str,
    timestamps,
    horizon: int,
) -> np.ndarray:
    values = np.asarray(history, dtype=np.float32)
    rows: list[np.ndarray] = []
    for step_idx in range(values.shape[0]):
        history_slice = values[: step_idx + 1]
        timestamp_slice = list(timestamps[: step_idx + 1])
        prediction = forecaster.predict(
            history_slice,
            horizon,
            signal_name=signal_name,
            history_timestamps=timestamp_slice,
        )
        rows.append(np.asarray(prediction, dtype=np.float32))
    if values.ndim == 1:
        return np.stack(rows, axis=0).astype(np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def test_fastlab_cache_matches_default_builder_for_lstm(tmp_path) -> None:
    cfg = _make_lstm_cfg(tmp_path)
    cache_result = build_or_load_observation_cache(cfg, split="train", refresh=True)
    store = ObservationCacheStore(cache_result.cache_dir)

    dataset = build_dataset(cfg, mode="train")
    reward_fn = NormalReward(cfg)
    normalizer = build_observation_normalizer(cfg)
    base_env = GridEnv(
        cfg,
        mode="train",
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=build_forecaster(cfg),
        obs_builder=DefaultObservationBuilder(
            local_features=cfg.obs.local_features,
            sequence_features=cfg.obs.sequence_features,
            future_horizon=cfg.env.future_horizon,
            adjacency_type=cfg.obs.adjacency_type,
            normalizer=normalizer,
        ),
        grid_core=_FakeGridCore(cfg.env.num_agents),
    )

    try:
        base_env.reset(episode_idx=0)
        episode_cache = store.episode(0)
        for _ in range(2):
            raw_obs = base_env.obs_builder.build_raw(base_env)
            step_idx = int(base_env.cur_step)
            assert np.allclose(episode_cache["calendar_time"][step_idx], raw_obs["local"][:, :4])
            assert np.allclose(episode_cache["price_seq"][step_idx], raw_obs["price_seq"])
            assert np.allclose(episode_cache["load_seq"][step_idx], raw_obs["load_seq"])
            assert np.allclose(episode_cache["pv_seq"][step_idx], raw_obs["pv_seq"])
            base_env.step([np.zeros((1,), dtype=np.float32) for _ in range(cfg.env.num_agents)])
    finally:
        base_env.close()


def test_lstm_forecaster_predict_episode_matrix_matches_stepwise_predict(tmp_path) -> None:
    cfg = _make_lstm_cfg(tmp_path)
    forecaster = build_forecaster(cfg)
    dataset = build_dataset(cfg, mode="train")
    episode = dataset.get_episode(0)
    signals = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in dict(episode.get("signals", {})).items()
    }
    meta = dict(episode.get("meta", {}))
    timestamps = list(meta.get("timestamps") or [])
    horizon = int(cfg.env.future_horizon) + 1

    forecaster.reset()
    forecaster.set_episode(signals, meta)

    for signal_name in ("price", "load", "pv"):
        expected = _build_stepwise_episode_forecast_matrix(
            forecaster,
            signals[signal_name],
            signal_name=signal_name,
            timestamps=timestamps,
            horizon=horizon,
        )
        actual = forecaster.predict_episode_matrix(
            signals[signal_name],
            horizon,
            signal_name=signal_name,
            history_timestamps=timestamps,
            batch_size=8,
        )
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


def test_fastlab_env_matches_base_env_and_minimal_info_history(tmp_path) -> None:
    cfg = make_smoke_config(tmp_path, algorithm="MATD3")
    cache_result = build_or_load_observation_cache(cfg, split="train", refresh=True)

    normalizer = build_observation_normalizer(cfg)
    base_env = GridEnv(
        cfg,
        mode="train",
        dataset=build_dataset(cfg, mode="train"),
        reward_fn=NormalReward(cfg),
        forecaster=build_forecaster(cfg),
        obs_builder=DefaultObservationBuilder(
            local_features=cfg.obs.local_features,
            sequence_features=cfg.obs.sequence_features,
            future_horizon=cfg.env.future_horizon,
            adjacency_type=cfg.obs.adjacency_type,
            normalizer=normalizer,
        ),
        grid_core=_FakeGridCore(cfg.env.num_agents),
    )
    fast_env = GridEnvFastLab(
        deepcopy(cfg),
        mode="train",
        dataset=build_dataset(cfg, mode="train"),
        reward_fn=NormalReward(cfg),
        obs_builder=CachedObservationBuilderFastLab(
            local_features=cfg.obs.local_features,
            sequence_features=cfg.obs.sequence_features,
            future_horizon=cfg.env.future_horizon,
            adjacency_type=cfg.obs.adjacency_type,
            normalizer=normalizer,
        ),
        grid_core=_FakeGridCore(cfg.env.num_agents),
        observation_cache_dir=cache_result.cache_dir,
        train_info_mode="minimal",
    )

    reward_metas = list(base_env.reward_fn.component_meta)
    base_history = init_episode_record(cfg.env.num_agents, cfg.env.init_soc, reward_metas)
    fast_history = init_episode_record(cfg.env.num_agents, cfg.env.init_soc, reward_metas)

    try:
        base_obs, _ = base_env.reset(episode_idx=0)
        fast_obs, _ = fast_env.reset(episode_idx=0)
        _assert_nested_close(base_obs, fast_obs)

        actions = [np.asarray([0.1], dtype=np.float32) for _ in range(cfg.env.num_agents)]
        for _ in range(2):
            next_base_obs, base_reward, base_term, base_trunc, base_info = base_env.step(actions)
            next_fast_obs, fast_reward, fast_term, fast_trunc, fast_info = fast_env.step(actions)
            _assert_nested_close(next_base_obs, next_fast_obs)
            assert np.allclose(np.asarray(base_reward, dtype=np.float32), np.asarray(fast_reward, dtype=np.float32))
            assert base_term == fast_term
            assert base_trunc == fast_trunc

            append_step_record(
                base_history,
                base_info,
                step_total=float(np.sum(np.asarray(base_reward, dtype=np.float32))),
                reward_metas=reward_metas,
            )
            append_step_record(
                fast_history,
                fast_info,
                step_total=float(np.sum(np.asarray(fast_reward, dtype=np.float32))),
                reward_metas=reward_metas,
            )

        _assert_nested_close(base_history, fast_history)
    finally:
        base_env.close()
        fast_env.close()
