from __future__ import annotations

import numpy as np
import pandas as pd

from envs.observation.normalization import ObservationNormalizer


_SAFETY_LOCAL_FIELDS = ["soc_raw", "load_raw", "pv_raw", "battery_capacity_kwh", "p_max_kw"]
_LOCAL_DIMS = {"time": 2, "calendar_time": 4, "wholesale_price": 1, "load": 1, "pv": 1, "soc": 1}
_SEQUENCE_SCOPES = {
    "wholesale_price": "shared",
    "wholesale_price_rank": "shared",
    "wholesale_price_relative": "shared",
    "wholesale_price_spread": "shared",
    "load": "per_agent",
    "pv": "per_agent",
}
_RAW_SEQUENCE_DEPENDENCIES = {
    "wholesale_price_rank": ("wholesale_price",),
    "wholesale_price_relative": ("wholesale_price",),
    "wholesale_price_spread": ("wholesale_price",),
}
_DEFAULT_PRICE_SPREAD_SCALE_EUR_PER_KWH = 0.20
_EPSILON = 1e-06
_TS_FALLBACK = pd.Timestamp("2000-01-01 00:00:00+00:00")


# 作用：把共享特征复制成每个 agent 一行的 observation 矩阵。
def _repeat_for_agents(values: np.ndarray, n_agents: int) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=np.float32).reshape(1, -1), int(n_agents), axis=0)


# 作用：按 sequence scope 统一计算序列 observation 的 shape。
def _sequence_shape(name: str, n_agents: int, sequence_length: int) -> tuple[int, ...]:
    if _SEQUENCE_SCOPES[name] == "shared":
        return (int(sequence_length),)
    return (int(n_agents), int(sequence_length))


# 作用：提取电价窗口及其最高最低价差，供派生电价特征复用。
def _price_window_range(values: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array, 0.0, 0.0, 0.0

    min_value = float(np.min(array))
    max_value = float(np.max(array))
    return array, min_value, max_value, max_value - min_value


# 作用：把 episode 内的步数位置编码成所有 agent 共享的周期时间特征。
def _time_features(cur_step: int, episode_length: int, n_agents: int) -> np.ndarray:
    phase = 2.0 * np.pi * float(cur_step) / float(episode_length)
    features = np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)
    return _repeat_for_agents(features, n_agents)


# 作用：把真实时间戳编码成日周期和年周期特征。
def _calendar_features(timestamp_value: str, n_agents: int) -> np.ndarray:
    timestamp = pd.Timestamp(timestamp_value)
    hour_of_day = float(timestamp.hour) + float(timestamp.minute) / 60.0
    day_of_year = float(timestamp.dayofyear - 1) + hour_of_day / 24.0
    phases = np.asarray(
        [
            np.sin(2.0 * np.pi * hour_of_day / 24.0),
            np.cos(2.0 * np.pi * hour_of_day / 24.0),
            np.sin(2.0 * np.pi * day_of_year / 365.25),
            np.cos(2.0 * np.pi * day_of_year / 365.25),
        ],
        dtype=np.float32,
    )
    return _repeat_for_agents(phases, n_agents)


# 作用：从当前步开始截取预测窗口，不足的尾部用 0 补齐。
def _pad_sequence(values, start: int, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    start_idx = int(start)
    sequence_length = int(length)
    tail = array[start_idx : start_idx + sequence_length]
    pad_rows = sequence_length - int(tail.shape[0])

    if pad_rows > 0:
        pad_width = ((0, pad_rows),) + tuple((0, 0) for _ in range(max(array.ndim - 1, 0)))
        tail = np.pad(tail, pad_width, mode="constant")

    result = tail if array.ndim == 1 else tail.T
    return result.astype(np.float32, copy=False)


# 作用：把未来电价窗口转换成 0 到 1 的相对排序序列。
def _rank_sequence(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    length = int(array.size)

    if length <= 1:
        return np.zeros((length,), dtype=np.float32)

    order = np.argsort(array, kind="mergesort")
    ranks = np.empty((length,), dtype=np.float32)
    ranks[order] = np.arange(length, dtype=np.float32)
    return (ranks / np.float32(length - 1)).astype(np.float32)


# 作用：把未来电价窗口按本窗口最高最低价缩放到 -1 到 1。
def _relative_price_sequence(values: np.ndarray) -> np.ndarray:
    array, min_value, _, spread = _price_window_range(values)
    if spread <= _EPSILON:
        return np.zeros_like(array, dtype=np.float32)

    return (2.0 * (array - np.float32(min_value)) / np.float32(spread) - 1.0).astype(np.float32)


# 作用：用单个比例值描述未来电价窗口的价差强度。
def _spread_price_sequence(values: np.ndarray, scale_eur_per_kwh: float) -> np.ndarray:
    array, _, _, spread = _price_window_range(values)
    scale = max(float(scale_eur_per_kwh), _EPSILON)
    value = np.float32(np.clip(spread / scale, 0.0, 1.0))
    return np.full_like(array, value, dtype=np.float32)


# 作用：拼接历史信号和当前 episode 已发生信号，供 forecaster 使用。
def _signal_history(env, signal_name: str) -> np.ndarray:
    signal = env.get_signal(signal_name)
    prefix = np.asarray(env.history_signals.get(signal_name), dtype=np.float32)

    if prefix.size == 0:
        return signal[: env.cur_step + 1].copy()

    return np.concatenate([prefix, signal[: env.cur_step + 1]], axis=0).astype(np.float32, copy=False)


# 作用：按 observation 配置把环境状态组装成 DRL 使用的 observation 字典。
class DefaultObservationBuilder:
    # 作用：保存 observation 特征合同，并提前拒绝未知特征名。
    def __init__(
        self,
        local_features: list[str],
        sequence_features: list[str],
        future_horizon: int,
        normalizer: ObservationNormalizer | None = None,
        precomputed: bool = False,
        price_spread_scale_eur_per_kwh: float = _DEFAULT_PRICE_SPREAD_SCALE_EUR_PER_KWH,
    ):
        self.local_feature_names = [str(name) for name in local_features]
        self.sequence_feature_names = [str(name) for name in sequence_features]

        unknown_local = sorted(set(self.local_feature_names) - set(_LOCAL_DIMS))
        unknown_sequence = sorted(set(self.sequence_feature_names) - set(_SEQUENCE_SCOPES))
        if unknown_local:
            raise ValueError(f"Unknown local feature(s): {unknown_local}")
        if unknown_sequence:
            raise ValueError(f"Unknown sequence feature(s): {unknown_sequence}")

        self.future_horizon = int(future_horizon)
        self.sequence_length = self.future_horizon + 1
        self.normalizer = normalizer
        self.precomputed = bool(precomputed)
        self.price_spread_scale_eur_per_kwh = float(price_spread_scale_eur_per_kwh)
        self.local_dim = sum(_LOCAL_DIMS[name] for name in self.local_feature_names)

    # 作用：给出每个 observation 分量的固定 shape 合同。
    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        n_agents = int(n_agents)
        schema = {
            "local": (n_agents, self.local_dim),
            "safety_local": (n_agents, len(_SAFETY_LOCAL_FIELDS)),
        }

        for name in self.sequence_feature_names:
            schema[f"{name}_seq"] = _sequence_shape(name, n_agents, self.sequence_length)

        return schema

    # 作用：给训练代码和调试代码提供 observation 分量的语义布局。
    def get_layout(self, n_agents: int) -> dict[str, dict]:
        layout = {
            "local": {
                "group": "local",
                "scope": "per_agent",
                "dim": int(self.local_dim),
                "fields": list(self.local_feature_names),
            },
            "safety_local": {
                "group": "projector",
                "scope": "per_agent",
                "dim": len(_SAFETY_LOCAL_FIELDS),
                "fields": list(_SAFETY_LOCAL_FIELDS),
            },
        }
        schema = self.get_schema(n_agents)

        for name in self.sequence_feature_names:
            layout[f"{name}_seq"] = {
                "feature_name": name,
                "group": "sequence",
                "scope": _SEQUENCE_SCOPES[name],
                "dim": 1,
                "shape": schema[f"{name}_seq"],
            }

        return layout

    # 作用：按 schema 生成全零 observation，占位或 reset 前调试时使用。
    def zeros(self, n_agents: int) -> dict[str, np.ndarray]:
        return {key: np.zeros(shape, dtype=np.float32) for key, shape in self.get_schema(n_agents).items()}

    # 作用：raw observation 需要补齐派生序列依赖的原始序列。
    def _raw_sequence_feature_names(self) -> list[str]:
        names = list(self.sequence_feature_names)

        for feature_name in self.sequence_feature_names:
            for dependency in _RAW_SEQUENCE_DEPENDENCIES.get(feature_name, ()):
                if dependency not in names:
                    names.insert(0, dependency)

        return names

    # 作用：构造未归一化 observation，供统计拟合和教学调试使用。
    def build_raw(self, env) -> dict[str, np.ndarray]:
        return self._build(env, normalize=False, sequence_feature_names=self._raw_sequence_feature_names())

    # 作用：构造训练和推理主线使用的 observation。
    def build(self, env) -> dict[str, np.ndarray]:
        return self._build(env, normalize=True, sequence_feature_names=self.sequence_feature_names)

    # 作用：在需要时把单个特征交给 ObservationNormalizer 转换。
    def _normalize(self, feature_name: str, values: np.ndarray, *, normalize: bool, sequence: bool) -> np.ndarray:
        if not normalize or self.normalizer is None:
            return np.asarray(values, dtype=np.float32)

        transform = self.normalizer.transform_sequence if sequence else self.normalizer.transform_local
        return np.asarray(transform(feature_name, values), dtype=np.float32)

    # 作用：提取当前步的日历时间特征，支持实时 timestamp 和预计算缓存两种来源。
    def _calendar_feature(self, env) -> np.ndarray:
        if self.precomputed:
            if "calendar_time" not in env._episode_precomputed:
                raise KeyError("Precomputed observation data does not contain 'calendar_time'.")
            return np.asarray(env._episode_precomputed["calendar_time"][env.cur_step], dtype=np.float32)

        timestamps = list(dict(getattr(env, "episode_meta", {})).get("timestamps") or [])
        cur_step = int(env.cur_step)
        if 0 <= cur_step < len(timestamps):
            timestamp = str(timestamps[cur_step])
        else:
            timestamp = str(_TS_FALLBACK + pd.Timedelta(minutes=15 * cur_step))
        return _calendar_features(timestamp, env.n)

    # 作用：从环境当前状态提取单步 per-agent 本地特征。
    def _local_feature(self, env, name: str) -> np.ndarray:
        if name == "time":
            return _time_features(env.cur_step, env.episode_length, env.n)

        if name == "calendar_time":
            return self._calendar_feature(env)

        if name == "soc":
            return np.asarray(env.soc, dtype=np.float32).reshape(-1, 1)

        if name == "wholesale_price":
            return np.full((env.n, 1), float(env.get_signal_step(name)), dtype=np.float32)

        return np.asarray(env.get_signal_step(name), dtype=np.float32).reshape(-1, 1)

    # 作用：读取预计算序列并校验当前 horizon 和 agent 数量合同。
    def _precomputed_sequence_feature(self, env, name: str) -> np.ndarray:
        cache_key = f"{name}_seq"
        if cache_key not in env._episode_precomputed:
            raise KeyError(f"Precomputed observation data does not contain '{cache_key}'.")

        values = np.asarray(env._episode_precomputed[cache_key][env.cur_step], dtype=np.float32)
        expected_shape = _sequence_shape(name, env.n, self.sequence_length)

        if tuple(values.shape) != expected_shape:
            raise ValueError(
                "Precomputed observation horizon contract mismatch at "
                "DefaultObservationBuilder._sequence_feature(...): old "
                f"shared-data object '{getattr(env, '_precomputed_data_dir', None)}' "
                f"returned '{cache_key}' with shape {tuple(values.shape)}, but "
                f"current cfg.env.future_horizon={self.future_horizon} and "
                f"env.n={int(env.n)} require shape {expected_shape}. Expected "
                "shared-data generated with the current config. Re-run "
                "notebooks/forecast/forecast_lstm.ipynb, then rerun the "
                "consuming notebook or entrypoint."
            )

        return values

    # 作用：调用 forecaster 预测当前信号的未来 observation 窗口。
    def _forecast_sequence_feature(self, env, name: str) -> np.ndarray:
        history = _signal_history(env, name)
        timestamps = list(dict(getattr(env, "episode_meta", {})).get("timestamps") or [])
        history_timestamps = [*env.history_timestamps, *[str(timestamp) for timestamp in timestamps[: max(0, int(env.cur_step) + 1)]]]
        prediction = env.forecaster.predict(history, self.sequence_length, signal_name=name, history_timestamps=history_timestamps)
        return np.asarray(prediction, dtype=np.float32)

    # 作用：从环境或预计算缓存提取未来窗口序列特征。
    def _sequence_feature(self, env, name: str, cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if cache is not None and name in cache:
            return cache[name]

        if self.precomputed:
            result = self._precomputed_sequence_feature(env, name)

        elif name == "wholesale_price_rank":
            result = _rank_sequence(self._sequence_feature(env, "wholesale_price", cache))

        elif name == "wholesale_price_relative":
            result = _relative_price_sequence(self._sequence_feature(env, "wholesale_price", cache))

        elif name == "wholesale_price_spread":
            result = _spread_price_sequence(self._sequence_feature(env, "wholesale_price", cache), self.price_spread_scale_eur_per_kwh)

        elif getattr(env, "forecaster", None) is None:
            result = _pad_sequence(env.get_signal(name), env.cur_step, self.sequence_length)

        else:
            result = self._forecast_sequence_feature(env, name)

        if cache is not None:
            cache[name] = result

        return result

    # 作用：统一组装 local、safety 和 sequence observation 分量。
    def _build(self, env, *, normalize: bool, sequence_feature_names: list[str]) -> dict[str, np.ndarray]:
        local_parts = [
            self._normalize(name, self._local_feature(env, name), normalize=normalize, sequence=False)
            for name in self.local_feature_names
        ]

        if local_parts:
            local = np.concatenate(local_parts, axis=1).astype(np.float32)
        else:
            local = np.zeros((env.n, 0), dtype=np.float32)

        obs = {
            "local": local,
            "safety_local": env._current_safety_local().astype(np.float32),
        }

        sequence_cache: dict[str, np.ndarray] = {}
        for name in sequence_feature_names:
            obs[f"{name}_seq"] = self._normalize(name, self._sequence_feature(env, name, sequence_cache), normalize=normalize, sequence=True)

        return obs
