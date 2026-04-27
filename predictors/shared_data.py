from __future__ import annotations  # 允许类型注解延迟解析，避免运行期提前求值。

import hashlib, json, shutil, uuid  # 导入哈希、JSON、目录替换和临时目录命名工具。
from datetime import datetime, timezone  # 导入带时区的创建时间工具。
from pathlib import Path  # 导入跨平台路径对象。
from typing import Any  # 导入通用类型标注。

import numpy as np, pandas as pd, torch  # 导入数组、时间序列和 Torch 运行时依赖。
from numpy.lib.format import open_memmap  # 导入 NumPy memmap 写入接口。
from tqdm.auto import tqdm  # 导入进度条，用于长时间 shared-data 生成。

from data.loaders.registry import (  # 导入数据集窗口、构建和 episode 限制解析入口。
    _resolve_split_dates,  # 解析 train/validation/test 的显式日期边界。
    build_dataset,  # 按 split 构建 prosumer 数据集。
    resolve_dataset_window_spec,  # 解析 split 对应的窗口策略和 episode 长度。
    resolve_test_episode_limit,  # 解析测试 episode 上限。
    resolve_train_episode_limit,  # 解析训练 episode 上限。
)  # 结束数据加载器导入列表。
from predictors.lstm_forecaster import (  # 导入 LSTM forecast artifact 读取能力。
    LSTMForecaster,  # 用于标准化 artifact bundle。
    load_lstm_forecaster_artifacts,  # 用于读取模型元数据以计算签名。
)  # 结束 LSTM forecaster 导入列表。
from predictors.registry import build_forecaster  # 导入运行时 forecaster 构建入口。
from predictors.time_features import (  # 导入时间特征构造所需工具。
    DEFAULT_LOCAL_TIMEZONE,  # 默认本地时区，用于日历特征对齐。
    coerce_timestamp_index,  # 将输入时间戳转成 Pandas DatetimeIndex。
)  # 结束时间特征导入列表。
from predictors.training import ensure_lstm_artifacts  # 导入 LSTM artifact 准备入口。
from scripts.utils.price_protocol import (  # 导入价格协议字段和校验。
    PRICE_PROTOCOL_VERSION,  # 当前价格协议版本。
    WHOLESALE_PRICE_SEQ_FIELD,  # shared-data 中批发电价序列字段名。
    WHOLESALE_PRICE_SIGNAL,  # 原始时间线中批发电价信号名。
    assert_no_legacy_price_schema,  # 校验旧价格 schema 不再被接受。
)  # 结束价格协议导入列表。
from scripts.utils.project_paths import get_shared_data_root  # 导入 shared-data 根目录解析入口。

_FLOAT_PRECISION, _SCHEMA_VERSION = 6, 9  # 定义签名浮点精度和 shared-data schema 版本。
SHARED_DATA_SCHEMA_VERSION = _SCHEMA_VERSION  # 暴露当前 shared-data schema 版本给外部模块。
PRICE_OBSERVATION_CONTRACT = "oracle_window_relative_price_v1"  # 定义价格观测合同名。
WHOLESALE_PRICE_RANK_SEQ_FIELD = "wholesale_price_rank_seq"  # 定义窗口内电价排序特征字段。
WHOLESALE_PRICE_RELATIVE_SEQ_FIELD = "wholesale_price_relative_seq"  # 定义窗口内相对电价特征字段。
WHOLESALE_PRICE_SPREAD_SEQ_FIELD = "wholesale_price_spread_seq"  # 定义窗口内电价价差特征字段。
SHARED_DATA_CACHE_LAYOUT = "timeline_cache_v1"  # 定义 shared-data 缓存布局版本。
_SHARED_DATA_FORECAST_ROW_CHUNK_SIZE = 2048  # 定义 forecast 缓存按行写入的 chunk 大小。
_SHARED_DATA_FORECAST_BATCH_SIZE = 1024  # 定义向量化 LSTM 预测的 batch 大小。
_ARTIFACT_META_KEYS = (  # 定义会进入 shared-data 签名的 artifact 元数据字段。
    "signal_name",  # 记录该 artifact 预测的信号名。
    "future_horizon",  # 记录预测 horizon。
    "seq_len",  # 记录 LSTM 输入序列长度。
    "hidden_size",  # 记录 LSTM hidden size。
    "num_layers",  # 记录 LSTM 层数。
    "dropout",  # 记录 dropout 参数。
    "input_size",  # 记录输入维度。
    "time_feature_mode",  # 记录时间特征模式。
    "model_mode",  # 记录模型模式。
    "normalization_mode",  # 记录归一化模式。
    "postprocess_mode",  # 记录后处理模式。
    "baseline_mode",  # 记录 baseline 模式。
    "blend_weight",  # 记录 blend 权重。
    "component",  # 记录组件名。
    "agent_index",  # 记录 agent 下标。
    "agent_profile",  # 记录 agent profile。
)  # 结束 artifact 元数据字段列表。


def _json_default(value: Any):  # 为 json.dumps 提供项目内对象的默认序列化方式。
    if isinstance(value, (Path, torch.device)):  # 如果是路径或 Torch device。
        return str(value)  # 转成字符串写入 JSON。
    if hasattr(value, "item"):  # 如果是 NumPy/Torch 标量样式对象。
        try:  # 尝试走标量提取路径。
            return value.item()  # 返回 Python 原生标量。
        except Exception:  # 如果对象的 item() 不能正常工作。
            pass  # 保持原行为，继续落到 TypeError。
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")  # 对未知对象显式失败。


def _normalize_for_signature(value: Any) -> Any:  # 将任意配置值规范化成稳定签名输入。
    if isinstance(value, dict):  # 如果输入是字典。
        return {  # 返回按 key 字符串排序后的规范化字典。
            str(k): _normalize_for_signature(v)  # 递归规范化每个 value，并把 key 转成字符串。
            for (k, v) in sorted(dict(value).items(), key=lambda item: str(item[0]))  # 按 key 的字符串形式稳定排序。
        }  # 结束字典规范化结果。
    if isinstance(value, (list, tuple)):  # 如果输入是列表或元组。
        return [_normalize_for_signature(v) for v in value]  # 递归规范化序列中的每个元素。
    if isinstance(value, (np.floating, float)):  # 如果输入是浮点数。
        return f"{float(round(float(value), _FLOAT_PRECISION)):.{_FLOAT_PRECISION}f}"  # 固定精度为字符串以避免微小误差影响签名。
    if isinstance(value, (np.integer, int)):  # 如果输入是整数。
        return int(value)  # 转成 Python int 保持 JSON 稳定。
    return str(value) if isinstance(value, Path) else value  # Path 转字符串，其它类型原样返回。


def _signature_hash(payload: dict[str, object]) -> str:  # 计算 shared-data 配置签名短哈希。
    normalized_payload = _normalize_for_signature(payload)  # 先规范化 payload，消除顺序和类型噪声。
    payload_text = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=True)  # 以稳定 key 顺序转成 ASCII JSON。
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]  # 返回前 16 位 SHA-256 作为目录签名。


def _file_sha256(path: str | Path) -> str:  # 计算单个 artifact 文件的 SHA-256。
    digest = hashlib.sha256()  # 创建增量哈希对象。
    with Path(path).open("rb") as handle:  # 以二进制方式打开文件。
        for chunk in iter(lambda: handle.read(1048576), b""):  # 按 1 MiB 块读取直到 EOF。
            digest.update(chunk)  # 将当前块写入哈希。
    return digest.hexdigest()  # 返回完整十六进制哈希。


def _artifact_meta(meta: dict[str, object]) -> dict[str, object]:  # 提取进入 shared-data 签名的 artifact 元数据。
    normalized: dict[str, object] = {}  # 创建规范化元数据容器。
    for key in _ARTIFACT_META_KEYS:  # 按固定字段顺序遍历。
        if key == "future_horizon":  # future_horizon 兼容旧 meta 中的 pred_len 名称。
            raw_value = meta.get("future_horizon", meta.get("pred_len", 0))  # 优先读取 future_horizon，缺失时读取 pred_len。
            normalized[key] = int(0 if raw_value is None else raw_value)  # 空值归零并转成 int。
            continue  # 当前字段处理完毕。
        if key in {"seq_len", "hidden_size", "num_layers", "input_size"}:  # 这些字段必须按整数参与签名。
            raw_value = meta.get(key, 0)  # 读取字段，缺失时默认 0。
            normalized[key] = int(0 if raw_value is None else raw_value)  # 空值归零并转成 int。
            continue  # 当前字段处理完毕。
        if key == "agent_index":  # agent_index 允许不存在。
            raw_value = meta.get(key)  # 读取 agent_index 原始值。
            normalized[key] = None if raw_value is None else int(raw_value)  # None 保持 None，否则转成 int。
            continue  # 当前字段处理完毕。
        if key == "dropout":  # dropout 作为浮点值参与签名。
            raw_value = meta.get(key, 0.0)  # 读取 dropout，缺失时默认 0。
            normalized[key] = float(0.0 if raw_value is None else raw_value)  # 空值归零并转成 float。
            continue  # 当前字段处理完毕。
        normalized[key] = meta.get(key)  # 其它字段按原值写入。
    return _normalize_for_signature(normalized)  # 最后统一走签名规范化。


def _artifact_fingerprint(forecast_ready: dict[str, object]) -> dict[str, object]:  # 为 LSTM artifact 生成文件级指纹。
    artifacts = dict(forecast_ready.get("artifacts") or {})  # 取出 forecast_ready 中的 artifact 映射。
    if not artifacts:  # 如果没有可用 artifact。
        raise FileNotFoundError("No managed LSTM artifacts are available for shared MADRL data generation.")  # 显式拒绝生成。
    return {  # 返回按 signal_name 排序后的 artifact 指纹字典。
        str(signal_name): [  # 每个 signal 对应一个或多个模型 artifact。
            {  # 为单个模型 artifact 记录文件哈希和规范化 meta。
                "model_sha256": _file_sha256(model_path),  # 记录模型权重文件哈希。
                "meta_sha256": _file_sha256(meta_path),  # 记录 meta 文件哈希。
                "scaler_sha256": _file_sha256(scaler_path),  # 记录 scaler 文件哈希。
                "meta": _artifact_meta(  # 记录参与签名的规范化 meta。
                    load_lstm_forecaster_artifacts(model_path, meta_path, scaler_path)[0]  # 读取 artifact meta。
                ),  # 结束 meta 规范化。
            }  # 结束单个模型 artifact 指纹。
            for (model_path, meta_path, scaler_path) in LSTMForecaster._normalize_artifact_bundle(bundle)  # 保持 artifact bundle 的原始规范化顺序。
        ]  # 结束当前 signal 的 artifact 指纹列表。
        for (signal_name, bundle) in sorted(artifacts.items())  # 按 signal 名排序，保证签名稳定。
    }  # 结束 artifact 指纹字典。


def _shared_data_signature_payload(  # 构建 shared-data 目录签名的完整 payload。
    cfg,  # 当前实验配置对象。
    *,  # 后续参数必须按关键字传入。
    artifact_fingerprint: dict[str, object],  # forecast artifact 文件和 meta 指纹。
) -> dict[str, object]:  # 返回可 JSON 序列化的签名 payload。
    return {  # 返回影响 shared-data 内容的所有控制项。
        "schema_version": _SCHEMA_VERSION,  # schema 版本变化必须生成新缓存。
        "price_observation_contract": PRICE_OBSERVATION_CONTRACT,  # 价格观测合同变化必须生成新缓存。
        "num_agents": int(cfg.env.num_agents),  # agent 数量影响数组形状。
        "episode_limit": int(cfg.env.episode_limit),  # episode 长度影响训练窗口。
        "train_episode_limit": int(resolve_train_episode_limit(cfg)),  # 训练 episode 上限影响 manifest。
        "test_episode_limit": int(resolve_test_episode_limit(cfg)),  # 测试 episode 上限影响 manifest。
        "train_window_days": int(getattr(cfg.env, "train_window_days", 1)),  # 训练窗口天数影响 dataset 切片。
        "window_stride_days": int(getattr(cfg.env, "window_stride_days", 1)),  # 窗口步长影响 episode 切片。
        "future_horizon": int(cfg.env.future_horizon),  # 预测 horizon 影响序列长度。
        "agent_profiles": list(cfg.data.agent_profiles),  # agent profile 影响负荷/PV 数据。
        "agent_bus_ids": list(cfg.grid.agent_bus_ids),  # bus id 影响 grid-agent 对齐。
        "train_year": int(cfg.data.train_year),  # 训练年份影响数据源。
        "test_year": int(cfg.data.test_year),  # 测试年份影响数据源。
        "train_start_date": cfg.data.train_start_date,  # 显式训练开始日期影响训练 split。
        "train_end_date": cfg.data.train_end_date,  # 显式训练结束日期影响训练 split。
        "test_manifest_scope": "full_year",  # 测试 manifest 固定为全年范围。
        "load_components": list(cfg.data.load_components),  # 负荷组件影响观测序列。
        "pv_reference": str(cfg.data.pv_reference),  # PV reference 影响 PV 序列。
        "pv_capacity_kw": list(cfg.data.pv_capacity_kw),  # PV 容量影响 PV 序列。
        "load_scale": list(cfg.data.load_scale),  # 负荷缩放影响负荷序列。
        "pv_scale": list(cfg.data.pv_scale),  # PV 缩放影响 PV 序列。
        "wholesale_price_spread_scale_eur_per_kwh": float(  # 价差归一化尺度影响价格特征。
            getattr(cfg.obs, "wholesale_price_spread_scale_eur_per_kwh", 0.20)  # 缺省保持原 0.20。
        ),  # 结束价差尺度字段。
        "price_protocol_version": int(PRICE_PROTOCOL_VERSION),  # 价格协议版本影响价格字段语义。
        "artifacts": artifact_fingerprint,  # LSTM artifact 指纹影响预测序列。
    }  # 结束签名 payload。


def _resolve_shared_split_controls(cfg, split: str) -> dict[str, object]:  # 解析写入 manifest 的 split 控制项。
    if str(split) == "test":  # test split 固定生成全年 manifest。
        selected_year, start_date, end_date = (  # test 不使用运行时选择窗口。
            int(cfg.data.test_year),  # test 年份来自配置。
            None,  # test manifest 不固定开始日期。
            None,  # test manifest 不固定结束日期。
        )  # 结束 test split 控制项。
    else:  # 非 test split 使用数据加载器的标准 split 日期解析。
        selected_year, start_date, end_date = _resolve_split_dates(cfg, split)  # 解析 train 等 split 的日期控制项。
    window_spec = resolve_dataset_window_spec(cfg, split)  # 解析该 split 的窗口策略。
    return {  # 返回 manifest 可记录的 split 控制项。
        "split": str(split),  # 记录 split 名称。
        "year": int(selected_year),  # 记录 split 年份。
        "start_date": start_date,  # 记录显式开始日期。
        "end_date": end_date,  # 记录显式结束日期。
        "window_strategy": str(window_spec["window_strategy"]),  # 记录窗口策略。
        "window_days": int(window_spec["window_days"]),  # 记录窗口天数。
        "window_stride_days": int(window_spec["window_stride_days"]),  # 记录窗口步长天数。
        "base_episode_length": int(window_spec["base_episode_length"]),  # 记录基础 episode 长度。
        "episode_length": int(window_spec["episode_length"]),  # 记录最终 episode 长度。
        "manifest_scope": "full_year" if str(split) == "test" else "configured_split",  # test 为全年，其它 split 为配置范围。
    }  # 结束 split 控制项。


def _episode_manifest_entry(dataset, episode_idx: int, episode: dict[str, object]) -> dict[str, object]:  # 构建单个 episode 的 manifest 记录。
    history_start_idx, active_start_idx, active_end_idx, next_active_idx = list(  # 从数据集内部切片表取出索引边界。
        getattr(dataset, "_episode_slices", [])  # 读取 episode 切片元数据。
    )[int(episode_idx)]  # 按 episode_idx 选出当前 episode 切片。
    timestamps = [  # 将 episode meta 中的时间戳转成字符串列表。
        str(value)  # 保留时间戳的字符串表示。
        for value in list(dict(episode.get("meta", {})).get("timestamps") or [])  # 读取 meta.timestamps，缺失时使用空列表。
    ]  # 结束时间戳列表。
    first, last = (None, None) if not timestamps else (str(timestamps[0]), str(timestamps[-1]))  # 记录首末时间戳。
    return {  # 返回当前 episode 的 manifest 条目。
        "episode_idx": int(episode_idx),  # 记录 episode 下标。
        "first_timestamp": first,  # 记录首个时间戳。
        "last_timestamp": last,  # 记录最后一个时间戳。
        "first_local_date": None if first is None else str(pd.Timestamp(first).date()),  # 记录首个本地日期。
        "last_local_date": None if last is None else str(pd.Timestamp(last).date()),  # 记录最后一个本地日期。
        "history_start_idx": int(history_start_idx),  # 记录历史窗口起点。
        "active_start_idx": int(active_start_idx),  # 记录 active 窗口起点。
        "active_end_idx": int(active_end_idx),  # 记录 active 窗口终点。
        "next_active_idx": None if next_active_idx is None else int(next_active_idx),  # 记录 bootstrap 下一行索引。
    }  # 结束 episode manifest 条目。


def _build_calendar_time_matrix(  # 构建每个时间步、每个 agent 共享的日历特征。
    timestamps: list[str | pd.Timestamp] | np.ndarray | tuple[str | pd.Timestamp, ...],  # 输入时间戳序列。
    n_agents: int,  # agent 数量。
) -> np.ndarray:  # 返回形状为 (时间步, agent, 4) 的 float32 日历特征。
    index = coerce_timestamp_index(timestamps)  # 将输入统一成 DatetimeIndex。
    if getattr(index, "tz", None) is not None:  # 如果时间戳已经带时区。
        index = index.tz_convert(DEFAULT_LOCAL_TIMEZONE)  # 转成默认本地时区。
    hour = index.hour.to_numpy(dtype=np.float32) + index.minute.to_numpy(dtype=np.float32) / np.float32(60.0)  # 计算小时小数。
    year = index.dayofyear.to_numpy(dtype=np.float32) - np.float32(1.0) + hour / np.float32(24.0)  # 计算年内日小数。
    per_step = np.stack(  # 构建每个时间步的 sin/cos 周期特征。
        [  # 开始日历特征列表。
            np.sin(np.float32(2.0 * np.pi) * hour / np.float32(24.0)),  # 日内正弦特征。
            np.cos(np.float32(2.0 * np.pi) * hour / np.float32(24.0)),  # 日内余弦特征。
            np.sin(np.float32(2.0 * np.pi) * year / np.float32(365.25)),  # 年内正弦特征。
            np.cos(np.float32(2.0 * np.pi) * year / np.float32(365.25)),  # 年内余弦特征。
        ],  # 结束日历特征列表。
        axis=1,  # 每行对应一个时间步。
    ).astype(np.float32, copy=False)  # 保证输出为 float32。
    return np.broadcast_to(per_step[:, None, :], (len(index), int(n_agents), 4)).astype(np.float32, copy=False)  # 广播到所有 agent。


def _compute_future_mean_price(  # 计算当前和下一步的未来均价辅助特征。
    wholesale_price: np.ndarray,  # 输入批发电价时间线。
    future_horizon: int,  # 未来窗口长度。
) -> tuple[np.ndarray, np.ndarray]:  # 返回当前步和下一步的未来均价。
    values = np.asarray(wholesale_price, dtype=np.float32).reshape(-1)  # 将电价转成一维 float32。
    if values.size == 0:  # 如果没有任何电价样本。
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)  # 返回两个空数组。
    horizon = int(max(future_horizon, 1))  # horizon 至少为 1。
    idx = np.arange(values.size, dtype=np.int64)  # 构建每个时间步的下标。
    csum = np.concatenate([[0.0], np.cumsum(values.astype(np.float64), dtype=np.float64)])  # 构建 float64 前缀和。
    start, end = idx + 1, np.minimum(idx + 1 + horizon, values.size)  # 当前步未来窗口为 t+1 到 t+horizon。
    next_idx, start_next = np.minimum(idx + 1, values.size - 1), np.minimum(idx + 2, values.size)  # 下一步窗口从 t+2 开始。
    end_next = np.minimum(next_idx + 1 + horizon, values.size)  # 计算下一步未来窗口终点。
    mu_t = np.where(  # 计算当前步未来均价。
        start < values.size,  # 只有存在未来样本时使用窗口均值。
        (csum[end] - csum[start]) / np.maximum(end - start, 1),  # 使用前缀和求窗口均值。
        values[np.minimum(idx, values.size - 1)],  # 没有未来样本时回退到当前/最后电价。
    )  # 结束当前步均价计算。
    mu_next = np.where(  # 计算下一步未来均价。
        start_next < values.size,  # 只有存在下一步未来样本时使用窗口均值。
        (csum[end_next] - csum[start_next]) / np.maximum(end_next - start_next, 1),  # 使用前缀和求下一步窗口均值。
        values[next_idx],  # 没有未来样本时使用下一步/最后电价。
    )  # 结束下一步均价计算。
    return mu_t.astype(np.float32), mu_next.astype(np.float32)  # 返回 float32 结果。


def _rank_sequence_matrix(values: np.ndarray) -> np.ndarray:  # 将价格序列转换为窗口内排序特征。
    array = np.asarray(values, dtype=np.float32)  # 输入转为 float32 数组。
    if array.ndim == 1:  # 如果只有单条序列。
        array = array.reshape(1, -1)  # 转成二维矩阵。
    if array.ndim != 2:  # 排序特征只接受一维或二维输入。
        raise ValueError(f"Expected 1D or 2D price sequence values for rank feature, got shape {array.shape}.")  # 输入维度不合法时失败。
    length = int(array.shape[-1])  # 读取序列长度。
    if length <= 1:  # 长度为 0 或 1 时没有排序信息。
        return np.zeros_like(array, dtype=np.float32)  # 返回全零特征。
    order = np.argsort(array, axis=-1, kind="mergesort")  # 使用稳定排序得到每行从低到高的位置。
    ranks = np.empty_like(order, dtype=np.float32)  # 创建 rank 输出矩阵。
    row_index = np.arange(array.shape[0])[:, None]  # 构建二维行索引。
    ranks[row_index, order] = np.arange(length, dtype=np.float32)  # 将排序位置反写成 rank。
    return (ranks / np.float32(max(length - 1, 1))).astype(np.float32)  # 归一化到 [0, 1]。


def _relative_price_sequence_matrix(values: np.ndarray) -> np.ndarray:  # 将价格序列转换为窗口内相对价格特征。
    array = np.asarray(values, dtype=np.float32)  # 输入转为 float32 数组。
    if array.ndim == 1:  # 如果只有单条序列。
        array = array.reshape(1, -1)  # 转成二维矩阵。
    if array.ndim != 2:  # 相对价格特征只接受一维或二维输入。
        raise ValueError(f"Expected 1D or 2D price sequence values for relative feature, got shape {array.shape}.")  # 输入维度不合法时失败。
    if int(array.shape[-1]) <= 0:  # 如果序列长度为空。
        return np.zeros_like(array, dtype=np.float32)  # 返回全零特征。
    minimum = np.min(array, axis=-1, keepdims=True)  # 计算每条序列的最小价格。
    spread = np.max(array, axis=-1, keepdims=True) - minimum  # 计算每条序列的价格范围。
    scaled = 2.0 * (array - minimum) / np.maximum(spread, np.float32(1e-06)) - 1.0  # 将价格线性映射到 [-1, 1]。
    return np.where(spread > np.float32(1e-06), scaled, np.zeros_like(array, dtype=np.float32)).astype(np.float32)  # 无价差时返回零。


def _spread_price_sequence_matrix(  # 将价格序列转换为窗口价差强度特征。
    values: np.ndarray,  # 输入价格序列。
    scale_eur_per_kwh: float,  # 价差归一化尺度。
) -> np.ndarray:  # 返回与输入矩阵同形状的价差特征。
    array = np.asarray(values, dtype=np.float32)  # 输入转为 float32 数组。
    if array.ndim == 1:  # 如果只有单条序列。
        array = array.reshape(1, -1)  # 转成二维矩阵。
    if array.ndim != 2:  # 价差特征只接受一维或二维输入。
        raise ValueError(f"Expected 1D or 2D price sequence values for spread feature, got shape {array.shape}.")  # 输入维度不合法时失败。
    if int(array.shape[-1]) <= 0:  # 如果序列长度为空。
        return np.zeros_like(array, dtype=np.float32)  # 返回全零特征。
    spread = np.max(array, axis=-1, keepdims=True) - np.min(array, axis=-1, keepdims=True)  # 计算每条序列的价格范围。
    scale = max(float(scale_eur_per_kwh), 1e-06)  # 防止除以 0 或极小尺度。
    value = np.clip(spread / np.float32(scale), 0.0, 1.0).astype(np.float32)  # 将价差缩放并截断到 [0, 1]。
    return np.broadcast_to(value, array.shape).astype(np.float32, copy=False)  # 广播到每个 horizon 位置。


def _merge_history_with_episode_signals(  # 将历史信号与 episode 信号拼接。
    signals: dict[str, np.ndarray],  # 当前 episode 的信号。
    history_signals: dict[str, np.ndarray],  # episode 前置历史信号。
) -> tuple[dict[str, np.ndarray], int]:  # 返回拼接后的信号和历史前缀长度。
    history_arrays = {  # 先把所有历史信号转成 float32 数组。
        name: np.asarray(history_signals.get(name), dtype=np.float32)  # 缺失值保持 NumPy 原行为。
        for name in signals  # 只处理当前信号集合中的 key。
    }  # 结束历史数组映射。
    merged = {  # 为每个信号构建历史+当前的合并数组。
        name: np.concatenate(  # 如果存在历史，则拼接历史和当前信号。
            [history_arrays[name], np.asarray(signal, dtype=np.float32)],  # 拼接顺序保持历史在前。
            axis=0,  # 沿时间维拼接。
        ).astype(np.float32, copy=False)  # 输出保持 float32。
        if history_arrays[name].size  # 只有历史数组非空才拼接。
        else np.asarray(signal, dtype=np.float32).copy()  # 没有历史时复制当前信号。
        for (name, signal) in signals.items()  # 遍历当前 episode 信号。
    }  # 结束合并信号映射。
    prefix = max((int(values.shape[0]) for values in history_arrays.values()), default=0)  # 计算历史前缀长度。
    return merged, prefix  # 返回合并信号和前缀长度。


def _open_split_arrays(  # 为某个 split 创建全部 shared-data memmap 文件。
    split_dir: Path,  # split 输出目录。
    *,  # 后续参数必须按关键字传入。
    num_timestamps: int,  # 时间线总行数。
    n_agents: int,  # agent 数量。
    sequence_length: int,  # 预测序列长度。
) -> tuple[dict[str, str], dict[str, np.ndarray]]:  # 返回 manifest 文件名映射和 memmap 数组映射。
    files = {  # 定义 logical field 到 .npy 文件名的映射。
        "calendar_time": "calendar_time.npy",  # 日历时间特征文件。
        WHOLESALE_PRICE_SEQ_FIELD: f"{WHOLESALE_PRICE_SEQ_FIELD}.npy",  # 批发电价序列文件。
        WHOLESALE_PRICE_RANK_SEQ_FIELD: f"{WHOLESALE_PRICE_RANK_SEQ_FIELD}.npy",  # 批发电价 rank 序列文件。
        WHOLESALE_PRICE_RELATIVE_SEQ_FIELD: f"{WHOLESALE_PRICE_RELATIVE_SEQ_FIELD}.npy",  # 相对电价序列文件。
        WHOLESALE_PRICE_SPREAD_SEQ_FIELD: f"{WHOLESALE_PRICE_SPREAD_SEQ_FIELD}.npy",  # 价差序列文件。
        "load_seq": "load_seq.npy",  # 负荷预测序列文件。
        "pv_seq": "pv_seq.npy",  # PV 预测序列文件。
        "mu_t": "mu_t.npy",  # 当前步未来均价文件。
        "mu_next": "mu_next.npy",  # 下一步未来均价文件。
    }  # 结束文件映射。
    shapes = {  # 定义每个 memmap 数组的形状。
        "calendar_time": (num_timestamps, n_agents, 4),  # 日历特征包含 4 个周期分量。
        WHOLESALE_PRICE_SEQ_FIELD: (num_timestamps, sequence_length),  # 电价序列为时间步 x horizon。
        WHOLESALE_PRICE_RANK_SEQ_FIELD: (num_timestamps, sequence_length),  # rank 序列与电价序列同形。
        WHOLESALE_PRICE_RELATIVE_SEQ_FIELD: (num_timestamps, sequence_length),  # 相对价格序列与电价序列同形。
        WHOLESALE_PRICE_SPREAD_SEQ_FIELD: (num_timestamps, sequence_length),  # 价差序列与电价序列同形。
        "load_seq": (num_timestamps, n_agents, sequence_length),  # 负荷序列按 agent 维展开。
        "pv_seq": (num_timestamps, n_agents, sequence_length),  # PV 序列按 agent 维展开。
        "mu_t": (num_timestamps,),  # 当前步未来均价是一维时间线。
        "mu_next": (num_timestamps,),  # 下一步未来均价是一维时间线。
    }  # 结束形状映射。
    arrays = {  # 创建每个字段对应的 memmap 数组。
        name: open_memmap(split_dir / file_name, mode="w+", dtype=np.float32, shape=shapes[name])  # 以写入模式创建 float32 .npy memmap。
        for (name, file_name) in files.items()  # 遍历文件映射。
    }  # 结束 memmap 数组映射。
    return files, arrays  # 返回 manifest 文件映射和打开的数组。


def _close_split_arrays(arrays: dict[str, np.ndarray]) -> None:  # 关闭 split 生成期间打开的 memmap 数组。
    for array in arrays.values():  # 遍历所有 memmap 数组。
        array.flush()  # 先将内存中的改动刷到磁盘。
        mmap = getattr(array, "_mmap", None)  # 取出底层 mmap 对象。
        if mmap is not None:  # 如果底层 mmap 存在。
            mmap.close()  # 关闭底层 mmap 文件句柄。


def _release_shared_data_cuda_cache(device: torch.device | str) -> None:  # 在 chunk 之间释放 CUDA cache。
    device = torch.device(device)  # 将字符串或 device 统一成 torch.device。
    if device.type != "cuda" or not torch.cuda.is_available():  # 只有 CUDA 可用且当前设备为 CUDA 才处理。
        return  # CPU 或无 CUDA 时不做任何事。
    torch.cuda.synchronize(device)  # 等待当前 CUDA 设备上的操作完成。
    torch.cuda.empty_cache()  # 清理缓存以降低长时间生成过程的显存压力。


def _forecaster_context_steps(forecaster, signal_name: str) -> int:  # 读取某个信号预测所需的最大历史步数。
    runtimes = list(getattr(forecaster, "signal_runtimes", {}).get(signal_name) or [])  # 读取该 signal 的所有 runtime。
    if not runtimes:  # 如果该 signal 没有 runtime。
        raise KeyError(f"Shared-data LSTM forecaster has no runtime for signal '{signal_name}'. Available: {sorted(getattr(forecaster, 'signal_runtimes', {}))}.")  # 明确指出缺失 signal。
    return max(max(int(getattr(runtime, "seq_len")) - 1, 0) for runtime in runtimes)  # 返回最大 seq_len 对应的历史上下文长度。


def _slice_timeline_signals(  # 从完整时间线中切出一个预测 chunk。
    timeline_signals: dict[str, np.ndarray],  # 完整时间线信号映射。
    start: int,  # chunk 开始行。
    end: int,  # chunk 结束行。
) -> dict[str, np.ndarray]:  # 返回每个 signal 的 chunk 切片。
    return {  # 构建切片后的信号映射。
        name: np.asarray(values, dtype=np.float32)[start:end]  # 转成 float32 后按时间维切片。
        for (name, values) in timeline_signals.items()  # 遍历全部时间线信号。
    }  # 结束切片信号映射。


def _write_signal_prediction_cache(  # 写入单个 signal 的预测序列缓存。
    forecaster,  # 已构建的 forecaster。
    *,  # 后续参数必须按关键字传入。
    timeline_signals: dict[str, np.ndarray],  # 完整时间线信号。
    timestamps: list[str],  # 完整时间线时间戳。
    meta_template: dict[str, object],  # 数据集 meta 模板。
    out_array: np.ndarray,  # 目标 memmap 数组。
    out_name: str,  # 输出字段名。
    signal_name: str,  # 输入 signal 名。
    sequence_length: int,  # 输出预测序列长度。
    progress,  # 共享进度条对象。
) -> None:  # 该函数只写入 memmap，无返回值。
    if signal_name not in timeline_signals:  # 如果时间线缺少当前 signal。
        raise KeyError(f"Shared-data timeline is missing signal '{signal_name}' required for output '{out_name}'. Available: {sorted(timeline_signals)}.")  # 显式报告缺失信号。
    num_timestamps = int(len(timestamps))  # 计算时间线总长度。
    context_steps = _forecaster_context_steps(forecaster, signal_name)  # 计算 LSTM 所需历史上下文长度。
    future_context = max(int(sequence_length) - 1, 0)  # 预测序列需要额外看到的未来上下文行数。
    row_chunk_size = max(int(_SHARED_DATA_FORECAST_ROW_CHUNK_SIZE), 1)  # 防御性保证 chunk 大小至少为 1。
    batch_size = max(int(_SHARED_DATA_FORECAST_BATCH_SIZE), 1)  # 防御性保证 batch 大小至少为 1。
    for active_start in range(0, num_timestamps, row_chunk_size):  # 按 chunk 遍历待写入时间线。
        active_end = min(active_start + row_chunk_size, num_timestamps)  # 计算当前 active chunk 结束行。
        context_start = max(active_start - context_steps, 0)  # 向前扩展 LSTM 历史上下文。
        context_end = min(active_end + future_context, num_timestamps)  # 向后扩展预测 horizon 上下文。
        local_start, local_end = active_start - context_start, active_end - context_start  # 记录 active chunk 在局部预测结果中的切片。
        chunk_signals = _slice_timeline_signals(timeline_signals, context_start, context_end)  # 切出当前预测 chunk。
        forecaster.reset()  # 重置 forecaster 状态，避免跨 chunk 泄漏。
        forecaster.set_episode(chunk_signals, meta_template)  # 将 chunk 信号设为当前 episode。
        prediction = forecaster.predict_episode_matrix(  # 批量预测整个 chunk 的目标 signal。
            chunk_signals[signal_name],  # 当前 signal 的历史/未来上下文序列。
            sequence_length,  # 目标预测序列长度。
            signal_name=signal_name,  # 指定预测 signal。
            history_timestamps=timestamps[context_start:context_end],  # 传入与 chunk 对齐的时间戳。
            batch_size=batch_size,  # 使用配置的批量大小。
        )  # 结束批量预测调用。
        out_array[active_start:active_end] = np.asarray(prediction[local_start:local_end], dtype=np.float32)  # 只写回 active 行。
        progress.update(active_end - active_start)  # 更新共享进度条。
        progress.set_postfix(signal=signal_name, rows=f"{active_end}/{num_timestamps}", refresh=False)  # 在进度条上显示当前 signal 和行数。
        del prediction, chunk_signals  # 主动释放大数组引用。
        _release_shared_data_cuda_cache(getattr(forecaster, "device", "cpu"))  # 如果使用 CUDA，则清理缓存。


def _write_price_rank_cache(  # 写入批发电价 rank 特征缓存。
    price_seq_array: np.ndarray,  # 输入电价序列 memmap。
    rank_array: np.ndarray,  # 输出 rank 特征 memmap。
    *,  # 后续参数必须按关键字传入。
    split: str,  # 当前 split 名称。
) -> None:  # 该函数只写入 memmap，无返回值。
    num_timestamps = int(price_seq_array.shape[0])  # 读取总行数。
    row_chunk_size = max(int(_SHARED_DATA_FORECAST_ROW_CHUNK_SIZE), 1)  # 防御性保证 chunk 大小至少为 1。
    for start in tqdm(range(0, num_timestamps, row_chunk_size), desc=f"shared_data[{split}] price ranks", unit="chunk", leave=True):  # 按 chunk 写 rank。
        end = min(start + row_chunk_size, num_timestamps)  # 计算当前 chunk 结束行。
        rank_array[start:end] = _rank_sequence_matrix(price_seq_array[start:end])  # 计算并写入 rank 特征。


def _write_window_relative_price_cache(  # 写入窗口相对价格和价差特征缓存。
    price_seq_array: np.ndarray,  # 输入电价序列 memmap。
    relative_array: np.ndarray,  # 输出相对价格 memmap。
    spread_array: np.ndarray,  # 输出价差强度 memmap。
    *,  # 后续参数必须按关键字传入。
    split: str,  # 当前 split 名称。
    spread_scale_eur_per_kwh: float,  # 价差归一化尺度。
) -> None:  # 该函数只写入 memmap，无返回值。
    num_timestamps = int(price_seq_array.shape[0])  # 读取总行数。
    row_chunk_size = max(int(_SHARED_DATA_FORECAST_ROW_CHUNK_SIZE), 1)  # 防御性保证 chunk 大小至少为 1。
    for start in tqdm(range(0, num_timestamps, row_chunk_size), desc=f"shared_data[{split}] relative prices", unit="chunk", leave=True):  # 按 chunk 写相对价格。
        end = min(start + row_chunk_size, num_timestamps)  # 计算当前 chunk 结束行。
        chunk = price_seq_array[start:end]  # 读取当前电价序列 chunk。
        relative_array[start:end] = _relative_price_sequence_matrix(chunk)  # 计算并写入相对价格特征。
        spread_array[start:end] = _spread_price_sequence_matrix(chunk, spread_scale_eur_per_kwh)  # 计算并写入价差特征。


def _write_split_shared_data(  # 生成单个 split 的 shared-data 缓存。
    cfg,  # 当前实验配置对象。
    *,  # 后续参数必须按关键字传入。
    split: str,  # split 名称，通常为 train 或 test。
    split_dir: Path,  # 当前 split 的输出目录。
) -> dict[str, object]:  # 返回该 split 的 manifest 内容。
    split_controls = _resolve_shared_split_controls(cfg, str(split))  # 解析并记录 split 控制项。
    dataset = build_dataset(  # 构建当前 split 对应的数据集。
        cfg,  # 传入当前配置。
        mode=str(split),  # 设置数据集模式。
        override_start_date=split_controls["start_date"],  # 使用 shared-data 合同解析出的开始日期。
        override_end_date=split_controls["end_date"],  # 使用 shared-data 合同解析出的结束日期。
    )  # 结束数据集构建。
    n_episodes, episode_length, n_agents, sequence_length = (  # 读取写缓存所需的核心维度。
        int(dataset.num_episodes()),  # 当前 split 的 episode 数。
        int(dataset.episode_length),  # 当前 split 的 episode 长度。
        int(cfg.env.num_agents),  # agent 数量。
        int(cfg.env.future_horizon) + 1,  # 预测序列长度等于 horizon + 当前步。
    )  # 结束维度读取。
    timeline_signals = {  # 从数据集内部时间线取出所有信号。
        key: np.asarray(value, dtype=np.float32)  # 每个信号统一转成 float32。
        for (key, value) in dict(getattr(dataset, "_signals", {})).items()  # 遍历数据集信号。
    }  # 结束时间线信号映射。
    raw_timestamps = getattr(dataset, "_timestamps", None)  # 读取数据集时间戳。
    timestamps = [] if raw_timestamps is None else [str(value) for value in list(raw_timestamps)]  # 时间戳转成字符串列表。
    num_timestamps = len(timestamps)  # 计算时间线总行数。
    if num_timestamps <= 0:  # shared-data 不能在空时间线上生成。
        raise ValueError(f"Cannot build shared-data timeline cache for split='{split}' with zero timestamps.")  # 显式报告空时间线。
    split_dir.mkdir(parents=True, exist_ok=True)  # 确保 split 输出目录存在。
    files, arrays = _open_split_arrays(  # 创建所有 memmap 输出数组。
        split_dir,  # 输出目录。
        num_timestamps=num_timestamps,  # 时间线总行数。
        n_agents=n_agents,  # agent 数量。
        sequence_length=sequence_length,  # 预测序列长度。
    )  # 结束 memmap 创建。
    episode_manifest, predict_specs = (  # 初始化 episode manifest 和预测字段映射。
        [],  # episode manifest 条目列表。
        {WHOLESALE_PRICE_SEQ_FIELD: WHOLESALE_PRICE_SIGNAL, "load_seq": "load", "pv_seq": "pv"},  # 输出字段到原始 signal 的映射。
    )  # 结束初始化。
    try:  # 确保 memmap 在成功或失败后都会关闭。
        forecaster = build_forecaster(cfg)  # 构建当前配置对应的 forecaster。
        vectorized = hasattr(forecaster, "predict_episode_matrix")  # 判断 forecaster 是否支持向量化预测。
        meta_template = dict(getattr(dataset, "_meta_template", {}))  # 读取数据集 meta 模板。
        forecaster.reset()  # 重置 forecaster 状态。
        forecaster.set_episode(timeline_signals, meta_template)  # 将完整时间线设为当前 episode。
        arrays["mu_t"][:], arrays["mu_next"][:] = _compute_future_mean_price(  # 写入当前和下一步未来均价。
            timeline_signals[WHOLESALE_PRICE_SIGNAL],  # 输入批发电价时间线。
            int(cfg.env.future_horizon),  # 使用当前配置的 horizon。
        )  # 结束未来均价写入。
        arrays["calendar_time"][:] = _build_calendar_time_matrix(timestamps, n_agents)  # 写入日历时间特征。
        if vectorized:  # 如果 forecaster 支持批量矩阵预测。
            total_rows = num_timestamps * len(predict_specs)  # 进度条总行数为时间步乘信号数。
            with tqdm(total=total_rows, desc=f"shared_data[{split}] forecast cache", unit="row", leave=True) as progress:  # 创建 forecast 缓存进度条。
                for (out_name, signal_name) in predict_specs.items():  # 遍历每个需要预测的 signal。
                    _write_signal_prediction_cache(  # 写入当前 signal 的预测缓存。
                        forecaster,  # 复用当前 forecaster。
                        timeline_signals=timeline_signals,  # 传入完整时间线信号。
                        timestamps=timestamps,  # 传入完整时间戳。
                        meta_template=meta_template,  # 传入 meta 模板。
                        out_array=arrays[out_name],  # 传入目标 memmap。
                        out_name=out_name,  # 传入输出字段名。
                        signal_name=signal_name,  # 传入输入 signal 名。
                        sequence_length=sequence_length,  # 传入序列长度。
                        progress=progress,  # 传入共享进度条。
                    )  # 结束单 signal 缓存写入。
            _write_price_rank_cache(arrays[WHOLESALE_PRICE_SEQ_FIELD], arrays[WHOLESALE_PRICE_RANK_SEQ_FIELD], split=str(split))  # 写入价格 rank 特征。
            _write_window_relative_price_cache(  # 写入相对价格和价差特征。
                arrays[WHOLESALE_PRICE_SEQ_FIELD],  # 输入电价序列。
                arrays[WHOLESALE_PRICE_RELATIVE_SEQ_FIELD],  # 输出相对价格。
                arrays[WHOLESALE_PRICE_SPREAD_SEQ_FIELD],  # 输出价差强度。
                split=str(split),  # 当前 split 名。
                spread_scale_eur_per_kwh=float(getattr(cfg.obs, "wholesale_price_spread_scale_eur_per_kwh", 0.20)),  # 价差尺度。
            )  # 结束相对价格缓存写入。
        else:  # 如果 forecaster 只能逐步预测。
            for step_idx in tqdm(range(num_timestamps), desc=f"shared_data[{split}] timeline", leave=True):  # 按时间步逐行预测。
                current_timestamps, end = timestamps[: step_idx + 1], step_idx + 1  # 当前预测只能看到截至当前步的历史。
                for (out_name, signal_name) in predict_specs.items():  # 遍历每个输出字段。
                    arrays[out_name][step_idx] = forecaster.predict(  # 写入当前时间步的预测序列。
                        timeline_signals[signal_name][:end],  # 传入截至当前步的 signal 历史。
                        sequence_length,  # 目标预测序列长度。
                        signal_name=signal_name,  # 指定 signal 名。
                        history_timestamps=current_timestamps,  # 传入截至当前步的时间戳。
                    ).astype(np.float32)  # 输出转成 float32。
            _write_price_rank_cache(arrays[WHOLESALE_PRICE_SEQ_FIELD], arrays[WHOLESALE_PRICE_RANK_SEQ_FIELD], split=str(split))  # 写入价格 rank 特征。
            _write_window_relative_price_cache(  # 写入相对价格和价差特征。
                arrays[WHOLESALE_PRICE_SEQ_FIELD],  # 输入电价序列。
                arrays[WHOLESALE_PRICE_RELATIVE_SEQ_FIELD],  # 输出相对价格。
                arrays[WHOLESALE_PRICE_SPREAD_SEQ_FIELD],  # 输出价差强度。
                split=str(split),  # 当前 split 名。
                spread_scale_eur_per_kwh=float(getattr(cfg.obs, "wholesale_price_spread_scale_eur_per_kwh", 0.20)),  # 价差尺度。
            )  # 结束相对价格缓存写入。
        for episode_idx in tqdm(range(n_episodes), desc=f"shared_data[{split}] manifest", unit="episode", leave=False):  # 遍历所有 episode。
            episode_manifest.append(_episode_manifest_entry(dataset, episode_idx, dataset.get_episode(episode_idx)))  # 写入 episode manifest 条目。
    finally:  # 无论生成是否成功都关闭 memmap。
        _close_split_arrays(arrays)  # 刷盘并关闭所有数组。
    manifest = {  # 构建 split manifest。
        "schema_version": _SCHEMA_VERSION,  # 记录 schema 版本。
        "cache_layout": SHARED_DATA_CACHE_LAYOUT,  # 记录缓存布局版本。
        "price_protocol_version": int(PRICE_PROTOCOL_VERSION),  # 记录价格协议版本。
        "price_observation_contract": PRICE_OBSERVATION_CONTRACT,  # 记录价格观测合同。
        "split": str(split),  # 记录 split 名。
        "num_episodes": n_episodes,  # 记录 episode 数量。
        "num_timestamps": num_timestamps,  # 记录时间线行数。
        "episode_length": episode_length,  # 记录 episode 长度。
        "num_agents": n_agents,  # 记录 agent 数量。
        "sequence_length": sequence_length,  # 记录预测序列长度。
        "split_controls": split_controls,  # 记录 split 控制项。
        "dropped_tail_steps": int(getattr(dataset, "dropped_tail_steps", 0)),  # 记录数据集尾部丢弃步数。
        "files": files,  # 记录字段到文件名映射。
        "episodes": episode_manifest,  # 记录 episode 切片信息。
    }  # 结束 split manifest。
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")  # 写入 split manifest。
    return manifest  # 返回 split manifest 内容。


def _shared_data_root(root: str | Path | None = None) -> Path:  # 解析 mainline shared-data 根目录。
    return get_shared_data_root(root) / "mainline"  # 所有当前合同缓存都位于 mainline 子目录。


def _root_manifest_is_complete(shared_dir: Path, signature_hash: str) -> bool:  # 判断目标 shared-data 目录是否已经完整可复用。
    try:  # 尝试读取根 manifest。
        manifest = json.loads((shared_dir / "manifest.json").read_text(encoding="utf-8"))  # 读取并解析根 manifest。
    except (OSError, json.JSONDecodeError, FileNotFoundError):  # 目录缺失、读取失败或 JSON 损坏都视为不可复用。
        return False  # 返回不可复用。
    return str(manifest.get("signature_hash", "")) == str(signature_hash) and all(  # 签名一致且 train/test manifest 都存在才完整。
        (shared_dir / split / "manifest.json").exists()  # 检查当前 split manifest 是否存在。
        for split in ("train", "test")  # 只检查 canonical train/test split。
    )  # 返回完整性判断结果。


class SharedDataResult:  # 表示 shared-data 生成或复用的结果。
    def __init__(  # 初始化结果对象。
        self,  # 当前实例。
        *,  # 后续参数必须按关键字传入。
        shared_data_dir: Path,  # shared-data package 目录。
        signature_hash: str,  # shared-data 签名哈希。
        manifest: dict[str, object],  # 根 manifest 内容。
        reused: bool,  # 是否复用了已有缓存。
    ):  # 结束初始化签名。
        self.shared_data_dir, self.signature_hash, self.manifest, self.reused = (  # 一次性写入四个结果字段。
            Path(shared_data_dir),  # 规范保存 shared-data 路径对象。
            str(signature_hash),  # 规范保存签名字符串。
            dict(manifest),  # 复制 manifest，避免外部原地修改。
            bool(reused),  # 规范保存复用标记。
        )  # 结束结果字段赋值。


def build_shared_data_status_summary(  # 构建给 notebook/脚本展示的 shared-data 状态摘要。
    result: SharedDataResult,  # shared-data 生成结果。
    *,  # 后续参数必须按关键字传入。
    test_start_date: str | None = None,  # 可选测试开始日期。
    test_end_date: str | None = None,  # 可选测试结束日期。
) -> dict[str, object]:  # 返回状态摘要字典。
    return {  # 构建状态摘要。
        "shared_data_status": "reused_existing_shared_data" if result.reused else "generated_new_shared_data",  # 根据 reused 标记写状态。
        "shared_data_message": "Reused existing shared MADRL data package." if result.reused else "Generated a new shared MADRL data package from available forecast artifacts.",  # 根据 reused 标记写提示信息。
        "shared_data_dir": str(result.shared_data_dir),  # 记录 shared-data 目录。
        "shared_data_signature": str(result.signature_hash),  # 记录 shared-data 签名。
        "shared_data_reused": bool(result.reused),  # 记录是否复用。
        "test_start_date": test_start_date,  # 透传测试开始日期。
        "test_end_date": test_end_date,  # 透传测试结束日期。
    }  # 结束状态摘要。


class PrecomputedObservationStore:  # 运行时读取 shared-data split 的预计算观测缓存。
    def __init__(self, split_dir: str | Path):  # 初始化 split 级预计算缓存读取器。
        self.split_dir = Path(split_dir).resolve()  # 解析 split 目录为绝对路径。
        self.manifest = json.loads((self.split_dir / "manifest.json").read_text(encoding="utf-8"))  # 读取 split manifest。
        assert_no_legacy_price_schema(dict(self.manifest.get("files", {})).keys(), context="Shared-data manifest")  # 拒绝旧价格字段。
        if int(self.manifest.get("schema_version", -1)) != int(_SCHEMA_VERSION):  # 校验 split schema 版本。
            raise ValueError(f"Unsupported shared-data schema version at predictors.shared_data.PrecomputedObservationStore(...): old object '{self.split_dir/'manifest.json'}' declares schema_version={self.manifest.get('schema_version')!r}. New contract expects schema_version={_SCHEMA_VERSION}, price_observation_contract='{PRICE_OBSERVATION_CONTRACT}', cache_layout='{SHARED_DATA_CACHE_LAYOUT}', and files '{WHOLESALE_PRICE_RELATIVE_SEQ_FIELD}'/'{WHOLESALE_PRICE_SPREAD_SEQ_FIELD}'. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun the consuming notebook or entrypoint.")  # 旧 schema 显式失败。
        if str(self.manifest.get("cache_layout", "")) != SHARED_DATA_CACHE_LAYOUT:  # 校验缓存布局版本。
            raise ValueError(f"Unsupported shared-data cache layout at predictors.shared_data.PrecomputedObservationStore(...): old object '{self.split_dir/'manifest.json'}' declares cache_layout={self.manifest.get('cache_layout')!r}. New contract expects cache_layout='{SHARED_DATA_CACHE_LAYOUT}' with one cached row per physical timestamp. Re-run notebooks/forecast/forecast_lstm.ipynb.")  # 旧布局显式失败。
        if str(self.manifest.get("price_observation_contract", "")) != PRICE_OBSERVATION_CONTRACT:  # 校验价格观测合同。
            raise ValueError(f"Unsupported shared-data price observation contract at predictors.shared_data.PrecomputedObservationStore(...): old object '{self.split_dir/'manifest.json'}' declares price_observation_contract={self.manifest.get('price_observation_contract')!r}. New contract expects '{PRICE_OBSERVATION_CONTRACT}'. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun the consuming notebook or entrypoint.")  # 旧价格合同显式失败。
        if int(self.manifest.get("price_protocol_version", PRICE_PROTOCOL_VERSION)) != int(PRICE_PROTOCOL_VERSION):  # 校验价格协议版本。
            raise ValueError(f"Unsupported shared-data price protocol version at '{self.split_dir}': expected={PRICE_PROTOCOL_VERSION}, actual={self.manifest.get('price_protocol_version')!r}.")  # 协议版本不匹配时失败。
        missing_price_files = sorted(  # 检查当前价格合同要求的派生价格文件。
            {WHOLESALE_PRICE_RANK_SEQ_FIELD, WHOLESALE_PRICE_RELATIVE_SEQ_FIELD, WHOLESALE_PRICE_SPREAD_SEQ_FIELD}  # 必需价格派生字段。
            - set(dict(self.manifest.get("files", {})))  # manifest 中实际声明的字段集合。
        )  # 结束缺失字段计算。
        if missing_price_files:  # 如果缺少必需价格派生文件。
            raise KeyError(f"Unsupported shared-data files at predictors.shared_data.PrecomputedObservationStore(...): old object '{self.split_dir/'manifest.json'}' is missing {missing_price_files}. New contract expects schema_version={_SCHEMA_VERSION}, price_observation_contract='{PRICE_OBSERVATION_CONTRACT}', and window-relative wholesale price observations. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun the consuming notebook or entrypoint.")  # 缺字段时显式失败。
        self._arrays = {  # 打开 manifest 声明的全部 .npy 文件。
            name: np.load(self.split_dir / file_name, mmap_mode="r")  # 以只读 mmap 打开数组。
            for (name, file_name) in dict(self.manifest.get("files", {})).items()  # 遍历 manifest 文件映射。
        }  # 结束数组映射。

    def episode(self, episode_idx: int) -> dict[str, np.ndarray]:  # 读取某个 episode 对应的预计算观测数组。
        episodes = list(self.manifest.get("episodes") or [])  # 读取 episode manifest 列表。
        if int(episode_idx) < 0 or int(episode_idx) >= len(episodes):  # 校验 episode_idx 范围。
            raise IndexError(f"shared-data episode_idx={episode_idx} is out of range [0, {len(episodes)-1}].")  # 越界时抛出明确错误。
        entry = dict(episodes[int(episode_idx)])  # 复制当前 episode 条目。
        start, end = int(entry["active_start_idx"]), int(entry["active_end_idx"])  # 读取 active 时间线范围。
        next_active = entry.get("next_active_idx")  # 读取 bootstrap 下一 active 行。
        indices = list(range(start, end))  # 构建 active 时间线索引。
        if next_active is not None:  # 如果 manifest 提供 bootstrap 行。
            indices.append(int(next_active))  # 将 bootstrap 行追加到 episode 末尾。
        if not indices:  # 如果该 episode 没有任何索引。
            return {name: np.asarray(array[[]]) for (name, array) in self._arrays.items()}  # 返回所有字段的空切片。
        index_array = np.asarray(indices, dtype=np.int64)  # 将索引列表转成 NumPy 整数数组。
        num_timestamps = int(self.manifest.get("num_timestamps", 0))  # 读取 manifest 中声明的时间线行数。
        if int(np.max(index_array)) >= num_timestamps:  # 防御性检查索引是否超出缓存范围。
            raise ValueError(f"Shared-data manifest '{self.split_dir/'manifest.json'}' contains episode index {int(np.max(index_array))} outside num_timestamps={num_timestamps}. Re-run notebooks/forecast/forecast_lstm.ipynb.")  # manifest 损坏时提示重跑。
        return {name: np.asarray(array[index_array]) for (name, array) in self._arrays.items()}  # 按 episode 索引取出所有字段。


def load_madrl_shared_data_manifest(path: str | Path) -> dict[str, object]:  # 读取 shared-data 根 manifest。
    return json.loads((Path(path).resolve() / "manifest.json").read_text(encoding="utf-8"))  # 从指定目录读取 manifest.json。


def _manifest_int(value: object, *, field: str, manifest_path: Path) -> int:  # 将 manifest 字段解析为整数。
    try:  # 尝试转换成 int。
        return int(value)  # 返回整数值。
    except (TypeError, ValueError) as exc:  # 捕获缺失或非整数字段。
        raise ValueError(f"Invalid shared-data manifest field '{field}' at '{manifest_path}': expected integer, actual={value!r}. Expected a current shared MADRL data contract. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun the consuming notebook or entrypoint.") from exc  # 抛出带上下文的合同错误。


def validate_madrl_shared_data_runtime_contract(  # 校验运行时配置与 shared-data package 是否匹配。
    cfg,  # 当前运行配置。
    *,  # 后续参数必须按关键字传入。
    shared_data_dir: str | Path,  # shared-data 根目录。
    split_dir: str | Path,  # 当前 split 目录。
    split_manifest: dict[str, object],  # 当前 split manifest。
) -> dict[str, object]:  # 返回根 manifest。
    shared_data_dir = Path(shared_data_dir).resolve()  # 解析根目录绝对路径。
    split_dir = Path(split_dir).resolve()  # 解析 split 目录绝对路径。
    root_manifest_path = shared_data_dir / "manifest.json"  # 构造根 manifest 路径。
    split_manifest_path = split_dir / "manifest.json"  # 构造 split manifest 路径。
    root_manifest = load_madrl_shared_data_manifest(shared_data_dir)  # 读取根 manifest。
    expected_horizon = int(cfg.env.future_horizon)  # 当前配置期望的 future horizon。
    expected_sequence_length = expected_horizon + 1  # 当前配置期望的预测序列长度。
    split_name = str(split_manifest.get("split") or split_dir.name)  # 解析 split 名。
    expected_window_spec = resolve_dataset_window_spec(cfg, split_name)  # 根据当前配置解析 split 窗口。
    expected_episode_length = int(expected_window_spec["episode_length"])  # 当前配置期望的 episode 长度。
    data_controls = dict(root_manifest.get("data_controls") or {})  # 读取根 manifest 中的数据控制项。
    actual_horizon = _manifest_int(  # 读取 shared-data 实际 horizon。
        data_controls.get("future_horizon"),  # manifest 中的 future_horizon。
        field="data_controls.future_horizon",  # 错误上下文字段名。
        manifest_path=root_manifest_path,  # 错误上下文 manifest 路径。
    )  # 结束 horizon 读取。
    actual_sequence_length = _manifest_int(  # 读取 split manifest 实际序列长度。
        split_manifest.get("sequence_length"),  # manifest 中的 sequence_length。
        field="sequence_length",  # 错误上下文字段名。
        manifest_path=split_manifest_path,  # 错误上下文 manifest 路径。
    )  # 结束 sequence_length 读取。
    actual_episode_length = _manifest_int(  # 读取 split manifest 实际 episode 长度。
        split_manifest.get("episode_length"),  # manifest 中的 episode_length。
        field="episode_length",  # 错误上下文字段名。
        manifest_path=split_manifest_path,  # 错误上下文 manifest 路径。
    )  # 结束 episode_length 读取。
    mismatches = []  # 收集所有合同不匹配项。
    if int(root_manifest.get("schema_version", -1)) != int(_SCHEMA_VERSION):  # 检查根 schema 版本。
        mismatches.append(f"old shared-data root manifest '{root_manifest_path}' declares schema_version={root_manifest.get('schema_version')!r}, but the new contract expects schema_version={_SCHEMA_VERSION}, price_observation_contract='{PRICE_OBSERVATION_CONTRACT}', cache_layout='{SHARED_DATA_CACHE_LAYOUT}', and files '{WHOLESALE_PRICE_RELATIVE_SEQ_FIELD}'/'{WHOLESALE_PRICE_SPREAD_SEQ_FIELD}'")  # 记录根 schema mismatch。
    if int(split_manifest.get("schema_version", -1)) != int(_SCHEMA_VERSION):  # 检查 split schema 版本。
        mismatches.append(f"old split manifest '{split_manifest_path}' declares schema_version={split_manifest.get('schema_version')!r}, but the new contract expects schema_version={_SCHEMA_VERSION}, price_observation_contract='{PRICE_OBSERVATION_CONTRACT}', cache_layout='{SHARED_DATA_CACHE_LAYOUT}', and files '{WHOLESALE_PRICE_RELATIVE_SEQ_FIELD}'/'{WHOLESALE_PRICE_SPREAD_SEQ_FIELD}'")  # 记录 split schema mismatch。
    if str(split_manifest.get("cache_layout", "")) != SHARED_DATA_CACHE_LAYOUT:  # 检查 split cache layout。
        mismatches.append(f"old split manifest '{split_manifest_path}' declares cache_layout={split_manifest.get('cache_layout')!r}, but the new contract expects cache_layout='{SHARED_DATA_CACHE_LAYOUT}'")  # 记录 layout mismatch。
    if str(root_manifest.get("price_observation_contract", "")) != PRICE_OBSERVATION_CONTRACT:  # 检查根价格观测合同。
        mismatches.append(f"old shared-data root manifest '{root_manifest_path}' declares price_observation_contract={root_manifest.get('price_observation_contract')!r}, but the new contract expects '{PRICE_OBSERVATION_CONTRACT}'")  # 记录根价格合同 mismatch。
    if str(split_manifest.get("price_observation_contract", "")) != PRICE_OBSERVATION_CONTRACT:  # 检查 split 价格观测合同。
        mismatches.append(f"old split manifest '{split_manifest_path}' declares price_observation_contract={split_manifest.get('price_observation_contract')!r}, but the new contract expects '{PRICE_OBSERVATION_CONTRACT}'")  # 记录 split 价格合同 mismatch。
    missing_price_files = sorted(  # 检查 split manifest 是否声明全部必需价格特征。
        {WHOLESALE_PRICE_RANK_SEQ_FIELD, WHOLESALE_PRICE_RELATIVE_SEQ_FIELD, WHOLESALE_PRICE_SPREAD_SEQ_FIELD}  # 当前合同必需字段。
        - set(dict(split_manifest.get("files", {})))  # 实际文件字段集合。
    )  # 结束缺失价格字段计算。
    if missing_price_files:  # 如果有价格字段缺失。
        mismatches.append(f"old split manifest '{split_manifest_path}' is missing {missing_price_files}, but the new contract requires '{PRICE_OBSERVATION_CONTRACT}' window-relative wholesale price observations")  # 记录缺字段 mismatch。
    if actual_horizon != expected_horizon:  # 检查 horizon 是否匹配当前配置。
        mismatches.append(f"old shared-data object '{shared_data_dir}' declares data_controls.future_horizon={actual_horizon}, but current cfg.env.future_horizon={expected_horizon}")  # 记录 horizon mismatch。
    if actual_episode_length != expected_episode_length:  # 检查 episode 长度是否匹配当前配置。
        mismatches.append(f"old split manifest '{split_manifest_path}' declares episode_length={actual_episode_length}, but current split='{split_name}' expects episode_length={expected_episode_length}")  # 记录 episode_length mismatch。
    if actual_sequence_length != expected_sequence_length:  # 检查序列长度是否匹配当前配置。
        mismatches.append(f"old split manifest '{split_manifest_path}' declares sequence_length={actual_sequence_length}, but current cfg.env.future_horizon={expected_horizon} requires sequence_length={expected_sequence_length}")  # 记录 sequence_length mismatch。
    if mismatches:  # 如果存在任一合同不匹配。
        raise ValueError("Shared-data horizon contract mismatch at scripts.builder.build_env(...): " + "; ".join(mismatches) + ". Expected a shared MADRL data package generated with the current config and price_observation_contract='" + PRICE_OBSERVATION_CONTRACT + "'. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun notebooks/madrl/local_MPC.ipynb or notebooks/madrl/ADMM_mpc.ipynb.")  # 汇总并抛出合同错误。
    return root_manifest  # 校验通过时返回根 manifest。


def select_shared_data_episode_indices(  # 根据运行时日期窗口选择 shared-data episode。
    manifest: dict[str, object],  # split manifest。
    *,  # 后续参数必须按关键字传入。
    start_date: str | None,  # 运行时开始日期。
    end_date: str | None,  # 运行时结束日期。
) -> list[int]:  # 返回满足窗口的 episode_idx 列表。
    if not (episodes := list(manifest.get("episodes") or [])):  # 如果 manifest 没有 episode 元数据。
        raise ValueError("Shared-data split manifest does not contain episode metadata.")  # 显式拒绝缺失 episode metadata。
    if start_date in (None, "") and end_date in (None, ""):  # 如果没有日期筛选。
        return [int(entry["episode_idx"]) for entry in episodes]  # 返回全部 episode_idx。
    normalized_start, normalized_end = (  # 将日期筛选边界转为 date。
        None if start_date in (None, "") else pd.Timestamp(str(start_date)).date(),  # 规范化开始日期。
        None if end_date in (None, "") else pd.Timestamp(str(end_date)).date(),  # 规范化结束日期。
    )  # 结束日期边界规范化。
    selected: list[int] = []  # 收集选中的 episode_idx。
    selected_dates: list[tuple[int, object, object]] = []  # 收集选中 episode 的首末日期。
    for entry in episodes:  # 遍历 manifest 中的全部 episode。
        first_local_date = None if entry.get("first_local_date") in (None, "") else pd.Timestamp(str(entry["first_local_date"])).date()  # 解析 episode 首日。
        last_local_date = None if entry.get("last_local_date") in (None, "") else pd.Timestamp(str(entry["last_local_date"])).date()  # 解析 episode 末日。
        if normalized_start is not None and (first_local_date is None or first_local_date < normalized_start):  # 如果 episode 没有完全覆盖开始边界。
            continue  # 跳过该 episode。
        if normalized_end is not None and (last_local_date is None or last_local_date > normalized_end):  # 如果 episode 超出结束边界。
            continue  # 跳过该 episode。
        selected.append(int(entry["episode_idx"]))  # 记录选中的 episode_idx。
        selected_dates.append((int(entry["episode_idx"]), first_local_date, last_local_date))  # 记录选中 episode 的日期边界。
    if not selected:  # 如果没有任何 episode 完全落在窗口内。
        raise ValueError(f"No shared-data episodes fall fully inside the requested date window: start_date={start_date!r}, end_date={end_date!r}.")  # 抛出空选择错误。
    if any(current != previous + 1 for (previous, current) in zip(selected, selected[1:])):  # 检查 episode_idx 是否连续。
        raise ValueError(f"Shared-data test date window selected non-contiguous episode indices from the full-year manifest: selected={selected}, start_date={start_date!r}, end_date={end_date!r}. Re-run notebooks/forecast/forecast_lstm.ipynb to regenerate a complete full-year manifest.")  # 非连续时要求重建完整 manifest。
    for (_, _, previous_last), (current_idx, current_first, _) in zip(selected_dates, selected_dates[1:]):  # 逐对检查相邻 episode 的日期连续性。
        if previous_last is None or current_first is None:  # 如果缺少日期 metadata。
            raise ValueError(f"Shared-data test date window selected episode metadata without local dates before episode_idx={current_idx}: start_date={start_date!r}, end_date={end_date!r}. Re-run notebooks/forecast/forecast_lstm.ipynb to regenerate a complete full-year test manifest.")  # 缺少日期时要求重建。
        previous_last_date, current_first_date = pd.Timestamp(previous_last).date(), pd.Timestamp(current_first).date()  # 规范化相邻日期。
        if current_first_date not in {previous_last_date, (pd.Timestamp(previous_last) + pd.Timedelta(days=1)).date()}:  # 允许同日或下一日衔接。
            raise ValueError(f"Shared-data test date window selected non-contiguous local dates before episode_idx={current_idx}: start_date={start_date!r}, end_date={end_date!r}. Re-run notebooks/forecast/forecast_lstm.ipynb to regenerate a complete full-year test manifest.")  # 日期不连续时要求重建。
    return selected  # 返回连续且完整覆盖窗口的 episode_idx。


def ensure_madrl_shared_data(  # 确保当前配置存在 canonical shared-data package。
    cfg,  # 当前实验配置对象。
    *,  # 后续参数必须按关键字传入。
    root: str | Path | None = None,  # 可选 artifact 根目录覆盖。
) -> SharedDataResult:  # 返回生成或复用结果。
    forecast_ready = None if str(cfg.forecast.type).strip().lower() != "lstm" else ensure_lstm_artifacts(cfg, device=cfg.runtime.device)  # LSTM 模式确保 artifact 存在，非 LSTM 模式不需要 artifact。
    artifact_info = {  # 构建 artifact 状态和签名指纹。
        "forecast_ready": forecast_ready,  # 记录 forecast artifact 准备结果。
        "fingerprint": {"mode": "non_lstm"} if forecast_ready is None else _artifact_fingerprint(forecast_ready),  # 非 LSTM 使用固定指纹，LSTM 使用文件指纹。
    }  # 结束 artifact 信息。
    signature_payload = _shared_data_signature_payload(cfg, artifact_fingerprint=dict(artifact_info["fingerprint"]))  # 构建 shared-data 签名 payload。
    signature_hash = _signature_hash(signature_payload)  # 计算 shared-data 签名目录名。
    root_dir = _shared_data_root(root)  # 解析 mainline shared-data 根目录。
    shared_dir = (root_dir / signature_hash).resolve()  # 解析当前签名对应的最终目录。
    root_dir.mkdir(parents=True, exist_ok=True)  # 确保 shared-data 根目录存在。
    if _root_manifest_is_complete(shared_dir, signature_hash):  # 如果目标目录已有完整匹配 manifest。
        return SharedDataResult(  # 直接返回复用结果。
            shared_data_dir=shared_dir,  # 复用的 shared-data 目录。
            signature_hash=signature_hash,  # 当前签名哈希。
            manifest=load_madrl_shared_data_manifest(shared_dir),  # 读取已有根 manifest。
            reused=True,  # 标记为复用已有缓存。
        )  # 结束复用结果。
    temp_dir = (root_dir / f"{signature_hash}.tmp.{uuid.uuid4().hex}").resolve()  # 创建原子替换用临时目录路径。
    try:  # 确保失败时临时目录会被清理。
        temp_dir.mkdir(parents=True, exist_ok=False)  # 创建唯一临时目录。
        train_manifest, test_manifest = (  # 生成 train/test 两个 split。
            _write_split_shared_data(cfg, split="train", split_dir=temp_dir / "train"),  # 生成 train split 缓存。
            _write_split_shared_data(cfg, split="test", split_dir=temp_dir / "test"),  # 生成 test split 缓存。
        )  # 结束 split 生成。
        manifest = {  # 构建根 manifest。
            "schema_version": _SCHEMA_VERSION,  # 记录 schema 版本。
            "cache_layout": SHARED_DATA_CACHE_LAYOUT,  # 记录缓存布局版本。
            "price_protocol_version": int(PRICE_PROTOCOL_VERSION),  # 记录价格协议版本。
            "price_observation_contract": PRICE_OBSERVATION_CONTRACT,  # 记录价格观测合同。
            "signature_hash": signature_hash,  # 记录签名哈希。
            "signature": _normalize_for_signature(signature_payload),  # 记录规范化签名 payload。
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # 记录 UTC 创建时间。
            "shared_data_dir": str(shared_dir),  # 记录最终 shared-data 目录。
            "artifact_inventory": dict(artifact_info["fingerprint"]),  # 记录 artifact 指纹快照。
            "data_controls": {  # 记录数据和窗口控制项。
                "agent_profiles": list(cfg.data.agent_profiles),  # agent profile 列表。
                "agent_bus_ids": list(cfg.grid.agent_bus_ids),  # agent bus id 列表。
                "train_year": int(cfg.data.train_year),  # 训练年份。
                "test_year": int(cfg.data.test_year),  # 测试年份。
                "train_start_date": cfg.data.train_start_date,  # 训练开始日期。
                "train_end_date": cfg.data.train_end_date,  # 训练结束日期。
                "test_manifest_scope": "full_year",  # 测试 manifest 范围固定为全年。
                "test_selection_contract": "runtime_date_selection",  # 测试日期选择发生在运行时。
                "train_window_days": int(getattr(cfg.env, "train_window_days", 1)),  # 训练窗口天数。
                "window_stride_days": int(getattr(cfg.env, "window_stride_days", 1)),  # 窗口步长天数。
                "train_episode_limit": int(resolve_train_episode_limit(cfg)),  # 训练 episode 上限。
                "test_episode_limit": int(resolve_test_episode_limit(cfg)),  # 测试 episode 上限。
                "load_scale": _normalize_for_signature(list(cfg.data.load_scale)),  # 负荷缩放列表。
                "pv_scale": _normalize_for_signature(list(cfg.data.pv_scale)),  # PV 缩放列表。
                "pv_capacity_kw": _normalize_for_signature(list(cfg.data.pv_capacity_kw)),  # PV 容量列表。
                "load_components": list(cfg.data.load_components),  # 负荷组件列表。
                "pv_reference": str(cfg.data.pv_reference),  # PV reference。
                "future_horizon": int(cfg.env.future_horizon),  # 未来预测 horizon。
                "episode_limit": int(cfg.env.episode_limit),  # 环境 episode limit。
                "price_observation_contract": PRICE_OBSERVATION_CONTRACT,  # 价格观测合同。
                "wholesale_price_spread_scale_eur_per_kwh": float(getattr(cfg.obs, "wholesale_price_spread_scale_eur_per_kwh", 0.20)),  # 价差归一化尺度。
            },  # 结束数据控制项。
            "splits": {"train": train_manifest, "test": test_manifest},  # 嵌入 train/test split manifest。
        }  # 结束根 manifest。
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")  # 写入临时根 manifest。
        if shared_dir.exists():  # 如果最终目录已存在。
            shutil.rmtree(shared_dir, ignore_errors=True)  # 删除旧目录以便原子替换。
        temp_dir.replace(shared_dir)  # 将临时目录替换为最终 shared-data 目录。
        return SharedDataResult(shared_data_dir=shared_dir, signature_hash=signature_hash, manifest=manifest, reused=False)  # 返回新生成结果。
    finally:  # 无论成功或失败都检查临时目录。
        if temp_dir.exists():  # 如果临时目录仍然存在。
            shutil.rmtree(temp_dir, ignore_errors=True)  # 清理残留临时目录。
