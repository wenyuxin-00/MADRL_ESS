"""LSTM 时序预测器。

使用训练好的 LSTM 模型对电价、负荷、光伏等信号进行多步预测。

主要类:
    LSTMForecaster -- LSTM 预测器
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from predictors.base import Forecaster
from predictors.lstm_model import LSTMForecastModel

# LSTM 产物文件的标准后缀
LSTM_META_SUFFIX = "_meta.json"       # 元数据文件后缀
LSTM_SCALER_SUFFIX = "_scaler.pkl"    # 标准化器文件后缀
# 元数据文件中必须包含的字段列表
LSTM_REQUIRED_META_FIELDS = (
    "seq_len",
    "pred_len",
    "hidden_size",
    "num_layers",
    "dropout",
)


def resolve_lstm_artifact_paths(
    model_path,
    meta_path=None,
    scaler_path=None,
) -> tuple[Path, Path, Path]:
    """解析单个已保存 LSTM 模型的标准伴随文件路径。

    根据模型文件路径自动推导元数据和标准化器的存放位置，
    遵循 ``<stem>_meta.json`` 和 ``<stem>_scaler.pkl`` 命名约定。

    参数:
        model_path: 模型权重文件路径（.pt 文件）。
        meta_path: 元数据文件路径，为 None 时按命名约定自动推导。
        scaler_path: 标准化器文件路径，为 None 时按命名约定自动推导。

    返回:
        (model_path, meta_path, scaler_path) 三元组。
    """
    model_path = Path(model_path)
    stem = model_path.stem
    # 未指定时按约定在同一目录下生成伴随文件名
    meta_path = Path(meta_path) if meta_path is not None else model_path.with_name(f"{stem}{LSTM_META_SUFFIX}")
    scaler_path = (
        Path(scaler_path)
        if scaler_path is not None
        else model_path.with_name(f"{stem}{LSTM_SCALER_SUFFIX}")
    )
    return model_path, meta_path, scaler_path


def save_lstm_forecaster_artifacts(
    model_path,
    state_dict,
    scaler,
    *,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    signal_name: str = "price",
    future_horizon: int | None = None,
) -> dict[str, str]:
    """保存单个信号的标准 模型/元数据/标准化器 三件套产物。

    参数:
        model_path: 模型权重的目标保存路径。
        state_dict: PyTorch 模型的 state_dict。
        scaler: 已拟合的标准化器对象（不可为 None）。
        seq_len: 输入序列长度。
        pred_len: 预测步数。
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 层数。
        dropout: Dropout 比率。
        signal_name: 信号名称，默认 ``"price"``。
        future_horizon: 预测时域长度，为 None 时使用 pred_len。

    返回:
        包含三个保存路径的字典: ``model_path``, ``meta_path``, ``scaler_path``。
    """
    if scaler is None:
        raise ValueError("LSTM forecaster artifacts require a fitted scaler object.")

    model_path, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    torch.save(state_dict, model_path)

    # 保存元数据 JSON（包含模型结构参数和信号信息）
    meta = {
        "artifact_format": "lstm_forecaster_v2",
        "signal_name": str(signal_name),
        "future_horizon": int(pred_len if future_horizon is None else future_horizon),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 保存标准化器（pickle 序列化）
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "scaler_path": str(scaler_path),
    }


def load_lstm_forecaster_artifacts(
    model_path,
    meta_path=None,
    scaler_path=None,
) -> tuple[dict, object]:
    """加载单个信号模型的标准 元数据/标准化器 伴随文件。

    参数:
        model_path: 模型权重文件路径，用于推导伴随文件位置。
        meta_path: 元数据文件路径，为 None 时自动推导。
        scaler_path: 标准化器文件路径，为 None 时自动推导。

    返回:
        (meta, scaler) 元组，meta 为字典，scaler 为反序列化的标准化器对象。

    异常:
        FileNotFoundError: 元数据或标准化器文件不存在时抛出。
        ValueError: 元数据缺少必要字段时抛出。
    """
    _, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path, meta_path, scaler_path)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM meta artifact: '{meta_path}'. Expected the standard '<stem>_meta.json' file next to the model."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM scaler artifact: '{scaler_path}'. Expected the standard '<stem>_scaler.pkl' file next to the model."
        )

    # 读取并校验元数据
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing_fields = [field for field in LSTM_REQUIRED_META_FIELDS if field not in meta]
    if missing_fields:
        raise ValueError(
            f"LSTM meta artifact '{meta_path}' is missing required fields: {missing_fields}"
        )

    # 反序列化标准化器
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    return meta, scaler


@dataclass
class _SignalForecasterRuntime:
    """单个信号的 LSTM 预测运行时上下文。

    封装了一个信号所需的模型、标准化器及超参数，供 LSTMForecaster 内部使用。

    属性:
        signal_name: 信号名称（如 ``"price"``、``"load"``、``"pv"``）。
        seq_len: 模型输入序列长度。
        pred_len: 模型单次预测步数。
        hidden_size: LSTM 隐藏层维度。
        num_layers: LSTM 层数。
        dropout: Dropout 比率。
        model: 已加载的 PyTorch LSTM 模型（推理模式）。
        scaler: 标准化器对象，用于输入归一化和输出反归一化；可为 None。
    """
    signal_name: str
    seq_len: int
    pred_len: int
    hidden_size: int
    num_layers: int
    dropout: float
    model: torch.nn.Module
    scaler: object | None


class LSTMForecaster(Forecaster):
    """运行时 LSTM 预测器，支持共享信号和每智能体信号的多步预测。

    支持单信号和多信号模式：
    - 单信号模式：通过 ``model_path`` 直接构造，或通过 ``from_artifacts`` 加载。
    - 多信号模式：通过 ``from_signal_artifacts`` 同时加载多个信号的模型。

    当历史序列不足以覆盖整个预测时域时，会自动进行滚动递推预测。

    属性:
        device: 推理设备（CPU 或 CUDA）。
        signal_runtimes: 信号名到运行时上下文的映射字典。
        seq_len: 兼容属性，主信号的输入序列长度。
        pred_len: 兼容属性，主信号的预测步数。
        scaler: 兼容属性，主信号的标准化器。
        model: 兼容属性，主信号的 LSTM 模型。
    """

    def __init__(
        self,
        model_path: str | None = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.23,
        pred_len: int = 4,
        seq_len: int = 1344,
        device: str | torch.device = "cpu",
        scaler=None,
        signal_runtimes: dict[str, _SignalForecasterRuntime] | None = None,
    ):
        """初始化 LSTM 预测器。

        参数:
            model_path: 模型权重文件路径，为 None 时使用随机初始化的模型。
            hidden_size: LSTM 隐藏层维度，默认 128。
            num_layers: LSTM 层数，默认 2。
            dropout: Dropout 比率，默认 0.23。
            pred_len: 单次预测步数，默认 4。
            seq_len: 输入序列长度，默认 1344。
            device: 推理设备，默认 ``"cpu"``。
            scaler: 标准化器对象，可为 None。
            signal_runtimes: 预构建的信号运行时字典；为 None 时自动创建默认 price 运行时。
        """
        self.device = torch.device(device)

        # 未提供运行时字典时，使用传入参数构建默认的 price 信号运行时
        if signal_runtimes is None:
            runtime = self._build_runtime(
                signal_name="price",
                model_path=model_path,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                pred_len=pred_len,
                seq_len=seq_len,
                scaler=scaler,
                device=self.device,
            )
            signal_runtimes = {"price": runtime}

        self.signal_runtimes = dict(signal_runtimes)
        self._set_legacy_attributes()

    def _set_legacy_attributes(self) -> None:
        """将首选信号的运行时参数暴露为实例属性，保持历史单信号 API 的兼容性。"""
        # 优先选择 price 信号，否则取第一个可用信号
        preferred_signal = "price" if "price" in self.signal_runtimes else next(iter(self.signal_runtimes))
        runtime = self.signal_runtimes[preferred_signal]
        self.seq_len = int(runtime.seq_len)
        self.pred_len = int(runtime.pred_len)
        self.scaler = runtime.scaler
        self.model = runtime.model

    def _sync_legacy_price_runtime(self) -> None:
        """将实例属性的直接修改同步回 price 运行时，兼容测试和 notebook 的直接赋值。"""
        if "price" not in self.signal_runtimes:
            return
        runtime = self.signal_runtimes["price"]
        runtime.seq_len = int(getattr(self, "seq_len", runtime.seq_len))
        runtime.pred_len = int(getattr(self, "pred_len", runtime.pred_len))
        runtime.scaler = getattr(self, "scaler", runtime.scaler)
        runtime.model = getattr(self, "model", runtime.model)

    @staticmethod
    def _build_runtime(
        *,
        signal_name: str,
        model_path: str | None,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pred_len: int,
        seq_len: int,
        scaler,
        device: torch.device,
    ) -> _SignalForecasterRuntime:
        """构建单个信号的运行时上下文（模型创建、权重加载、推理模式切换）。

        参数:
            signal_name: 信号名称。
            model_path: 模型权重路径，为 None 时使用随机初始化。
            hidden_size: LSTM 隐藏层维度。
            num_layers: LSTM 层数。
            dropout: Dropout 比率。
            pred_len: 预测步数。
            seq_len: 输入序列长度。
            scaler: 标准化器对象。
            device: 目标设备。

        返回:
            初始化完成的 _SignalForecasterRuntime 实例。
        """
        # 创建模型并移动到目标设备
        model = LSTMForecastModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
        ).to(device)

        # 如果提供了权重文件则加载
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        model.eval()  # 切换到推理模式
        return _SignalForecasterRuntime(
            signal_name=str(signal_name),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            model=model,
            scaler=scaler,
        )

    @classmethod
    def from_artifacts(
        cls,
        model_path: str,
        meta_path: str | None = None,
        scaler_path: str | None = None,
        device: str | torch.device = "cpu",
        signal_name: str = "price",
    ):
        """从单个产物三件套（模型/元数据/标准化器）构建单信号预测器。

        参数:
            model_path: 模型权重文件路径。
            meta_path: 元数据文件路径，为 None 时自动推导。
            scaler_path: 标准化器文件路径，为 None 时自动推导。
            device: 推理设备，默认 ``"cpu"``。
            signal_name: 信号名称备选值，元数据中有时优先使用元数据中的名称。

        返回:
            已初始化的 LSTMForecaster 实例。
        """
        meta, scaler = load_lstm_forecaster_artifacts(
            model_path=model_path,
            meta_path=meta_path,
            scaler_path=scaler_path,
        )
        return cls(
            model_path=model_path,
            hidden_size=int(meta["hidden_size"]),
            num_layers=int(meta["num_layers"]),
            dropout=float(meta["dropout"]),
            pred_len=int(meta["pred_len"]),
            seq_len=int(meta["seq_len"]),
            device=device,
            scaler=scaler,
            signal_runtimes=None,
        ).rename_default_signal(meta.get("signal_name", signal_name))

    @classmethod
    def from_signal_artifacts(
        cls,
        signal_artifacts: dict[str, tuple[str, str | None, str | None]],
        *,
        device: str | torch.device = "cpu",
    ):
        """从多组产物三件套构建多信号运行时预测器。

        参数:
            signal_artifacts: 信号名到 ``(model_path, meta_path, scaler_path)`` 的映射。
                meta_path 和 scaler_path 可为 None（自动推导）。
            device: 推理设备，默认 ``"cpu"``。

        返回:
            包含所有信号运行时的 LSTMForecaster 实例。
        """
        device = torch.device(device)
        signal_runtimes: dict[str, _SignalForecasterRuntime] = {}
        # 逐个信号加载产物并构建运行时
        for signal_name, (model_path, meta_path, scaler_path) in signal_artifacts.items():
            meta, scaler = load_lstm_forecaster_artifacts(
                model_path=model_path,
                meta_path=meta_path,
                scaler_path=scaler_path,
            )
            runtime = cls._build_runtime(
                signal_name=meta.get("signal_name", signal_name),
                model_path=model_path,
                hidden_size=int(meta["hidden_size"]),
                num_layers=int(meta["num_layers"]),
                dropout=float(meta["dropout"]),
                pred_len=int(meta["pred_len"]),
                seq_len=int(meta["seq_len"]),
                scaler=scaler,
                device=device,
            )
            signal_runtimes[str(signal_name)] = runtime

        return cls(device=device, signal_runtimes=signal_runtimes)

    def rename_default_signal(self, signal_name: str):
        """重命名兼容层默认信号（从 ``"price"`` 改为指定名称），用于加载旧产物后的适配。

        参数:
            signal_name: 新的信号名称。

        返回:
            self（支持链式调用）。
        """
        if "price" in self.signal_runtimes and signal_name != "price":
            self.signal_runtimes[str(signal_name)] = self.signal_runtimes.pop("price")
            self.signal_runtimes[str(signal_name)].signal_name = str(signal_name)
        self._set_legacy_attributes()
        return self

    def available_signals(self) -> list[str]:
        """返回所有已加载产物支持的信号名称列表（排序后）。"""
        return sorted(self.signal_runtimes)

    def reset(self) -> None:
        """重置回合状态（运行时预测器跨回合无状态，此处为空操作）。"""
        return None

    def _predict_univariate(
        self,
        runtime: _SignalForecasterRuntime,
        history: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """对单变量序列执行滚动递推预测。

        当所需的预测步数超过模型单次输出长度（pred_len）时，
        将预测结果追加到历史中，循环递推直至覆盖完整 horizon。

        参数:
            runtime: 信号运行时上下文（包含模型、标准化器等）。
            history: 一维历史序列，形状 ``(T,)``。
            horizon: 总预测步数（包含当前值）。

        返回:
            形状 ``(horizon,)`` 的预测数组，第一个元素为当前值。
        """
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32).reshape(-1)
        if history.size == 0:
            return np.zeros((horizon,), dtype=np.float32)

        # 提取当前时刻的值
        current_value = np.array([history[-1]], dtype=np.float32)
        if horizon == 1:
            return current_value.copy()

        # 滚动递推预测：每次用模型预测 pred_len 步，追加到历史后继续
        rolling_history = history.copy()
        future_chunks = []
        remaining = horizon - 1

        while remaining > 0:
            # 历史不足 seq_len 时用零左填充
            if rolling_history.size < runtime.seq_len:
                pad = np.zeros((runtime.seq_len - rolling_history.size,), dtype=np.float32)
                model_input = np.concatenate([pad, rolling_history], axis=0)
            else:
                model_input = rolling_history[-runtime.seq_len :]

            # 输入标准化
            if runtime.scaler is not None:
                model_input = runtime.scaler.transform(model_input.reshape(-1, 1)).reshape(-1).astype(np.float32)

            # 转为张量并前向推理
            model_tensor = torch.tensor(
                model_input,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)  # 添加 batch 维度

            with torch.no_grad():
                prediction = runtime.model(model_tensor).detach().cpu().numpy().reshape(-1)

            # 输出反标准化
            if runtime.scaler is not None:
                prediction = runtime.scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)

            prediction = np.asarray(prediction, dtype=np.float32)
            # 只取本轮需要的步数
            take = min(runtime.pred_len, remaining)
            prediction = prediction[:take].astype(np.float32)
            future_chunks.append(prediction)
            # 将预测结果追加到滚动历史中供下一轮使用
            rolling_history = np.concatenate([rolling_history, prediction], axis=0)
            remaining -= take

        # 拼接当前值和所有未来预测块
        future = np.concatenate(future_chunks, axis=0).astype(np.float32)
        return np.concatenate([current_value, future], axis=0)[:horizon].astype(np.float32)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        """对指定信号执行多步前瞻预测。

        一维输入直接预测；二维输入按列（每个智能体）分别预测后堆叠。
        当只有一个信号运行时且请求的信号名不匹配时，自动回退到唯一可用的信号。

        参数:
            history: 历史信号，一维 ``(T,)`` 或二维 ``(T, N)``。
            horizon: 预测步数（包含当前值）。
            signal_name: 信号名称，默认 ``"price"``。

        返回:
            共享信号返回 ``(horizon,)``，每智能体信号返回 ``(N, horizon)``。
        """
        # 信号名不匹配时的回退逻辑
        if signal_name not in self.signal_runtimes:
            if len(self.signal_runtimes) == 1:
                signal_name = next(iter(self.signal_runtimes))
            else:
                available = self.available_signals()
                raise KeyError(f"LSTMForecaster has no runtime model for '{signal_name}'. Available: {available}")

        # 同步兼容层属性的修改
        self._sync_legacy_price_runtime()
        runtime = self.signal_runtimes[signal_name]
        history = np.asarray(history, dtype=np.float32)

        # 一维输入：直接单变量预测
        if history.ndim == 1:
            return self._predict_univariate(runtime, history, horizon)

        if history.ndim != 2:
            raise ValueError(f"LSTMForecaster expects 1D or 2D history, got shape {history.shape}")

        # 二维输入：按列（每个智能体）分别预测后堆叠
        predictions = [
            self._predict_univariate(runtime, history[:, column_idx], horizon)
            for column_idx in range(history.shape[1])
        ]
        return np.stack(predictions, axis=0).astype(np.float32)
