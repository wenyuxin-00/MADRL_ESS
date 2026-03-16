"""
forecast/registry.py
职责：预测器注册表与工厂函数。

用法：
    from forecast.registry import build_forecaster
    forecaster = build_forecaster(args)

支持的 forecaster_type：
  - "perfect" : PerfectForecaster（Oracle，直接返回真实未来价格，为默认值）
  - "naive"   : NaiveForecaster（历史滚动均值，无参数基准）
  - "lstm"    : LSTMForecaster（推荐使用标准产物：
                best_lstm.pt + best_lstm_meta.json + best_lstm_scaler.pkl）

后续扩展：注册新预测器只需在 FORECASTER_REGISTRY 中添加条目。
"""

from forecast.oracle import PerfectForecaster
from forecast.naive import NaiveForecaster
from forecast.lstm_forecaster import LSTMForecaster, resolve_lstm_artifact_paths

FORECASTER_REGISTRY: dict = {
    "perfect": PerfectForecaster,
    "naive":   NaiveForecaster,
    "lstm":    LSTMForecaster,
}


def build_forecaster(args):
    """根据 args.forecaster_type 构建预测器实例。

    Parameters
    ----------
    args : Config
        超参数对象。
        - forecaster_type: str，默认 "perfect"
        - naive_window:    int，仅 naive 需要，默认 96
        - lstm_model_path: str，仅 lstm 需要（必须指定）
        - lstm_meta_path:  str，可选；默认自动查找 <stem>_meta.json
        - lstm_scaler_path:str，可选；默认自动查找 <stem>_scaler.pkl
        - lstm_seq_len / lstm_pred_len / lstm_hidden_size / lstm_num_layers / lstm_dropout：
          当 sidecar 工件缺失时，可作为兼容兜底显式提供

    Returns
    -------
    Forecaster
    """
    ft = getattr(args, "forecaster_type", "perfect")

    if ft == "perfect":
        return PerfectForecaster()

    elif ft == "naive":
        window = int(getattr(args, "naive_window", 96))
        return NaiveForecaster(window=window)

    elif ft == "lstm":
        model_path = getattr(args, "lstm_model_path", None)
        if model_path is None:
            raise ValueError(
                "forecaster_type='lstm' requires args.lstm_model_path to point to "
                "the saved LSTM weights file."
            )
        meta_path = getattr(args, "lstm_meta_path", None)
        scaler_path = getattr(args, "lstm_scaler_path", None)
        device = getattr(args, "device", "cpu")

        _, default_meta_path, default_scaler_path = resolve_lstm_artifact_paths(
            model_path=model_path,
            meta_path=meta_path,
            scaler_path=scaler_path,
        )
        has_standard_artifacts = default_meta_path.exists() and default_scaler_path.exists()

        if meta_path is not None or scaler_path is not None or has_standard_artifacts:
            return LSTMForecaster.from_artifacts(
                model_path=str(model_path),
                meta_path=str(meta_path) if meta_path is not None else None,
                scaler_path=str(scaler_path) if scaler_path is not None else None,
                device=device,
            )

        explicit_meta = {
            "hidden_size": getattr(args, "lstm_hidden_size", None),
            "num_layers": getattr(args, "lstm_num_layers", None),
            "dropout": getattr(args, "lstm_dropout", None),
            "pred_len": getattr(args, "lstm_pred_len", None),
            "seq_len": getattr(args, "lstm_seq_len", None),
        }
        missing_explicit = [k for k, v in explicit_meta.items() if v is None]
        if missing_explicit:
            raise FileNotFoundError(
                "forecaster_type='lstm' could not find the standard sidecar artifacts "
                f"'{default_meta_path.name}' and '{default_scaler_path.name}', and these "
                f"explicit lstm_* settings are also missing: {missing_explicit}"
            )

        scaler = getattr(args, "lstm_scaler", None)
        if scaler is None:
            raise FileNotFoundError(
                "forecaster_type='lstm' requires a fitted scaler for inference. "
                f"Expected '{default_scaler_path}' or set args.lstm_scaler_path."
            )

        return LSTMForecaster(
            model_path=str(model_path),
            hidden_size=int(explicit_meta["hidden_size"]),
            num_layers=int(explicit_meta["num_layers"]),
            dropout=float(explicit_meta["dropout"]),
            pred_len=int(explicit_meta["pred_len"]),
            seq_len=int(explicit_meta["seq_len"]),
            device=device,
            scaler=scaler,
        )

    else:
        raise ValueError(
            f"未知 forecaster_type '{ft}'，可选: {list(FORECASTER_REGISTRY)}"
        )
