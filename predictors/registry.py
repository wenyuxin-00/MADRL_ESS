from __future__ import annotations
from pathlib import Path
from predictors.artifacts import get_default_lstm_artifact_dir
from predictors.lstm_forecaster import LSTMForecaster
from predictors.oracle import PerfectForecaster
from predictors.training import collect_available_lstm_artifacts, ensure_lstm_artifacts, required_forecast_signals
def build_forecaster(cfg):
    forecast_cfg = cfg.forecast
    forecaster_type = str(forecast_cfg.type).strip().lower()
    if forecaster_type == 'perfect':
        return PerfectForecaster()
    if forecaster_type != 'lstm':
        raise ValueError(f"Unknown forecaster_type '{forecaster_type}', available: ['perfect', 'lstm']")
    ensure_result = ensure_lstm_artifacts(cfg, device=cfg.runtime.device)
    artifact_map = dict(ensure_result.get('artifacts') or collect_available_lstm_artifacts(cfg))
    if not artifact_map:
        artifact_root = Path(forecast_cfg.lstm_artifact_root or get_default_lstm_artifact_dir())
        raise FileNotFoundError(f'No managed LSTM forecast artifacts are available. Checked root: {artifact_root} for future_horizon={cfg.env.future_horizon}.')
    active_signals = required_forecast_signals(cfg)
    missing_required = [signal_name for signal_name in active_signals if signal_name not in artifact_map]
    if missing_required:
        raise FileNotFoundError(f'Missing required LSTM forecast artifacts for observation signals: {missing_required}. Available managed signals: {sorted(artifact_map)}.')
    if ensure_result.get('trained_signals'):
        print(f"[forecast] trained new artifacts for signals: {', '.join(ensure_result['trained_signals'])}")
    if ensure_result.get('retrained_signals'):
        print(f"[forecast] refreshed incompatible artifacts for signals: {', '.join(ensure_result['retrained_signals'])}")
    return LSTMForecaster.from_signal_artifacts({signal_name: artifact_map[signal_name] for signal_name in active_signals}, device=cfg.runtime.device)
