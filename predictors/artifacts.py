from __future__ import annotations
from pathlib import Path
from scripts.utils.project_paths import get_forecast_artifact_root
DEFAULT_LSTM_ARTIFACT_DIR = Path('lstm')
DEFAULT_SUPPORTED_FORECAST_SIGNALS = ('wholesale_price', 'load', 'pv')
def get_default_lstm_artifact_dir(root: str | Path | None=None) -> Path:
    return Path(root) if root is not None else get_forecast_artifact_root(root) / DEFAULT_LSTM_ARTIFACT_DIR
def get_default_lstm_artifact_paths(root: str | Path | None=None, *, signal_name: str='wholesale_price', future_horizon: int=24, agent_index: int | None=None, component: str | None=None) -> dict[str, Path]:
    signal_name = str(signal_name)
    future_horizon = int(future_horizon)
    artifact_dir = get_default_lstm_artifact_dir(root) / f'h{future_horizon}' / signal_name
    stem = f'{signal_name}_lstm_h{future_horizon}'
    if agent_index is not None:
        agent_index = int(agent_index)
        artifact_dir = artifact_dir / f'agent_{agent_index}'
        stem = f'{signal_name}_agent{agent_index}_lstm_h{future_horizon}'
    if component is not None:
        component = str(component)
        artifact_dir = artifact_dir / component
        stem = f'{stem}_{component}'
    return {'artifact_dir': artifact_dir, 'model_path': artifact_dir / f'{stem}.pt', 'meta_path': artifact_dir / f'{stem}_meta.json', 'scaler_path': artifact_dir / f'{stem}_scaler.pkl'}
