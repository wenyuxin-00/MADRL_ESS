from __future__ import annotations
from pathlib import Path
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _resolve_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else project_root()

def get_data_root(root: str | Path | None = None) -> Path:
    return _resolve_root(root) / "data"

def get_notebook_root(root: str | Path | None = None) -> Path:
    return _resolve_root(root) / "notebooks"

def get_forecast_lstm_notebook_path(root: str | Path | None = None) -> Path:
    return get_notebook_root(root) / "forecast" / "forecast_lstm.ipynb"

def get_artifact_root(root: str | Path | None = None) -> Path:
    return _resolve_root(root) / "artifacts"

def get_forecast_artifact_root(root: str | Path | None = None) -> Path:
    return get_artifact_root(root) / "forecast"

def get_training_artifact_root(root: str | Path | None = None) -> Path:
    return get_artifact_root(root) / "training"

def get_shared_data_root(root: str | Path | None = None) -> Path:
    return get_training_artifact_root(root) / "shared_data"

def get_checkpoint_root(root: str | Path | None = None) -> Path:
    return get_training_artifact_root(root) / "checkpoints"

def get_tensorboard_root(root: str | Path | None = None) -> Path:
    return get_training_artifact_root(root) / "tensorboard"

def get_tensorboard_run_dir(
    algorithm: str,
    env_name: str,
    run_number: int,
    seed: int,
    root: str | Path | None = None,
) -> Path:
    run_name = f"{algorithm}_{env_name}_{run_number}_seed_{seed}"
    return get_tensorboard_root(root) / run_name
