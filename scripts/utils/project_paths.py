from __future__ import annotations
from pathlib import Path
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _resolve_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else project_root()

def _artifact_subdir(*parts: str, root: str | Path | None = None) -> Path:
    return _resolve_root(root).joinpath("artifacts", *parts)

def get_data_root(root: str | Path | None = None) -> Path:
    return _resolve_root(root) / "data"

def get_notebook_root(root: str | Path | None = None) -> Path:
    return _resolve_root(root) / "notebooks"

def get_forecast_lstm_notebook_path(root: str | Path | None = None) -> Path:
    return get_notebook_root(root) / "forecast" / "forecast_lstm.ipynb"

def get_artifact_root(root: str | Path | None = None) -> Path:
    return _artifact_subdir(root=root)

def get_forecast_artifact_root(root: str | Path | None = None) -> Path:
    return _artifact_subdir("forecast", root=root)

def get_training_artifact_root(root: str | Path | None = None) -> Path:
    return _artifact_subdir("training", root=root)

def get_shared_data_root(root: str | Path | None = None) -> Path:
    return _artifact_subdir("training", "shared_data", root=root)

def get_checkpoint_root(root: str | Path | None = None) -> Path:
    return _artifact_subdir("training", "checkpoints", root=root)

def get_tensorboard_root(root: str | Path | None = None) -> Path:
    return _artifact_subdir("training", "tensorboard", root=root)

def get_tensorboard_run_dir(
    algorithm: str,
    env_name: str,
    run_number: int,
    seed: int,
    root: str | Path | None = None,
) -> Path:
    run_name = f"{algorithm}_{env_name}_{run_number}_seed_{seed}"
    return get_tensorboard_root(root) / run_name
