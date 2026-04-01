"""项目路径工具。

提供项目根目录、数据目录等路径发现函数。

主要函数:
    project_root -- 返回项目根目录 Path
    get_data_root -- 返回数据目录 Path
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def _resolve_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else project_root()


def get_data_root(root: str | Path | None = None) -> Path:
    """Return the canonical data directory."""
    return _resolve_root(root) / "data"


def get_notebook_root(root: str | Path | None = None) -> Path:
    """Return the canonical notebook directory."""
    return _resolve_root(root) / "notebooks"


def get_forecast_lstm_notebook_path(root: str | Path | None = None) -> Path:
    """Return the canonical managed forecast notebook path."""
    return get_notebook_root(root) / "forecast" / "forecast_lstm.ipynb"


def get_artifact_root(root: str | Path | None = None) -> Path:
    """Return the canonical artifact directory."""
    return _resolve_root(root) / "artifacts"


def get_forecast_artifact_root(root: str | Path | None = None) -> Path:
    """Return the artifact directory used by forecast experiments."""
    return get_artifact_root(root) / "forecast"


def get_training_artifact_root(root: str | Path | None = None) -> Path:
    """Return the artifact directory used by training runs."""
    return get_artifact_root(root) / "training"


def get_shared_data_root(root: str | Path | None = None) -> Path:
    """Return the root directory that stores shared MADRL data packages."""
    return get_training_artifact_root(root) / "shared_data"


def get_checkpoint_root(root: str | Path | None = None) -> Path:
    """Return the root directory that stores training checkpoints."""
    return get_training_artifact_root(root) / "checkpoints"


def get_tensorboard_root(root: str | Path | None = None) -> Path:
    """Return the root directory that stores TensorBoard logs."""
    return get_training_artifact_root(root) / "tensorboard"


def get_tensorboard_run_dir(
    algorithm: str,
    env_name: str,
    run_number: int,
    seed: int,
    root: str | Path | None = None,
) -> Path:
    """Return the default log directory for one training run."""
    run_name = f"{algorithm}_{env_name}_{run_number}_seed_{seed}"
    return get_tensorboard_root(root) / run_name
