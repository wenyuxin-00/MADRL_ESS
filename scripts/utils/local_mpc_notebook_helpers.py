"""Cached rollout helpers for the local_MPC notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils import local_mpc_rollout_packages as local_rollout_packages


DEFAULT_LOCAL_MPC_PRICE_MODE = local_rollout_packages.DEFAULT_LOCAL_MPC_PRICE_MODE
DEFAULT_LOCAL_MPC_OBJECTIVE_MODE = local_rollout_packages.DEFAULT_LOCAL_MPC_OBJECTIVE_MODE
LOCAL_MPC_PERFECT_LABEL = "Local MPC + Perfect Forecast"
LOCAL_MPC_LSTM_LABEL = "Local MPC + LSTM Forecast"


def build_local_mpc_rollout_package(
    rollout: grid_nb.RolloutResult,
    *,
    controller_label: str,
    cfg: Any | None = None,
    cfg_snapshot: dict[str, Any] | None = None,
    prediction_mode: str | None = None,
    price_mode: str = DEFAULT_LOCAL_MPC_PRICE_MODE,
    objective_mode: str = DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
    extra_meta: dict[str, Any] | None = None,
    diagnostic_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return local_rollout_packages.build_local_mpc_rollout_package(
        rollout,
        controller_label=controller_label,
        cfg=cfg,
        cfg_snapshot=cfg_snapshot,
        prediction_mode=prediction_mode,
        price_mode=price_mode,
        objective_mode=objective_mode,
        extra_meta=extra_meta,
        diagnostic_summary=diagnostic_summary,
    )


def save_local_mpc_rollout_package(package: dict[str, Any], target_dir: str | Path) -> Path:
    return local_rollout_packages.save_local_mpc_rollout_package(package, target_dir)


def load_local_mpc_rollout_package(target_dir: str | Path) -> dict[str, Any]:
    return local_rollout_packages.load_local_mpc_rollout_package(target_dir)


def resolve_latest_compatible_local_mpc_rollout_package_dir(
    target_dir: str | Path,
    *,
    cfg: Any | None = None,
    prediction_mode: str | None = None,
    price_mode: str = DEFAULT_LOCAL_MPC_PRICE_MODE,
    objective_mode: str = DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
) -> Path:
    return local_rollout_packages.resolve_latest_compatible_local_mpc_rollout_package_dir(
        target_dir,
        cfg=cfg,
        prediction_mode=prediction_mode,
        price_mode=price_mode,
        objective_mode=objective_mode,
    )


def replay_local_mpc_rollout_package(
    cfg: Any,
    target_dir: str | Path,
    *,
    label: str | None = None,
    prediction_mode: str | None = None,
) -> dict[str, Any]:
    return local_rollout_packages.replay_local_mpc_rollout_package(
        cfg,
        target_dir,
        label=label,
        prediction_mode=prediction_mode,
    )


__all__ = [
    "DEFAULT_LOCAL_MPC_OBJECTIVE_MODE",
    "DEFAULT_LOCAL_MPC_PRICE_MODE",
    "LOCAL_MPC_LSTM_LABEL",
    "LOCAL_MPC_PERFECT_LABEL",
    "build_local_mpc_rollout_package",
    "load_local_mpc_rollout_package",
    "replay_local_mpc_rollout_package",
    "resolve_latest_compatible_local_mpc_rollout_package_dir",
    "save_local_mpc_rollout_package",
]
