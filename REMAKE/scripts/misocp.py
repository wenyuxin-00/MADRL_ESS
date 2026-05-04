from __future__ import annotations

from pathlib import Path
from typing import Any

from REMAKE.configs.cfg import Cfg
from REMAKE.controllers.mpc_global import MisocpController
from REMAKE.data.share_data import ShareData
from REMAKE.utils.records import collect_rollout, load_record, plot_aligned_records, save_rollout


def run_global_misocp(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    rollout = collect_rollout(cfg, MisocpController(cfg), share_data, forecast_mode="perfect", label="Global MISOCP")
    saved = save_rollout(cfg, run_dir, rollout, scheme_name="global_misocp")
    return {"rollout": saved["rollout"], "metrics_df": saved["metrics_df"], "record_dir": saved["record_dir"], "manifest": saved["manifest"]}


def load_misocp_record(run_dir: str | Path) -> dict[str, Any]:
    return load_record(run_dir, "perfect", "MISOCP")


def plot_misocp_outputs(record: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    return plot_aligned_records({"global_misocp": record["rollout"]}, run_dir, "misocp")


def _episode_start_timestamps(share_data: ShareData) -> list[str]:
    return [str(value) for value in share_data.eval["timestamps"][:, 0]]
