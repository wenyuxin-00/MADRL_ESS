from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from REMAKE.configs.cfg import Cfg
from REMAKE.controllers.mpc_admm import AdmmMpcController
from REMAKE.controllers.mpc_local import LocalMPCController
from REMAKE.data.share_data import ShareData
from REMAKE.utils.records import collect_rollout, load_record, plot_aligned_records, save_rollout

LOCAL_MPC_PERFECT_LABEL = "Local MPC + Perfect Forecast"
LOCAL_MPC_LSTM_LABEL = "Local MPC + LSTM Forecast"
ADMM_MPC_LSTM_LABEL = "ADMM MPC + LSTM Forecast"


def run_local_mpc(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    records = {}
    for mode in tqdm(("perfect", "lstm"), desc="local MPC modes", unit="mode", ascii=True):
        controller = LocalMPCController(cfg)
        label = LOCAL_MPC_PERFECT_LABEL if mode == "perfect" else LOCAL_MPC_LSTM_LABEL
        rollout = collect_rollout(cfg, controller, share_data, forecast_mode=mode, label=label)
        records[mode] = save_rollout(cfg, run_dir, rollout, scheme_name=f"local_mpc_{mode}")
        controller.close()
    return records


def run_admm_mpc(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    controller = AdmmMpcController(cfg)
    rollout = collect_rollout(cfg, controller, share_data, forecast_mode="lstm", label=ADMM_MPC_LSTM_LABEL)
    return save_rollout(cfg, run_dir, rollout, scheme_name="admm_mpc_lstm")


def load_mpc_record(run_dir: str | Path, mode: str, controller: str) -> dict[str, Any]:
    return load_record(run_dir, mode, controller)


def plot_mpc_outputs(records: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    rollouts = {
        "local_perfect": records["local"]["perfect"]["rollout"],
        "local_lstm": records["local"]["lstm"]["rollout"],
        "admm_lstm": records["admm"]["rollout"],
    }
    return plot_aligned_records(rollouts, run_dir, "mpc")
