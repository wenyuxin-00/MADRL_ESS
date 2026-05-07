from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from configs.cfg import Cfg
from utils.records import build_compare_economic_table, build_compare_safety_table, compare_rollout_metrics, load_record, plot_aligned_records
from utils.run_artifacts import load_eval_result, save_compare_summary

COMPARE_SCHEMES = (("perfect", "MISOCP"), ("perfect", "LOCAL_MPC"), ("lstm", "LOCAL_MPC"), ("lstm", "ADMM_MPC"), ("lstm", "MADRL_BASE"), ("lstm", "MADRL_PENALTY"), ("lstm", "MADRL_PROJECTION"))

def compare_all(cfg: Cfg, run_dir: Path, forecast_modes: tuple[str, ...] | None = None, controller_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    rows, results = [], []
    modes = set(cfg.eval.forecast_modes if forecast_modes is None else forecast_modes)
    names = None if controller_names is None else set(controller_names)
    schemes = tuple((mode, name) for mode, name in COMPARE_SCHEMES if mode in modes and (names is None or name in names))
    for fm, name in tqdm(schemes, desc="compare schemes", unit="scheme", ascii=True):
        result = load_eval_result(run_dir, cfg, fm, name)
        results.append(result); rows.append({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
    df = pd.DataFrame(rows); save_compare_summary(run_dir, df); _plot_all(Path(run_dir), df, results); return df


def _plot_all(run_dir: Path, df: pd.DataFrame, results: list[dict]) -> None:
    import matplotlib.pyplot as plt
    fig_dir = run_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3)); df.pivot(index="controller", columns="forecast_mode", values="storage_profit_eur_mean").plot(kind="bar", ax=ax); fig.tight_layout(); fig.savefig(fig_dir / "metrics_bar.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    for item in results: ax.plot(np.asarray(item["voltage_trace"]).reshape(-1), label=f"{item['controller']}/{item['forecast_mode']}", alpha=0.7)
    ax.legend(fontsize=7); ax.set_ylabel("vm pu"); fig.tight_layout(); fig.savefig(fig_dir / "voltage_envelope.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    for item in results: ax.plot(np.asarray(item["trafo_trace"]).reshape(-1), label=f"{item['controller']}/{item['forecast_mode']}", alpha=0.7)
    ax.legend(fontsize=7); ax.set_ylabel("trafo %"); fig.tight_layout(); fig.savefig(fig_dir / "trafo_loading.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    curves = pd.read_csv(run_dir / "tables" / "learning_curves.csv")
    for algo, group in curves.groupby("algo"): ax.plot(group["episode"], group["reward"], label=algo)
    ax.legend(fontsize=7)
    ax.set_ylabel("episode reward"); fig.tight_layout(); fig.savefig(fig_dir / "reward_curves.png", dpi=140); plt.close(fig)


def compare_records(cfg: Cfg, run_dir: Path) -> dict[str, object]:
    records = [load_record(run_dir, mode, controller) for mode, controller in COMPARE_SCHEMES]
    rollouts = [record["rollout"] for record in records]
    metrics_df = compare_rollout_metrics(*rollouts)
    economic_table = build_compare_economic_table(metrics_df)
    safety_table = build_compare_safety_table(metrics_df)
    table_dir = Path(run_dir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(table_dir / "compare_metrics.csv", index=False)
    economic_table.to_csv(table_dir / "economic_table.csv", index=False)
    safety_table.to_csv(table_dir / "safety_table.csv", index=False)
    with pd.ExcelWriter(table_dir / "compare.xlsx", engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="compare_metrics", index=False)
        economic_table.to_excel(writer, sheet_name="economic_table", index=False)
        safety_table.to_excel(writer, sheet_name="safety_table", index=False)
        for (mode, controller), record in zip(COMPARE_SCHEMES, records, strict=True):
            sheet = f"{mode}_{controller}"[:31]
            record["metrics_df"].to_excel(writer, sheet_name=sheet, index=False)
    figures = plot_aligned_records({f"{mode}_{controller}": rollout for (mode, controller), rollout in zip(COMPARE_SCHEMES, rollouts, strict=True)}, run_dir, "compare")
    return {"metrics_df": metrics_df, "economic_table": economic_table, "safety_table": safety_table, "figures": figures}
