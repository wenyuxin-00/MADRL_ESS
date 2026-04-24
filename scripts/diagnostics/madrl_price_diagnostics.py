from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WHOLESALE_PRICE_SEQ_FIELD = "wholesale_price_seq"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "record" / "diagnostics"
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _quantile_dict(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in QUANTILES}
    quantiles = np.quantile(finite.astype(np.float64), QUANTILES)
    return {f"q{int(q * 100):02d}": float(value) for q, value in zip(QUANTILES, quantiles, strict=False)}


def _load_split_price_sequences(shared_data_dir: Path, split: str) -> tuple[dict[str, Any], np.ndarray]:
    split_dir = shared_data_dir / split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared-data split manifest '{manifest_path}'. Expected a current shared-data package "
            "from notebooks/forecast/forecast_lstm.ipynb."
        )
    manifest = _read_json(manifest_path)
    files = dict(manifest.get("files") or {})
    if WHOLESALE_PRICE_SEQ_FIELD not in files:
        raise KeyError(
            f"Old shared-data split manifest '{manifest_path}' does not contain '{WHOLESALE_PRICE_SEQ_FIELD}'. "
            "Expected the current MADRL shared-data contract. Re-run notebooks/forecast/forecast_lstm.ipynb."
        )
    array_path = split_dir / str(files[WHOLESALE_PRICE_SEQ_FIELD])
    if not array_path.exists():
        raise FileNotFoundError(
            f"Shared-data manifest '{manifest_path}' points to missing price sequence file '{array_path}'. "
            "Re-run notebooks/forecast/forecast_lstm.ipynb."
        )
    return manifest, np.asarray(np.load(array_path, mmap_mode="r"), dtype=np.float32)


def _fit_train_price_normalizer(train_values: np.ndarray) -> dict[str, float]:
    values = np.asarray(train_values, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot fit price diagnostics normalizer: train wholesale price sequence is empty.")
    q01, q25, median, q75, q99 = np.quantile(finite.astype(np.float64), [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "q_low": float(q01),
        "q_high": float(q99),
        "median": float(median),
        "iqr": float(max(q75 - q25, 1e-6)),
        "tanh_scale": 2.0,
    }


def _normalize_price(values: np.ndarray, spec: dict[str, float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    clipped = np.clip(array, float(spec["q_low"]), float(spec["q_high"]))
    z_score = (clipped - float(spec["median"])) / max(float(spec["iqr"]), 1e-6)
    return np.tanh(z_score / max(float(spec["tanh_scale"]), 1e-6)).astype(np.float32)


def _window_arbitrage_ratio(price_seq: np.ndarray, *, min_spread_eur_per_kwh: float) -> float:
    array = np.asarray(price_seq, dtype=np.float32)
    if array.size == 0:
        return float("nan")
    spread = np.nanmax(array, axis=-1) - np.nanmin(array, axis=-1)
    return float(np.mean(spread > float(min_spread_eur_per_kwh)))


def _flatten_numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros((0,), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for value in frame[column].to_numpy():
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        chunks.append(array[np.isfinite(array)])
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _rollout_metrics(path: Path | None, *, power_eps_kw: float) -> dict[str, float | int]:
    empty = {
        "requested_battery_power_kw_mean_abs": float("nan"),
        "executed_battery_power_kw_mean_abs": float("nan"),
        "controller_action_gap_mean": float("nan"),
        "action_feasibility_gap_mean": float("nan"),
        "madrl_r_action_penalty_total": 0.0,
        "madrl_r_soc_regularization_total": 0.0,
        "madrl_r_throughput_bonus_total": 0.0,
        "madrl_r_safe_total": 0.0,
        "madrl_r_inc_total": float("nan"),
        "madrl_r_total_internal_total": float("nan"),
        "storage_profit_total": float("nan"),
        "charge_steps": 0,
        "discharge_steps": 0,
        "idle_steps": 0,
    }
    if path is None:
        return empty
    if not path.exists():
        raise FileNotFoundError(f"Rollout parquet '{path}' does not exist.")
    frame = pd.read_parquet(path)
    if "r_soc_pen" in frame.columns:
        raise ValueError(
            f"Old rollout object '{path}' contains removed field 'r_soc_pen'. New contract expects projected "
            "actions with action feasibility regularization; re-run notebooks/madrl/train_base.ipynb."
        )
    if "madrl_raw_action_boundary_violation" in frame.columns:
        raise ValueError(
            f"Old rollout object '{path}' contains removed field 'madrl_raw_action_boundary_violation'. "
            "Expected madrl_action_feasibility_gap from soc_penalty_unweighted. Re-run notebooks/madrl/train_base.ipynb."
        )
    req_col = _first_existing_column(frame, ("e_bat_req", "battery_power_req_kw"))
    exec_col = _first_existing_column(frame, ("e_bat", "battery_power_exec_kw", "battery_net_power_kw"))
    gap_col = _first_existing_column(frame, ("madrl_action_projection_gap", "controller_action_gap"))
    feasibility_gap_col = _first_existing_column(frame, ("madrl_action_feasibility_gap", "soc_penalty_unweighted"))
    profit_col = _first_existing_column(frame, ("madrl_r_inc", "storage_profit_total_eur", "storage_profit_eur"))
    total_internal_col = _first_existing_column(frame, ("madrl_r_total_internal",))
    action_penalty_col = _first_existing_column(frame, ("madrl_r_action_penalty",))
    soc_regularization_col = _first_existing_column(frame, ("madrl_r_soc_regularization",))
    throughput_bonus_col = _first_existing_column(frame, ("madrl_r_throughput_bonus",))
    safe_total_col = _first_existing_column(frame, ("madrl_r_safe_total",))
    requested = np.zeros((0,), dtype=np.float32) if req_col is None else _flatten_numeric_column(frame, req_col)
    executed = np.zeros((0,), dtype=np.float32) if exec_col is None else _flatten_numeric_column(frame, exec_col)
    gaps = np.zeros((0,), dtype=np.float32) if gap_col is None else _flatten_numeric_column(frame, gap_col)
    feasibility_gaps = np.zeros((0,), dtype=np.float32) if feasibility_gap_col is None else _flatten_numeric_column(frame, feasibility_gap_col)
    profits = np.zeros((0,), dtype=np.float32) if profit_col is None else _flatten_numeric_column(frame, profit_col)
    total_internal = np.zeros((0,), dtype=np.float32) if total_internal_col is None else _flatten_numeric_column(frame, total_internal_col)
    action_penalties = np.zeros((0,), dtype=np.float32) if action_penalty_col is None else _flatten_numeric_column(frame, action_penalty_col)
    soc_regularization = np.zeros((0,), dtype=np.float32) if soc_regularization_col is None else _flatten_numeric_column(frame, soc_regularization_col)
    throughput_bonus = np.zeros((0,), dtype=np.float32) if throughput_bonus_col is None else _flatten_numeric_column(frame, throughput_bonus_col)
    safe_total = np.zeros((0,), dtype=np.float32) if safe_total_col is None else _flatten_numeric_column(frame, safe_total_col)
    charge_steps = int(np.sum(executed > float(power_eps_kw))) if executed.size else 0
    discharge_steps = int(np.sum(executed < -float(power_eps_kw))) if executed.size else 0
    idle_steps = int(np.sum(np.abs(executed) <= float(power_eps_kw))) if executed.size else 0
    return {
        "requested_battery_power_kw_mean_abs": float(np.mean(np.abs(requested))) if requested.size else float("nan"),
        "executed_battery_power_kw_mean_abs": float(np.mean(np.abs(executed))) if executed.size else float("nan"),
        "controller_action_gap_mean": float(np.mean(gaps)) if gaps.size else float("nan"),
        "action_feasibility_gap_mean": float(np.mean(feasibility_gaps)) if feasibility_gaps.size else float("nan"),
        "madrl_r_action_penalty_total": float(np.sum(action_penalties)) if action_penalties.size else 0.0,
        "madrl_r_soc_regularization_total": float(np.sum(soc_regularization)) if soc_regularization.size else 0.0,
        "madrl_r_throughput_bonus_total": float(np.sum(throughput_bonus)) if throughput_bonus.size else 0.0,
        "madrl_r_safe_total": float(np.sum(safe_total)) if safe_total.size else 0.0,
        "madrl_r_inc_total": float(np.sum(profits)) if profits.size else float("nan"),
        "madrl_r_total_internal_total": float(np.sum(total_internal)) if total_internal.size else float("nan"),
        "storage_profit_total": float(np.sum(profits)) if profits.size else float("nan"),
        "charge_steps": charge_steps,
        "discharge_steps": discharge_steps,
        "idle_steps": idle_steps,
    }


def _run_controls(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = _read_json(path)
    result = dict(payload.get("result") or payload)
    experiment_controls = dict(result.get("experiment_controls") or {})
    return result, dict(experiment_controls.get("reward_controls") or {})


def build_diagnostics(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    shared_data_dir = Path(args.shared_data_dir).resolve()
    root_manifest_path = shared_data_dir / "manifest.json"
    if not root_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared-data root manifest '{root_manifest_path}'. Re-run notebooks/forecast/forecast_lstm.ipynb."
        )
    root_manifest = _read_json(root_manifest_path)
    run_result, reward_controls = _run_controls(None if args.run_result_json is None else Path(args.run_result_json).resolve())
    _, train_seq = _load_split_price_sequences(shared_data_dir, "train")
    _, test_seq = _load_split_price_sequences(shared_data_dir, "test")
    normalizer = _fit_train_price_normalizer(train_seq)
    normalized_train = _normalize_price(train_seq, normalizer)
    normalized_test = _normalize_price(test_seq, normalizer)
    import_markup = float(
        args.import_price_markup_eur_per_kwh
        if args.import_price_markup_eur_per_kwh is not None
        else reward_controls.get("import_price_markup_eur_per_kwh", 0.20)
    )
    storage_price_mode = str(
        args.storage_price_mode
        if args.storage_price_mode is not None
        else reward_controls.get("storage_price_mode", "real_time_price")
    )
    rollout_path = None if args.rollout_step_parquet is None else Path(args.rollout_step_parquet).resolve()
    rollout = _rollout_metrics(rollout_path, power_eps_kw=float(args.power_eps_kw))
    summary: dict[str, Any] = {
        "shared_data_signature": str(
            args.shared_data_signature
            or run_result.get("shared_data_signature")
            or root_manifest.get("signature_hash")
            or ""
        ),
        "storage_price_mode": storage_price_mode,
        "import_price_markup_eur_per_kwh": import_markup,
        "raw_wholesale_train_quantiles": _quantile_dict(train_seq),
        "raw_wholesale_test_quantiles": _quantile_dict(test_seq),
        "normalized_train_quantiles": _quantile_dict(normalized_train),
        "normalized_test_quantiles": _quantile_dict(normalized_test),
        "normalized_saturation_ratio": float(np.mean(np.abs(normalized_test.reshape(-1)) >= 0.995)),
        "import_price_quantiles": _quantile_dict(test_seq + np.float32(import_markup)),
        "perfect_future_arbitrage_ratio": float("nan"),
        "lstm_window_arbitrage_ratio": _window_arbitrage_ratio(
            test_seq,
            min_spread_eur_per_kwh=float(args.min_arbitrage_spread_eur_per_kwh),
        ),
        **rollout,
    }
    parquet_row = {
        key: json.dumps(value, sort_keys=True, default=_json_default) if isinstance(value, dict) else value
        for key, value in summary.items()
    }
    return summary, pd.DataFrame([parquet_row])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MADRL price and action diagnostics.")
    parser.add_argument("--shared-data-dir", required=True, help="Exact shared-data package directory containing manifest.json.")
    parser.add_argument("--rollout-step-parquet", default=None, help="Optional exact rollout step/agent parquet to summarize.")
    parser.add_argument("--run-result-json", default=None, help="Optional train_result.json for reward controls and signature.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for madrl_price_diagnostics parquet/json outputs.")
    parser.add_argument("--shared-data-signature", default=None, help="Optional signature override.")
    parser.add_argument("--storage-price-mode", default=None, help="Optional storage price mode override.")
    parser.add_argument("--import-price-markup-eur-per-kwh", type=float, default=None, help="Optional import markup override.")
    parser.add_argument("--min-arbitrage-spread-eur-per-kwh", type=float, default=0.0)
    parser.add_argument("--power-eps-kw", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, parquet_frame = build_diagnostics(args)
    parquet_path = output_dir / "madrl_price_diagnostics.parquet"
    json_path = output_dir / "madrl_price_diagnostics.json"
    parquet_frame.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(str(json_path))
    print(str(parquet_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
