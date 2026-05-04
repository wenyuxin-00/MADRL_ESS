from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, get_type_hints
import numpy as np

from REMAKE.configs.cfg import Cfg
from REMAKE.utils.paths import RUNS_DIR


def _json_default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _construct_dataclass(cls: type, payload: dict[str, Any]):
    names = {field.name: field for field in fields(cls)}
    hints = get_type_hints(cls)
    values = {}
    for name, value in payload.items():
        target = hints.get(name, names[name].type)
        values[name] = _construct_dataclass(target, value) if is_dataclass(target) and isinstance(value, dict) else value
    return cls(**values)


def write_config_json(cfg: Cfg, run_dir: Path) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def create_run_dir(cfg: Cfg, run_id: str | None = None) -> Path:
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{cfg.hash8()}"
    run_dir = RUNS_DIR / run_id
    for child in ("models", "forecast/artifacts", "forecast/tables", "forecast/figures", "share_data", "results", "tables", "figures"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    write_config_json(cfg, run_dir)
    (run_dir / "run.json").write_text(json.dumps({"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(), "cfg_hash": cfg.hash8()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def load_experiment_context(run_dir: str | Path) -> tuple[Cfg, Path]:
    path = Path(run_dir)
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return _construct_dataclass(Cfg, payload), path


def append_learning_curve(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    path = Path(run_dir) / "tables" / "learning_curves.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def save_eval_result(run_dir: Path, result: dict[str, Any]) -> None:
    out_dir = Path(run_dir) / "results" / str(result["forecast_mode"]) / str(result["controller"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scalars = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    traces = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    (out_dir / "metrics.json").write_text(json.dumps(scalars, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    np.savez_compressed(out_dir / "traces.npz", **traces)


def eval_result_exists(run_dir: Path, forecast_mode: str, controller: str) -> bool:
    base = Path(run_dir) / "results" / forecast_mode / controller
    return (base / "metrics.json").exists() and (base / "traces.npz").exists()


def load_eval_result(run_dir: Path, cfg: Cfg, forecast_mode: str, controller: str) -> dict[str, Any]:
    base = Path(run_dir) / "results" / forecast_mode / controller
    metrics_path, traces_path = base / "metrics.json", base / "traces.npz"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    traces = dict(np.load(traces_path))
    return {**metrics, **traces}


def load_eval_metrics(run_dir: Path, cfg: Cfg, forecast_mode: str, controller: str) -> dict[str, Any]:
    return {k: v for k, v in load_eval_result(run_dir, cfg, forecast_mode, controller).items() if not isinstance(v, np.ndarray)}


def save_eval_summary(run_dir: Path, df) -> None:
    path = Path(run_dir) / "tables" / "eval_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_compare_summary(run_dir: Path, df) -> None:
    path = Path(run_dir) / "tables" / "compare_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
