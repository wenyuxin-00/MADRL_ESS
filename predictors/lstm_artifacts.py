from __future__ import annotations

import json, pickle
from pathlib import Path
from typing import Any
import torch

from predictors.lstm_model import LSTMForecastModel


def artifact_path(root: Path, signal_name: str, agent_index: int | None = None, component: str | None = None) -> Path:
    parts = [Path(root), Path(signal_name)]
    if component is not None:
        parts.append(Path(str(component)))
    parts.append(Path("shared" if agent_index is None else f"agent_{int(agent_index)}"))
    return Path(*parts)


def validate_lstm_artifact(path: Path, *, signal_name: str, history_window: int, pred_len: int, agent_index: int | None) -> dict[str, Any]:
    base = Path(path)
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    return meta


def save_lstm_artifact(path: Path, model: LSTMForecastModel, scaler: object, meta: dict[str, Any]) -> None:
    base = Path(path); base.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base / "model.pt")
    (base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with (base / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)


def load_lstm_artifact(path: Path, *, signal_name: str, history_window: int, pred_len: int, agent_index: int | None, map_location: str | torch.device = "cuda") -> tuple[LSTMForecastModel, object, dict[str, Any]]:
    meta = validate_lstm_artifact(path, signal_name=signal_name, history_window=history_window, pred_len=pred_len, agent_index=agent_index)
    model = LSTMForecastModel(hidden_size=int(meta["hidden_size"]), num_layers=int(meta["num_layers"]), dropout=float(meta["dropout"]), pred_len=int(pred_len), input_size=int(meta["input_size"]))
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location=map_location))
    model.eval()
    with (Path(path) / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    return model, scaler, meta
