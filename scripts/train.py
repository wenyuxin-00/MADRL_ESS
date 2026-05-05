from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.cfg import Cfg
from data.share_data import ShareData
from scripts.madrl import SCHEMES, train_madrl_scheme


def train(cfg: Cfg, run_dir: Path, share_data: ShareData) -> Path:
    return Path(train_madrl_scheme(cfg, run_dir, share_data, SCHEMES[1])["model_path"])


def train_all(cfg: Cfg, run_dir: Path, share_data: ShareData, variants: tuple[str, ...] = ("MADRL_BASE", "MADRL_PENALTY", "MADRL_PROJECTION"), episodes: int | None = None) -> dict[str, Path]:
    by_controller = {str(item["controller"]): item for item in SCHEMES}
    models = {name: Path(train_madrl_scheme(cfg, run_dir, share_data, by_controller[name], episodes=episodes)["model_path"]) for name in variants}
    table_dir = Path(run_dir) / "tables"
    curves = [pd.read_csv(table_dir / f"learning_curve_{by_controller[name]['scheme']}.csv") for name in variants]
    pd.concat(curves, ignore_index=True).to_csv(table_dir / "learning_curves.csv", index=False)
    return models
