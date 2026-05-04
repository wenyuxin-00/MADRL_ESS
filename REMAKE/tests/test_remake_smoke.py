from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from REMAKE.configs.cfg import Cfg
from REMAKE.controllers.protocol import CONTROLLER_NAMES
from REMAKE.data.share_data import build_share_data, load_share_data
from REMAKE.predictors.lstm_training import train_forecasters
from REMAKE.scripts.compare import compare_all
from REMAKE.scripts.eval import eval_all
from REMAKE.scripts.train import train_all
from REMAKE.utils.run_artifacts import write_config_json


def small_cfg() -> Cfg:
    cfg = Cfg()
    return replace(
        cfg,
        data=replace(cfg.data, train_start_date="2019-04-01", train_end_date="2019-04-02", eval_start_date="2020-04-01", eval_end_date="2020-04-01"),
        env=replace(cfg.env, episode_steps=12),
        obs=replace(cfg.obs, sequence_length=4),
        forecast=replace(cfg.forecast, history_window=24, lstm_batch_size=64, price_lstm_epochs=1, load_lstm_epochs=1, pv_lstm_epochs=1, price_lstm_hidden_size=8, load_lstm_hidden_size=8, pv_lstm_hidden_size=8, price_lstm_num_layers=1, load_lstm_num_layers=1, pv_lstm_num_layers=1),
        train=replace(cfg.train, train_episodes=1),
        eval=replace(cfg.eval, n_episodes=1, forecast_modes=("perfect", "lstm")),
        mpc=replace(cfg.mpc, admm_max_iter=2),
    )


def test_remake_end_to_end_writes_cached_results(tmp_path: Path) -> None:
    cfg, run_dir = small_cfg(), tmp_path / "run"
    write_config_json(cfg, run_dir)
    forecast = train_forecasters(cfg, run_dir, overwrite=True)
    cfg = replace(cfg, forecast=replace(cfg.forecast, lstm_artifact_dir=str(forecast["artifact_dir"])))
    write_config_json(cfg, run_dir)
    share_dir = build_share_data(cfg, run_dir, forecast["artifact_dir"], overwrite=True)
    share_data = load_share_data(share_dir, cfg)
    assert share_data.eval["pv"].ndim == 2
    assert share_data.eval["lstm_pv_seq"].ndim == 3
    ckpts = train_all(cfg, run_dir, share_data, episodes=1)
    eval_df = eval_all(cfg, ckpts, run_dir, share_data=share_data, overwrite=True)
    assert eval_df.shape[0] == len(CONTROLLER_NAMES) * len(cfg.eval.forecast_modes)
    compare_df = compare_all(cfg, run_dir)
    assert compare_df.shape[0] == 7
    for name in ("compare_summary.csv", "eval_summary.csv", "learning_curves.csv"):
        assert (run_dir / "tables" / name).exists()
    assert (run_dir / "results" / "perfect" / "MISOCP" / "metrics.json").exists()


def test_compare_requires_cached_results(tmp_path: Path) -> None:
    cfg = small_cfg()
    with pytest.raises(FileNotFoundError):
        compare_all(cfg, tmp_path)


def test_remake_non_test_python_loc_under_4000() -> None:
    root = Path("REMAKE")
    total = 0
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))
    assert total <= 4000
