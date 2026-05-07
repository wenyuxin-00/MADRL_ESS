from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import pandas as pd

from configs.cfg import Cfg
from controllers.protocol import CONTROLLER_NAMES
from data.share_data import build_share_data, load_share_data
from predictors.lstm_training import train_forecasters
from scripts.compare import compare_all
from scripts.eval import eval_all
from scripts.madrl import REWARD_COMPONENT_COLUMNS, SCHEMES, _plot_reward_curves, train_madrl_scheme
from utils.run_artifacts import write_config_json


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
    madrl_models = {str(spec["controller"]): Path(train_madrl_scheme(cfg, run_dir, share_data, spec, episodes=1)["model_path"]) for spec in SCHEMES}
    table_dir = run_dir / "tables"
    pd.concat([pd.read_csv(table_dir / f"learning_curve_{spec['scheme']}.csv") for spec in SCHEMES], ignore_index=True).to_csv(table_dir / "learning_curves.csv", index=False)
    reward_df = pd.read_csv(run_dir / "tables" / "madrl_reward_curves_madrl_base.csv")
    assert {"total_reward", *REWARD_COMPONENT_COLUMNS} <= set(reward_df.columns)
    eval_df = eval_all(cfg, madrl_models, run_dir, share_data=share_data)
    assert eval_df.shape[0] == len(CONTROLLER_NAMES) * len(cfg.eval.forecast_modes)
    compare_df = compare_all(cfg, run_dir)
    assert compare_df.shape[0] == 7
    for name in ("compare_summary.csv", "eval_summary.csv", "learning_curves.csv"):
        assert (run_dir / "tables" / name).exists()
    assert (run_dir / "results" / "perfect" / "MISOCP" / "metrics.json").exists()


def test_madrl_reward_curve_plot_handles_components(tmp_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    component_rows = {key: [float(idx), float(idx + 1)] for idx, key in enumerate(REWARD_COMPONENT_COLUMNS)}
    rewards = pd.DataFrame({"scheme": ["madrl_base", "madrl_base"], "episode": [1, 2], "total_reward": [1.0, 2.0], **component_rows})
    fig = _plot_reward_curves(rewards, tmp_path, "madrl_base")
    assert len(fig.axes) == 2
    assert (tmp_path / "figures" / "madrl_base_learning_curve.png").exists()
    plt.close(fig)


def test_compare_requires_cached_results(tmp_path: Path) -> None:
    cfg = small_cfg()
    with pytest.raises(FileNotFoundError):
        compare_all(cfg, tmp_path)


def test_remake_non_test_python_loc_under_4000() -> None:
    root = Path(".")
    total = 0
    excluded = {".claude", ".conda", ".git", ".gitnexus", ".pytest_cache", ".venv", ".vscode", "artifacts", "node_modules", "tests", "tmp", "tools"}
    for path in root.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        if "tests" in path.parts:
            continue
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))
    assert total <= 4000
