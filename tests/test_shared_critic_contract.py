from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import torch

from configs.cfg import Cfg
from data.share_data import ShareData
from models.assembly import SharedTwinCritic
from scripts.compare import COMPARE_SCHEMES
from scripts.madrl import SCHEMES, SHARED_CRITIC_SPEC, _train_shared_actor_step, plot_madrl_outputs, run_madrl_shared_critic_experiment, train_madrl_shared_critic_scheme


def _small_cfg() -> Cfg:
    cfg = Cfg()
    return replace(
        cfg,
        env=replace(cfg.env, episode_steps=4, train_window_days=1),
        obs=replace(cfg.obs, sequence_length=3),
        train=replace(cfg.train, train_episodes=1, num_envs=1, batch_size=1, learning_starts=1, actor_learning_starts=1, n_step_return=1),
        eval=replace(cfg.eval, n_episodes=1, episode_indices=(0,)),
        model=replace(cfg.model, hidden_dim=16),
        runtime=replace(cfg.runtime, device="cpu"),
    )


def _obs(cfg: Cfg, batch: int = 2) -> dict[str, torch.Tensor]:
    n, s = int(cfg.env.num_agents), int(cfg.obs.sequence_length)
    return {
        "madrl_local": torch.zeros(batch, n, 7),
        "safety_local": torch.tensor([[[0.5, 0.2, 1.0, 2.0, 1.0, 100.0, 50.0, 60.0, 11.0]] * n] * batch, dtype=torch.float32),
        "wholesale_price_relative_seq": torch.zeros(batch, s),
        "wholesale_price_spread_seq": torch.zeros(batch, s),
        "load_seq": torch.zeros(batch, n, s),
        "pv_seq": torch.zeros(batch, n, s),
    }


def _share_data(cfg: Cfg, root: Path) -> ShareData:
    n, steps, horizon = int(cfg.env.num_agents), int(cfg.env.episode_steps), int(cfg.obs.sequence_length)
    timestamps = np.asarray([[f"2020-01-01T{hour:02d}:00:00" for hour in range(steps)]])
    price = np.full((1, steps), 0.20, dtype=np.float32)
    load = np.full((1, steps, n), 3.0, dtype=np.float32)
    pv = np.full((1, steps), 0.4, dtype=np.float32)
    price_seq = np.full((1, steps, horizon), 0.20, dtype=np.float32)
    load_seq = np.full((1, steps, horizon, n), 3.0, dtype=np.float32)
    pv_seq = np.full((1, steps, horizon), 0.4, dtype=np.float32)
    split = {
        "timestamps": timestamps, "price": price, "load": load, "pv": pv,
        "perfect_price_seq": price_seq, "perfect_load_seq": load_seq, "perfect_pv_seq": pv_seq,
        "lstm_price_seq": price_seq, "lstm_load_seq": load_seq, "lstm_pv_seq": pv_seq,
    }
    manifest = {"sequence_length": horizon, "num_agents": n, "episode_steps": steps}
    return ShareData(root=root, manifest=manifest, train=split, eval=split)


def test_shared_twin_critic_returns_per_agent_twin_q() -> None:
    cfg = _small_cfg()
    critic = SharedTwinCritic(cfg)
    action = torch.zeros(2, int(cfg.env.num_agents), int(cfg.model.action_dim))
    q1, q2 = critic(_obs(cfg), action)
    assert q1.shape == (2, int(cfg.env.num_agents))
    assert q2.shape == (2, int(cfg.env.num_agents))


def test_shared_actor_step_reads_the_matching_agent_q_column() -> None:
    source = inspect.getsource(_train_shared_actor_step)
    assert "q1_policy[:, idx]" in source


def test_shared_critic_is_not_in_mainline_scheme_lists() -> None:
    assert SHARED_CRITIC_SPEC["scheme"] not in {item["scheme"] for item in SCHEMES}
    assert ("lstm", SHARED_CRITIC_SPEC["controller"]) not in COMPARE_SCHEMES


def test_shared_smoke_writes_research_artifacts_without_model_checkpoint(tmp_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    cfg = _small_cfg()
    result = run_madrl_shared_critic_experiment(cfg, tmp_path, _share_data(cfg, tmp_path), episodes=1)
    figures = plot_madrl_outputs(result, tmp_path)
    summary = pd.read_csv(tmp_path / "tables" / "madrl_train_summary_madrl_base_shared_critic.csv")
    assert int(summary.loc[0, "train_episodes"]) == 1
    assert str(summary.loc[0, "critic_arch"]) == "shared_twin_backbone_per_agent_heads"
    assert not (tmp_path / "models" / "madrl" / "madrl_base_shared_critic" / "model.pt").exists()
    assert (tmp_path / "results" / "lstm" / "MADRL_BASE_SHARED_CRITIC" / "record" / "manifest.json").exists()
    assert (tmp_path / "figures" / "madrl_base_shared_critic_learning_curve.png").exists()
    for fig in figures.values():
        fig.clf()


def test_shared_training_path_has_no_checkpoint_or_fallback_branches() -> None:
    source = (inspect.getsource(train_madrl_shared_critic_scheme) + inspect.getsource(run_madrl_shared_critic_experiment)).lower()
    for token in ("checkpoint", "resume", "latest", "fallback", "model.pt"):
        assert token not in source
