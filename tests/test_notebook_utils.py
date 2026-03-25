from pathlib import Path

from scripts.utils.experiment_notebook_utils import (
    get_lstm_artifact_root,
    prepare_madrl_run_paths,
    resolve_madrl_model_root,
    sanity_check_runner,
)
from tests.support.helpers import make_case_dir, make_smoke_config


def test_sanity_check_runner_reports_core_shapes(tmp_path):
    case_dir = make_case_dir(tmp_path, "sanity_runner")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    summary = sanity_check_runner(cfg, seed=0)

    assert "observation_schema" in summary
    assert "rollout_obs_shapes" in summary
    assert summary["action_dim"] == 1


def test_notebook_utils_import_smoke_uses_common_location(tmp_path):
    case_dir = make_case_dir(tmp_path, "notebook_utils_import")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    artifact_root = get_lstm_artifact_root(case_dir)

    assert artifact_root.name == "lstm"
    assert cfg.runtime.device.type == "cpu"


def test_prepare_madrl_run_paths_uses_checkpoint_layout(tmp_path):
    paths = prepare_madrl_run_paths(
        checkpoint_root=tmp_path / "checkpoints",
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
        train_episodes=100,
        max_train_steps=None,
    )

    model_root = Path(paths["model_root"])
    assert model_root.parts[-4:-1] == ("MATD3", "perfect", "grid_mainline")
    assert model_root.name.startswith("matd3_perfect_grid_mainline_ep100_")
    assert Path(paths["meta_dir"]).parent == model_root


def test_resolve_madrl_model_root_supports_auto_discovery_and_algo_dir(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    run_root = checkpoint_root / "MATD3" / "perfect" / "grid_mainline" / "matd3_perfect_grid_mainline_ep10_20260325_120000"
    algo_dir = run_root / "MATD3"
    algo_dir.mkdir(parents=True, exist_ok=True)
    (algo_dir / "actor_agent_0_ep_10.pth").write_bytes(b"actor")
    (algo_dir / "critic_agent_0_ep_10.pth").write_bytes(b"critic")

    discovered = resolve_madrl_model_root(
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
        checkpoint_root=checkpoint_root,
    )
    explicit = resolve_madrl_model_root(
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
        model_root=algo_dir,
        checkpoint_root=checkpoint_root,
    )

    assert discovered == run_root
    assert explicit == run_root
