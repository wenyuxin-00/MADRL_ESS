from datetime import datetime
from pathlib import Path

from scripts.checkpoints import (
    LATEST_CHECKPOINT_MANIFEST,
    build_checkpoint_manifest,
    build_training_run_label,
    build_training_run_paths,
    find_latest_training_run,
    infer_latest_checkpoint_tag,
    resolve_checkpoint_to_load,
    write_checkpoint_manifest,
)
from tests.support.helpers import make_case_dir


def _touch_checkpoint_pair(algo_dir: Path, episode_tag: int) -> None:
    (algo_dir / f"actor_agent_0_ep_{episode_tag}.pth").write_bytes(b"actor")
    (algo_dir / f"critic_agent_0_ep_{episode_tag}.pth").write_bytes(b"critic")


def test_checkpoint_manifest_round_trip(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_roundtrip")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)

    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=9,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
    )
    write_checkpoint_manifest(algo_dir, manifest)

    assert (algo_dir / LATEST_CHECKPOINT_MANIFEST).exists()
    assert (algo_dir / "checkpoint_ep_7.json").exists()

    resolved = resolve_checkpoint_to_load(model_dir, "MADDPG")
    assert resolved["saved_episode_tag"] == 7
    assert resolved["episodes_completed"] == 9
    assert resolved["total_steps"] == 128


def test_resolve_checkpoint_falls_back_to_scan_without_manifest(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_scan")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MATD3"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=3)
    _touch_checkpoint_pair(algo_dir, episode_tag=5)

    assert infer_latest_checkpoint_tag(model_dir, "MATD3") == 5
    resolved = resolve_checkpoint_to_load(model_dir, "MATD3")
    assert resolved["saved_episode_tag"] == 5


def test_build_training_run_label_uses_expected_tokens():
    label = build_training_run_label(
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="Grid Mainline",
        train_episodes=100,
        max_train_steps=None,
        timestamp="20260325_101500",
    )

    assert label == "matd3_perfect_grid_mainline_ep100_20260325_101500"


def test_find_latest_training_run_discovers_newest_run(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    older = build_training_run_paths(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
        train_episodes=50,
        max_train_steps=None,
        timestamp=datetime(2026, 3, 25, 10, 0, 0),
    )
    newer = build_training_run_paths(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
        train_episodes=100,
        max_train_steps=None,
        timestamp=datetime(2026, 3, 25, 10, 30, 0),
    )
    Path(older["model_root"]).mkdir(parents=True, exist_ok=True)
    Path(newer["model_root"]).mkdir(parents=True, exist_ok=True)

    resolved = find_latest_training_run(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
    )

    assert resolved == Path(newer["model_root"])
