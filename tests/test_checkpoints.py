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


def _write_train_result(run_root: Path) -> None:
    meta_dir = run_root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "train_result.json").write_text("{}", encoding="utf-8")


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
    older_root = Path(older["model_root"])
    newer_root = Path(newer["model_root"])
    older_algo_dir = older_root / "MATD3"
    newer_algo_dir = newer_root / "MATD3"
    older_algo_dir.mkdir(parents=True, exist_ok=True)
    newer_algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(older_algo_dir, episode_tag=50)
    _touch_checkpoint_pair(newer_algo_dir, episode_tag=100)
    _write_train_result(older_root)
    _write_train_result(newer_root)

    resolved = find_latest_training_run(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
    )

    assert resolved == Path(newer["model_root"])


def test_find_latest_training_run_skips_incomplete_newest_run(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    complete = build_training_run_paths(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="normal",
        experiment_name="train_base",
        train_episodes=150,
        max_train_steps=None,
        timestamp=datetime(2026, 4, 21, 17, 46, 57),
    )
    incomplete = build_training_run_paths(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="normal",
        experiment_name="train_base",
        train_episodes=200,
        max_train_steps=None,
        timestamp=datetime(2026, 4, 22, 22, 20, 50),
    )
    complete_root = Path(complete["model_root"])
    incomplete_root = Path(incomplete["model_root"])
    complete_algo_dir = complete_root / "MATD3"
    complete_algo_dir.mkdir(parents=True, exist_ok=True)
    incomplete_root.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(complete_algo_dir, episode_tag=150)
    _write_train_result(complete_root)
    (incomplete_root / "_meta").mkdir(parents=True, exist_ok=True)
    (incomplete_root / "_meta" / "train.log").write_text("partial run", encoding="utf-8")

    resolved = find_latest_training_run(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="normal",
        experiment_name="train_base",
    )

    assert resolved == complete_root
