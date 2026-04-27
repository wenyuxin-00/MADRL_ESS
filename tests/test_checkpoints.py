from datetime import datetime
import json
from pathlib import Path

import pytest

from scripts.checkpoints import (
    ACTOR_ACTION_MAPPING_CONTRACT,
    LATEST_CHECKPOINT_MANIFEST,
    TRAINING_HEALTH_CONTRACT,
    TRAINING_CONTRACT_KEY,
    TRAINING_CONTRACT_SIGNATURE_KEY,
    build_checkpoint_manifest,
    build_training_run_label,
    build_training_run_paths,
    find_latest_training_run,
    infer_latest_checkpoint_tag,
    resolve_checkpoint_to_load,
    write_checkpoint_manifest,
)
from predictors.shared_data import PRICE_OBSERVATION_CONTRACT, SHARED_DATA_SCHEMA_VERSION
from tests.support.helpers import make_case_dir

TEST_TRAINING_CONTRACT = {
    "reward_contract": "madrl_incremental_storage_reward_v1",
    "rollout_soc_contract": "continuous_soc_v1",
    "actor_action_mapping_contract": ACTOR_ACTION_MAPPING_CONTRACT,
    "training_health_contract": TRAINING_HEALTH_CONTRACT,
    "price_observation_contract": PRICE_OBSERVATION_CONTRACT,
    "discount_gamma": 0.999,
    "train_window_days": 7,
    "window_stride_days": 1,
    "train_episode_limit": 672,
    "test_episode_limit": 96,
    "learning_starts_transitions": 5376,
    "actor_learning_starts_transitions": 8064,
    "n_step_return": 96,
    "feasible_random_exploration_start": 0.5,
    "feasible_random_exploration_end": 0.05,
    "feasible_random_exploration_decay_steps": 50000,
    "shared_data_schema_version": SHARED_DATA_SCHEMA_VERSION,
    "shared_data_signature": None,
    "observation_feature_set": {
        "local": ["calendar_time", "soc"],
        "sequence": ["wholesale_price_relative", "wholesale_price_spread", "load", "pv"],
    },
    "observation_normalization_signature": None,
    "wholesale_price_spread_scale_eur_per_kwh": 0.2,
    "storage_objective_mode": "max_storage_profit",
    "storage_price_mode": "real_time_price",
    "storage_profit_weight": 1.0,
    "action_boundary_penalty_weight": 0.05,
    "soc_boundary_regularization_weight": 0.005,
    "throughput_bonus_eur_per_kwh_max": 0.002,
    "soc_boundary_epsilon": 0.02,
    "soc_boundary_margin": 0.02,
    "export_subsidy_eur_per_kwh": 0.0,
    "import_price_markup_eur_per_kwh": 0.0,
    "w_voltage_pen": 400.0,
    "w_line_pen": 0.0,
    "w_trafo_pen": 10.0,
    "train_init_soc_low": 0.05,
    "train_init_soc_high": 0.05,
}


def _touch_checkpoint_pair(algo_dir: Path, episode_tag: int) -> None:
    (algo_dir / f"actor_agent_0_ep_{episode_tag}.pth").write_bytes(b"actor")
    (algo_dir / f"critic_agent_0_ep_{episode_tag}.pth").write_bytes(b"critic")


def _write_train_result(run_root: Path) -> None:
    meta_dir = run_root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "train_result.json").write_text("{}", encoding="utf-8")


def _write_manifest(model_dir: Path, algorithm: str, episode_tag: int, *, total_steps: int = 128) -> dict:
    algo_dir = model_dir / algorithm
    manifest = build_checkpoint_manifest(
        algorithm=algorithm,
        saved_episode_tag=episode_tag,
        episodes_completed=episode_tag,
        total_steps=total_steps,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract=TEST_TRAINING_CONTRACT,
    )
    return write_checkpoint_manifest(algo_dir, manifest)


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
        training_contract=TEST_TRAINING_CONTRACT,
    )
    write_checkpoint_manifest(algo_dir, manifest)

    assert (algo_dir / LATEST_CHECKPOINT_MANIFEST).exists()
    assert (algo_dir / "checkpoint_ep_7.json").exists()

    resolved = resolve_checkpoint_to_load(model_dir, "MADDPG")
    assert resolved["saved_episode_tag"] == 7
    assert resolved["episodes_completed"] == 9
    assert resolved["total_steps"] == 128
    assert resolved[TRAINING_CONTRACT_KEY] == TEST_TRAINING_CONTRACT
    assert resolved[TRAINING_CONTRACT_SIGNATURE_KEY] == manifest[TRAINING_CONTRACT_SIGNATURE_KEY]


def test_checkpoint_manifest_rejects_terminal_soc_value_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_terminal_contract")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract={**TEST_TRAINING_CONTRACT, "terminal_soc_value_weight": 0.1},
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="terminal_soc_value_weight"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_checkpoint_manifest_rejects_action_feasibility_regularization_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_action_feasibility_contract")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract={
            **TEST_TRAINING_CONTRACT,
            "action_feasibility_regularization_weight": 0.05,
        },
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="action_feasibility_regularization_weight"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_checkpoint_manifest_rejects_old_action_mapping_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_old_action_mapping_contract")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract={
            **TEST_TRAINING_CONTRACT,
            "action_mapping_contract": "soc_aware_actor_mapping_v1",
        },
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="actor_action_mapping_contract"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_checkpoint_manifest_rejects_missing_price_observation_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_missing_price_observation_contract")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    legacy_contract = dict(TEST_TRAINING_CONTRACT)
    legacy_contract.pop("price_observation_contract")
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract=legacy_contract,
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="price_observation_contract"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_checkpoint_manifest_rejects_wrong_price_observation_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_wrong_price_observation_contract")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract={
            **TEST_TRAINING_CONTRACT,
            "price_observation_contract": "old_train_year_absolute_price_v1",
        },
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="price_observation_contract"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_checkpoint_manifest_rejects_missing_discount_gamma_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_missing_discount_gamma")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    legacy_contract = dict(TEST_TRAINING_CONTRACT)
    legacy_contract.pop("discount_gamma")
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract=legacy_contract,
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="discount_gamma"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_checkpoint_manifest_rejects_missing_action_boundary_penalty_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_missing_action_boundary_contract")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MADDPG"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=7)
    legacy_contract = dict(TEST_TRAINING_CONTRACT)
    legacy_contract.pop("action_boundary_penalty_weight")
    manifest = build_checkpoint_manifest(
        algorithm="MADDPG",
        saved_episode_tag=7,
        episodes_completed=7,
        total_steps=128,
        num_envs=4,
        episode_limit=32,
        save_dir=algo_dir,
        training_contract=legacy_contract,
    )
    write_checkpoint_manifest(algo_dir, manifest)

    with pytest.raises(ValueError, match="action_boundary_penalty_weight"):
        resolve_checkpoint_to_load(model_dir, "MADDPG")


def test_resolve_checkpoint_requires_manifest_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "checkpoint_scan")
    model_dir = case_dir / "saved_models"
    algo_dir = model_dir / "MATD3"
    algo_dir.mkdir(parents=True, exist_ok=True)
    _touch_checkpoint_pair(algo_dir, episode_tag=3)
    _touch_checkpoint_pair(algo_dir, episode_tag=5)

    assert infer_latest_checkpoint_tag(model_dir, "MATD3") == 5
    with pytest.raises(FileNotFoundError, match="Latest checkpoint manifest"):
        resolve_checkpoint_to_load(model_dir, "MATD3")


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


def test_find_latest_training_run_discovers_newest_run_without_scanning_older_legacy_runs(tmp_path):
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
    legacy_manifest = {
        "algorithm": "MATD3",
        "saved_episode_tag": 50,
        "episodes_completed": 50,
        "total_steps": 4800,
        "num_envs": 1,
        "episode_limit": 96,
        "save_dir": str(older_algo_dir),
    }
    (older_algo_dir / LATEST_CHECKPOINT_MANIFEST).write_text(
        json.dumps(legacy_manifest),
        encoding="utf-8",
    )
    (older_algo_dir / "checkpoint_ep_50.json").write_text(
        json.dumps(legacy_manifest),
        encoding="utf-8",
    )
    _write_manifest(newer_root, "MATD3", 100)
    _write_train_result(older_root)
    _write_train_result(newer_root)

    resolved = find_latest_training_run(
        checkpoint_root,
        algorithm="MATD3",
        prediction_mode="perfect",
        experiment_name="grid_mainline",
    )

    assert resolved == Path(newer["model_root"])


def test_find_latest_training_run_rejects_incomplete_newest_run(tmp_path):
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
    _write_manifest(complete_root, "MATD3", 150)
    _write_train_result(complete_root)
    (incomplete_root / "_meta").mkdir(parents=True, exist_ok=True)
    (incomplete_root / "_meta" / "train.log").write_text("partial run", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Latest training run is incomplete"):
        find_latest_training_run(
            checkpoint_root,
            algorithm="MATD3",
            prediction_mode="normal",
            experiment_name="train_base",
        )
