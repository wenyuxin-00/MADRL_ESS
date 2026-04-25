from scripts.mainline_madrl import resolve_madrl_model_root
from scripts.checkpoints import ACTOR_ACTION_MAPPING_CONTRACT, TRAINING_HEALTH_CONTRACT, build_checkpoint_manifest, write_checkpoint_manifest
from predictors.shared_data import PRICE_OBSERVATION_CONTRACT, SHARED_DATA_SCHEMA_VERSION


def test_resolve_madrl_model_root_supports_auto_discovery_and_algo_dir(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    run_root = checkpoint_root / "MATD3" / "perfect" / "grid_mainline" / "matd3_perfect_grid_mainline_ep10_20260325_120000"
    algo_dir = run_root / "MATD3"
    algo_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "_meta").mkdir(parents=True, exist_ok=True)
    (run_root / "_meta" / "train_result.json").write_text("{}", encoding="utf-8")
    (algo_dir / "actor_agent_0_ep_10.pth").write_bytes(b"actor")
    (algo_dir / "critic_agent_0_ep_10.pth").write_bytes(b"critic")
    manifest = build_checkpoint_manifest(
        algorithm="MATD3",
        saved_episode_tag=10,
        episodes_completed=10,
        total_steps=96,
        num_envs=1,
        episode_limit=96,
        save_dir=algo_dir,
        training_contract={
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
                "adjacency_type": "identity",
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
        },
    )
    write_checkpoint_manifest(algo_dir, manifest)

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
