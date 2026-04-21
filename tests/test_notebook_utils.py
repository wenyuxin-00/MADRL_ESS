from scripts.utils.experiment_notebook_utils import resolve_madrl_model_root


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
