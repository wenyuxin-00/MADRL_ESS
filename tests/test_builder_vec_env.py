import pytest

from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from scripts.builder import _build_train_vec_env, build_env, build_train_runner
from scripts.utils.madrl_shared_data import ensure_madrl_shared_data
from tests.support.helpers import make_case_dir, make_smoke_config, write_prosumer_processed_dataset


def _make_multiday_cfg(tmp_path, *, evaluation_days: int = 5):
    cfg = make_smoke_config(tmp_path, algorithm="MADDPG")
    cfg.env.episode_limit = 96
    cfg.env.future_horizon = 1
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    write_prosumer_processed_dataset(
        cfg.data.data_dir,
        agent_profiles=list(cfg.data.agent_profiles),
        train_year=2019,
        test_year=2020,
        train_steps=96 * evaluation_days,
        test_steps=96 * evaluation_days,
    )
    return cfg


def test_build_train_vec_env_falls_back_to_dummy_when_subproc_is_unsupported(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "vec_env_fallback")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.vec_env_type = "subproc"
    cfg.train.num_envs = 2

    monkeypatch.setattr(
        "scripts.builder._subproc_vec_env_is_supported_in_current_process",
        lambda: (False, "simulated interactive session"),
    )

    with pytest.warns(RuntimeWarning, match="Falling back to DummyVecEnv"):
        vec_env = _build_train_vec_env(cfg, seed=0)

    try:
        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 2
    finally:
        vec_env.close()


def test_subproc_vec_env_surfaces_worker_init_errors(tmp_path):
    case_dir = make_case_dir(tmp_path, "vec_env_worker_init_error")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.data.data_dir = case_dir / "missing_data"

    with pytest.raises(RuntimeError, match="Missing processed prosumer file"):
        SubprocVecEnv(1, cfg, mode="train")


def test_build_env_uses_shared_data_without_building_live_forecaster(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "shared_data_env")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    shared_data = ensure_madrl_shared_data(cfg, root=case_dir / "artifacts" / "training" / "shared_data")
    cfg.runtime.shared_data_dir = str(shared_data.shared_data_dir)
    cfg.runtime.shared_data_signature = str(shared_data.signature_hash)

    def _unexpected_build_forecaster(_cfg):
        raise AssertionError("build_forecaster should not be called when shared_data_dir is set")

    monkeypatch.setattr("scripts.builder.build_forecaster", _unexpected_build_forecaster)

    env = build_env(cfg, mode="test")
    try:
        assert env.forecaster is None
        assert env.has_precomputed_observations() is True
    finally:
        env.close()


def test_build_env_resets_runtime_split_state_without_shared_data(tmp_path):
    case_dir = make_case_dir(tmp_path, "cfg_split_runtime")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.runtime.effective_split_controls = {"source": "stale"}
    cfg.runtime.selected_episode_indices = [99]

    env = build_env(cfg, mode="test")
    try:
        assert env.has_precomputed_observations() is False
        assert cfg.runtime.effective_split_controls["source"] == "cfg"
        assert cfg.runtime.effective_split_controls["split"] == "test"
        assert cfg.runtime.selected_episode_indices is None
    finally:
        env.close()


def test_build_env_uses_shared_data_manifest_controls_and_episode_subset(tmp_path):
    case_dir = make_case_dir(tmp_path, "shared_data_manifest_slice")
    cfg = _make_multiday_cfg(case_dir, evaluation_days=5)
    shared_data = ensure_madrl_shared_data(cfg, root=case_dir / "artifacts" / "training" / "shared_data")
    cfg.runtime.shared_data_dir = str(shared_data.shared_data_dir)
    cfg.runtime.shared_data_signature = str(shared_data.signature_hash)
    cfg.data.test_start_date = "2020-01-02"
    cfg.data.test_end_date = "2020-01-04"

    env = build_env(cfg, mode="test")
    try:
        assert env.has_precomputed_observations() is True
        assert cfg.runtime.effective_split_controls["source"] == "shared_data_manifest"
        assert cfg.runtime.effective_split_controls["start_date"] is None
        assert cfg.runtime.effective_split_controls["end_date"] is None
        assert cfg.runtime.effective_split_controls["window_strategy"] == "full_year_runtime_slice"
        assert cfg.runtime.selected_episode_indices == [1, 2, 3]
    finally:
        env.close()


def test_build_train_runner_records_shared_data_metadata(tmp_path):
    case_dir = make_case_dir(tmp_path, "shared_data_runner")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    shared_data = ensure_madrl_shared_data(cfg, root=case_dir / "artifacts" / "training" / "shared_data")
    cfg.runtime.shared_data_dir = str(shared_data.shared_data_dir)
    cfg.runtime.shared_data_signature = str(shared_data.signature_hash)

    runner = build_train_runner(cfg, seed=0)
    try:
        metadata = dict(getattr(runner, "shared_data_metadata", {}) or {})
        assert metadata["shared_data_dir"] == str(shared_data.shared_data_dir)
        assert metadata["shared_data_signature"] == str(shared_data.signature_hash)
        assert set(metadata["split_dirs"]) == {"train", "test"}
    finally:
        runner.close()
