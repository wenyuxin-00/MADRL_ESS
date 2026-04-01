import pytest

from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from scripts.builder import _build_train_vec_env, build_env, build_train_runner
from scripts.utils.madrl_shared_data import ensure_madrl_shared_data
from tests.support.helpers import make_case_dir, make_smoke_config


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
