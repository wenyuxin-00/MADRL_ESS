import pytest

from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from scripts.builder import _build_train_vec_env
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
