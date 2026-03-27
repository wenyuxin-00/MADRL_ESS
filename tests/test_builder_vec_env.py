import pytest

from envs.fastlab.subproc_vec_env_fastlab import SubprocVecEnvFastLab
from envs.vec_env import DummyVecEnv
from scripts.builder_fastlab import _build_train_vec_env_fastlab
from scripts.utils.madrl_observation_cache_lab import build_or_load_observation_cache
from tests.support.helpers import make_case_dir, make_smoke_config


def test_build_train_vec_env_falls_back_to_dummy_when_subproc_is_unsupported(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "vec_env_fallback")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.vec_env_type = "subproc"
    cfg.train.num_envs = 2
    cache_result = build_or_load_observation_cache(
        cfg,
        split="train",
        refresh=True,
        root=case_dir / "cache",
    )
    cfg.runtime.fastlab_observation_cache_dir = str(cache_result.cache_dir)
    cfg.runtime.fastlab_train_info_mode = "minimal"
    cfg.runtime.fastlab_fast_grid_core = True

    monkeypatch.setattr(
        "scripts.builder_fastlab._subproc_vec_env_is_supported_in_current_process",
        lambda: (False, "simulated interactive session"),
    )

    with pytest.warns(RuntimeWarning, match="Falling back to DummyVecEnv"):
        vec_env = _build_train_vec_env_fastlab(cfg, seed=0)

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
        SubprocVecEnvFastLab(1, cfg, mode="train")
