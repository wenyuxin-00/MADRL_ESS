"""Integration tests for GridEnv.

Uses a ``FakeGridCore`` so no real pandapower network is required.
All tests run quickly without the @pytest.mark.slow decorator.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

N_AGENTS = 3
EPISODE_LIMIT = 16   # keep it short for fast tests


class FakeGridCore:
    """Minimal GridCore mock that returns zero violations without pandapower."""

    def __init__(self, n_agents: int = N_AGENTS) -> None:
        self._n_agents = n_agents

    def reset(self, base_load_kw, base_pv_kw) -> None:
        pass

    def step(self, p_batt_kw, base_load_kw):
        from envs.grid.core.grid_types import GridStepResult

        n = len(p_batt_kw)
        return GridStepResult(
            converged=True,
            vm_pu=np.ones(20, dtype=np.float32),
            va_degree=np.zeros(20, dtype=np.float32),
            line_loading_pct=np.zeros(30, dtype=np.float32),
            p_mw_from=np.zeros(30, dtype=np.float32),
            agent_vm_pu=np.ones(n, dtype=np.float32),
            v_violation=np.zeros(n, dtype=np.float32),
            l_violation=0.0,
            n_buses=20,
            n_lines=30,
        )


def _make_cfg(n_agents: int = N_AGENTS, episode_limit: int = EPISODE_LIMIT):
    """Return a full ExperimentConfig with simbench prosumer dataset."""
    from configs import compose_experiment_config

    cfg = compose_experiment_config(
        profile="debug",
        algorithm="MADDPG",
        model_family="mlp",
        reward_type="grid_composite",
        observation_profile="simbench",
        forecast_type="perfect",
    )
    cfg.env.env_type = "grid_pf"
    cfg.env.num_agents = n_agents
    cfg.env.episode_limit = episode_limit
    cfg.data.dataset_type = "csv_prosumer"
    cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.forecast.target_signals = ["price", "load", "pv"]
    return cfg


@pytest.fixture(scope="module")
def grid_env():
    """Build a GridEnv backed by a FakeGridCore."""
    from data.loaders.registry import build_dataset
    from envs.grid_env import GridEnv
    from envs.rewards import get_reward_fn
    from predictors.registry import build_forecaster
    from envs.observation.registry import build_obs_builder
    from scripts.builder import _finalize_runtime_from_env
    from models import validate_and_finalize_model_config

    cfg = _make_cfg()
    dataset = build_dataset(cfg, mode="test")
    reward_fn = get_reward_fn(cfg.reward.type, cfg)
    forecaster = build_forecaster(cfg)
    obs_builder = build_obs_builder(cfg)

    env = GridEnv(
        cfg,
        mode="test",
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=forecaster,
        obs_builder=obs_builder,
        grid_core=FakeGridCore(n_agents=N_AGENTS),
    )
    _finalize_runtime_from_env(cfg, env)
    validate_and_finalize_model_config(cfg)
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_required_attributes_present(grid_env) -> None:
    """All attributes accessed by downstream training/eval code must exist."""
    env = grid_env
    assert hasattr(env, "n") and isinstance(env.n, int)
    assert hasattr(env, "soc") and env.soc.shape == (N_AGENTS,)
    assert hasattr(env, "obs_builder")
    assert hasattr(env, "observation_schema")
    assert hasattr(env, "observation_layout")
    assert hasattr(env, "action_space") and len(env.action_space) == N_AGENTS
    assert hasattr(env, "reward_fn")


def test_reset_returns_correct_obs_shape(grid_env) -> None:
    obs = grid_env.reset()
    schema = grid_env.observation_schema
    for key, shape in schema.items():
        assert key in obs, f"Missing key '{key}' in observation"
        assert obs[key].shape == shape, (
            f"obs['{key}'].shape={obs[key].shape} but schema says {shape}"
        )


def test_step_return_types(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    obs, reward_list, done_list, info = grid_env.step(actions)

    assert isinstance(reward_list, list) and len(reward_list) == N_AGENTS
    assert isinstance(done_list, list) and len(done_list) == N_AGENTS
    assert isinstance(info, dict)


def test_info_contains_required_fields(grid_env) -> None:
    """episode_recorder.append_step_record() depends on these exact fields."""
    grid_env.reset()
    actions = [np.array([0.1], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, info = grid_env.step(actions)

    required = ["price", "e_bat_req", "e_bat", "soc_next",
                "pf_converged", "vm_pu", "agent_vm_pu",
                "line_loading_pct", "v_violation", "l_violation",
                "n_v_violations", "n_l_violations"]
    for key in required:
        assert key in info, f"Missing required info key: '{key}'"


def test_info_contains_reward_component_keys(grid_env) -> None:
    """All ComponentMeta keys must be present in info for episode_recorder."""
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, info = grid_env.step(actions)

    for meta in grid_env.reward_fn.component_meta:
        assert meta.key in info, f"Reward component key '{meta.key}' missing from info"


def test_soc_stays_in_bounds(grid_env) -> None:
    """SOC must remain in [soc_min, soc_max] throughout an episode."""
    grid_env.reset()
    for _ in range(EPISODE_LIMIT):
        actions = [
            np.array([np.random.uniform(-1, 1)], dtype=np.float32)
            for _ in range(N_AGENTS)
        ]
        _, _, done_list, _ = grid_env.step(actions)
        soc = grid_env.soc
        assert np.all(soc >= grid_env.soc_min - 1e-5), f"SoC below soc_min: {soc}"
        assert np.all(soc <= grid_env.soc_max + 1e-5), f"SoC above soc_max: {soc}"
        if any(done_list):
            break


def test_episode_recorder_compatible(grid_env) -> None:
    """append_step_record() must not crash when called with GridEnv's info dict."""
    from scripts.recorders.episode_recorder import init_episode_record, append_step_record

    reward_metas = grid_env.reward_fn.component_meta
    history = init_episode_record(
        n_agents=N_AGENTS,
        init_soc=grid_env.init_soc,
        reward_metas=reward_metas,
    )

    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, reward_list, _, info = grid_env.step(actions)

    # Should not raise.
    append_step_record(history, info, step_total=sum(reward_list), reward_metas=reward_metas)

    assert len(history["price"]) == 1
    assert len(history["e_bat_exec"][0]) == 1


def test_shared_reward_is_broadcast(grid_env) -> None:
    """Phase 1: all agents see the same reward value."""
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, reward_list, _, _ = grid_env.step(actions)
    assert len(set(reward_list)) == 1, f"Expected uniform shared reward, got {reward_list}"


def test_grid_fields_shapes(grid_env) -> None:
    """Grid-specific info fields must have the correct shapes."""
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, info = grid_env.step(actions)

    assert info["agent_vm_pu"].shape == (N_AGENTS,), (
        f"agent_vm_pu shape: {info['agent_vm_pu'].shape}"
    )
    assert info["v_violation"].shape == (N_AGENTS,), (
        f"v_violation shape: {info['v_violation'].shape}"
    )
    assert isinstance(info["l_violation"], float)
    assert isinstance(info["n_v_violations"], int)
    assert isinstance(info["n_l_violations"], int)
