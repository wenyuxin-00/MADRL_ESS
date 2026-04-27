"""Integration tests for GridEnv using a fake GridCore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from controllers.madrl.safety_projector import _local_bounds_numpy, build_safety_local_numpy
from envs.rewards.NormalReward import NormalReward
from predictors.shared_data import ensure_madrl_shared_data
from tests.support.helpers import write_prosumer_processed_dataset

N_AGENTS = 3
EPISODE_LIMIT = 16


class FakeGridCore:
    """Minimal GridCore mock that returns configurable violations without pandapower."""

    def __init__(self, n_agents: int = N_AGENTS) -> None:
        self._n_agents = n_agents
        self.n_buses = 20
        self.n_lines = 30
        self.n_trafos = 2
        self.last_pf_error = ""
        self.converged = True
        self._return_violations = True

    def reset(self, base_load_kw, base_pv_kw) -> None:
        del base_load_kw, base_pv_kw

    def step(self, p_batt_kw, base_load_kw):
        del base_load_kw
        from envs.grid.grid_core import GridStepResult

        n = len(p_batt_kw)
        line_loading_pct = np.zeros(self.n_lines, dtype=np.float32)
        trafo_loading_pct = np.zeros(self.n_trafos, dtype=np.float32)
        if self._return_violations:
            v_violation = np.array([0.02, 0.00, 0.01], dtype=np.float32)
            psi_v_raw = float(0.03 ** 2 + 0.02 ** 2)
            psi_line_raw = float(0.05 ** 2)
            psi_trafo_raw = float(0.04 ** 2)
            line_loading_pct[5] = 105.0
            trafo_loading_pct[0] = 104.0
        else:
            v_violation = np.zeros(n, dtype=np.float32)
            psi_v_raw = 0.0
            psi_line_raw = 0.0
            psi_trafo_raw = 0.0

        return GridStepResult(
            converged=bool(self.converged),
            vm_pu=np.ones(self.n_buses, dtype=np.float32),
            line_loading_pct=line_loading_pct,
            trafo_loading_pct=trafo_loading_pct,
            v_violation=v_violation,
            psi_v_raw=psi_v_raw,
            psi_line_raw=psi_line_raw,
            psi_trafo_raw=psi_trafo_raw,
            trafo_p_signed_kw=np.asarray([12.5, -1.5], dtype=np.float32),
        )


def _make_cfg(n_agents: int = N_AGENTS, episode_limit: int = EPISODE_LIMIT):
    from configs.profiles import compose_experiment_config

    cfg = compose_experiment_config(
        profile="debug",
        algorithm="MADDPG",
        model_family="mlp",
        forecast_type="perfect",
    )
    cfg.env.num_agents = n_agents
    cfg.env.episode_limit = episode_limit
    cfg.env.train_window_days = 1
    cfg.env.window_stride_days = 1
    cfg.obs.local_features = ["calendar_time", "soc"]
    cfg.obs.sequence_features = ["wholesale_price", "load", "pv"]
    cfg.forecast.target_signals = ["wholesale_price", "load", "pv"]
    cfg.reward.w_trafo_pen = 7.5
    cfg.grid.train_compact_info = False
    cfg.grid.agent_bus_ids = [10, 6, 12][:n_agents]
    cfg.data.agent_profiles = ["SFH12", "SFH14", "SFH16"][:n_agents]
    cfg.data.load_scale = [10.0] * n_agents
    cfg.data.pv_scale = [10.0] * n_agents
    cfg.data.load_components = ["household", "heatpump"]
    cfg.data.pv_reference = "south"
    cfg.data.data_dir = _ensure_case_data(n_agents=n_agents, total_steps=episode_limit * 3)
    return cfg


def _ensure_case_data(*, n_agents: int, total_steps: int) -> Path:
    data_dir = Path(__file__).resolve().parent / ".tmp" / "grid_env_case" / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14", "SFH16"][:n_agents],
        train_steps=total_steps,
        test_steps=max(total_steps, 96 * 110),
    )
    return data_dir


def _build_env(cfg=None, *, mode: str = "test", grid_core: FakeGridCore | None = None):
    from data.loaders.registry import build_dataset
    from envs.grid_env import GridEnv
    from envs.observation.default_builder import DefaultObservationBuilder
    from models.assembly import validate_and_finalize_model_config
    from predictors.registry import build_forecaster
    from scripts.builder import _finalize_runtime_from_env

    cfg = _make_cfg() if cfg is None else cfg
    grid_core = grid_core or FakeGridCore(n_agents=int(cfg.env.num_agents))

    dataset = build_dataset(cfg, mode="test")
    reward_fn = NormalReward(cfg)
    forecaster = build_forecaster(cfg)
    obs_builder = DefaultObservationBuilder(
        local_features=cfg.obs.local_features,
        sequence_features=cfg.obs.sequence_features,
        future_horizon=cfg.env.future_horizon,
    )

    env = GridEnv(
        cfg,
        mode=mode,
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=forecaster,
        obs_builder=obs_builder,
        grid_core=grid_core,
    )
    _finalize_runtime_from_env(cfg, env)
    validate_and_finalize_model_config(cfg)
    return env


def _zero_actions() -> list[np.ndarray]:
    return [np.array([0.0, 1.0], dtype=np.float32) for _ in range(N_AGENTS)]


def test_grid_env_accepts_precomputed_data_dir(tmp_path) -> None:
    from data.loaders.registry import build_dataset
    from envs.grid_env import GridEnv
    from envs.observation.default_builder import DefaultObservationBuilder

    cfg = _make_cfg()
    shared_data = ensure_madrl_shared_data(cfg, root=tmp_path / "shared_data")
    dataset = build_dataset(cfg, mode="test")
    obs_builder = DefaultObservationBuilder(
        local_features=cfg.obs.local_features,
        sequence_features=cfg.obs.sequence_features,
        future_horizon=cfg.env.future_horizon,
        precomputed=True,
    )
    env = GridEnv(
        cfg,
        mode="test",
        dataset=dataset,
        reward_fn=NormalReward(cfg),
        forecaster=None,
        obs_builder=obs_builder,
        grid_core=FakeGridCore(n_agents=int(cfg.env.num_agents)),
        precomputed_data_dir=shared_data.shared_data_dir / "test",
    )

    try:
        obs, _ = env.reset(episode_idx=0)
        assert env.forecaster is None
        assert getattr(env, "_precomputed_store", None) is not None
        for key, shape in env.observation_schema.items():
            assert obs[key].shape == shape
    finally:
        env.close()


@pytest.fixture(scope="module")
def grid_env():
    return _build_env(_make_cfg(), mode="test")


def test_required_attributes_present(grid_env) -> None:
    env = grid_env
    assert hasattr(env, "n") and isinstance(env.n, int)
    assert hasattr(env, "soc") and env.soc.shape == (N_AGENTS,)
    assert hasattr(env, "obs_builder")
    assert hasattr(env, "observation_schema")
    assert hasattr(env, "observation_layout")
    assert hasattr(env, "action_space") and len(env.action_space) == N_AGENTS
    assert env.action_space[0].shape == (2,)
    assert hasattr(env, "reward_fn")


def test_reset_returns_correct_obs_shape(grid_env) -> None:
    obs, reset_info = grid_env.reset()
    for key, shape in grid_env.observation_schema.items():
        assert key in obs
        assert obs[key].shape == shape
    assert "episode_idx" in reset_info
    assert "p_max" in reset_info


def test_default_battery_config_uses_fixed_defaults(grid_env) -> None:
    _, reset_info = grid_env.reset(episode_idx=0)
    assert np.allclose(reset_info["p_max"], np.asarray(grid_env.agent_p_max, dtype=np.float32))
    assert np.allclose(reset_info["battery_capacity_kwh"], np.asarray(grid_env.agent_c_bat, dtype=np.float32))


def test_fixed_battery_mode_uses_cfg_defaults() -> None:
    cfg = _make_cfg()
    cfg.env.battery_capacity = [10.0, 12.0, 8.0]
    cfg.env.max_charge_rate = 0.5
    env = _build_env(cfg, mode="test")

    try:
        _, reset_info = env.reset(episode_idx=0)
        assert np.allclose(reset_info["p_max"], np.array([5.0, 6.0, 4.0], dtype=np.float32))
        assert np.allclose(reset_info["battery_capacity_kwh"], np.array([10.0, 12.0, 8.0], dtype=np.float32))
    finally:
        env.close()


def test_fixed_battery_mode_accepts_numpy_capacity_vector() -> None:
    cfg = _make_cfg()
    cfg.env.battery_capacity = np.asarray([10.0, 12.0, 8.0], dtype=np.float32)
    cfg.env.max_charge_rate = 0.5
    env = _build_env(cfg, mode="test")

    try:
        _, reset_info = env.reset(episode_idx=0)
        assert np.allclose(reset_info["p_max"], np.array([5.0, 6.0, 4.0], dtype=np.float32))
        assert np.allclose(reset_info["battery_capacity_kwh"], np.array([10.0, 12.0, 8.0], dtype=np.float32))
    finally:
        env.close()


def test_close_cleans_up_private_local_mpc_cache() -> None:
    env = _build_env(_make_cfg(), mode="test")

    class _DummySolver:
        def __init__(self) -> None:
            self.disposed = False

        def dispose(self) -> None:
            self.disposed = True

    solver = _DummySolver()
    env._local_mpc_solver_cache[("agent", 0)] = solver

    env.close()

    assert solver.disposed is True
    assert env._local_mpc_solver_cache == {}


def test_step_return_types(grid_env) -> None:
    grid_env.reset()
    actions = _zero_actions()
    obs, reward_list, terminated_list, truncated_list, info = grid_env.step(actions)
    assert isinstance(obs, dict)
    assert isinstance(reward_list, list) and len(reward_list) == N_AGENTS
    assert isinstance(terminated_list, list) and len(terminated_list) == N_AGENTS
    assert isinstance(truncated_list, list) and len(truncated_list) == N_AGENTS
    assert isinstance(info, dict)


def test_info_contains_required_fields(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.1, 1.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = grid_env.step(actions)

    required = [
        "wholesale_price",
        "import_price",
        "e_bat_req",
        "e_bat",
        "pv_effective",
        "pv_curtail",
        "pv_utilization",
        "grid_import_kw",
        "grid_export_kw",
        "soc_next",
        "pf_converged",
        "pf_error",
        "vm_pu",
        "line_loading_pct",
        "trafo_loading_pct",
        "trafo_p_signed_kw",
        "v_violation",
        "n_v_violations",
        "n_line_violations",
        "n_trafo_violations",
        "psi_v_raw",
        "psi_line_raw",
        "psi_trafo_raw",
    ]
    for key in required:
        assert key in info, f"Missing required info key: '{key}'"
    assert info["import_price"] == pytest.approx(
        info["wholesale_price"] + grid_env.import_price_markup_eur_per_kwh
    )
    assert np.allclose(
        np.asarray(info["trafo_p_signed_kw"], dtype=np.float32),
        np.asarray([12.5, -1.5], dtype=np.float32),
    )


def test_price_signal_remains_wholesale_but_cost_price_is_adjusted(grid_env) -> None:
    grid_env.reset(episode_idx=0)
    raw_price = float(grid_env.get_signal_step("wholesale_price", 0))

    _, _, _, _, info = grid_env.step(_zero_actions())

    assert float(grid_env.get_signal("wholesale_price")[0]) == pytest.approx(raw_price)
    assert info["wholesale_price"] == pytest.approx(raw_price)
    assert info["import_price"] == pytest.approx(raw_price + grid_env.import_price_markup_eur_per_kwh)


def test_info_contains_reward_component_keys(grid_env) -> None:
    grid_env.reset()
    actions = _zero_actions()
    _, _, _, _, info = grid_env.step(actions)
    for meta in grid_env.reward_fn.component_meta:
        assert meta.key in info


def test_soc_stays_in_bounds(grid_env) -> None:
    grid_env.reset()
    for _ in range(EPISODE_LIMIT):
        safety_local = build_safety_local_numpy(
            soc=np.asarray(grid_env.soc, dtype=np.float32),
            load_raw=np.asarray(grid_env.get_signal_step("load"), dtype=np.float32),
            pv_raw=np.asarray(grid_env.get_signal_step("pv"), dtype=np.float32),
            battery_capacity_kwh=np.asarray(grid_env.agent_c_bat, dtype=np.float32),
            p_max_kw=np.asarray(grid_env.agent_p_max, dtype=np.float32),
        )
        lower_kw, upper_kw, p_max_kw, _ = _local_bounds_numpy(
            safety_local,
            efficiency=float(grid_env.eff),
            dt_hours=float(grid_env.dt),
            soc_min=float(grid_env.soc_min),
            soc_max=float(grid_env.soc_max),
        )
        actions = []
        for lower, upper, p_max in zip(lower_kw, upper_kw, p_max_kw, strict=False):
            sampled_kw = float(np.random.uniform(lower, upper)) if upper > lower else float(lower)
            normalized = sampled_kw / max(float(p_max), 1e-6)
            actions.append(np.array([normalized, 1.0], dtype=np.float32))
        _, _, terminated_list, truncated_list, _ = grid_env.step(actions)
        assert np.all(grid_env.soc >= grid_env.soc_min - 1e-5)
        assert np.all(grid_env.soc <= grid_env.soc_max + 1e-5)
        if any(np.logical_or(terminated_list, truncated_list)):
            break


def test_reward_tracks_local_voltage_differences(grid_env) -> None:
    grid_env.reset()
    actions = _zero_actions()
    _, reward_list, _, _, info = grid_env.step(actions)
    assert not np.allclose(info["v_violation"], info["v_violation"][0])
    assert info["madrl_r_safe_v"][0] > info["madrl_r_safe_v"][2]
    assert np.any(np.asarray(reward_list, dtype=np.float32) != reward_list[1])


def test_trafo_penalty_is_shared(grid_env) -> None:
    grid_env.reset()
    actions = _zero_actions()
    _, _, _, _, info = grid_env.step(actions)
    assert (
        info["madrl_r_safe_trafo"][0]
        == info["madrl_r_safe_trafo"][1]
        == info["madrl_r_safe_trafo"][2]
    )


def test_grid_fields_shapes(grid_env) -> None:
    grid_env.reset()
    actions = _zero_actions()
    _, _, _, _, info = grid_env.step(actions)

    assert info["v_violation"].shape == (N_AGENTS,)
    assert info["pv_effective"].shape == (N_AGENTS,)
    assert info["pv_curtail"].shape == (N_AGENTS,)
    assert info["grid_import_kw"].shape == (N_AGENTS,)
    assert info["grid_export_kw"].shape == (N_AGENTS,)
    assert info["line_loading_pct"].shape == (30,)
    assert info["trafo_loading_pct"].shape == (2,)
    assert isinstance(info["n_v_violations"], int)
    assert isinstance(info["n_line_violations"], int)
    assert isinstance(info["psi_v_raw"], float)
    assert isinstance(info["psi_line_raw"], float)
    assert isinstance(info["psi_trafo_raw"], float)


def test_compact_info_omits_large_arrays() -> None:
    cfg = _make_cfg()
    cfg.grid.train_compact_info = True
    env = _build_env(cfg, mode="train")
    reward_fn = env.reward_fn

    env.reset()
    actions = _zero_actions()
    _, _, _, _, info = env.step(actions)

    assert set(info) == {
        "episode_done",
        "madrl_throughput_bonus_weight",
        "madrl_throughput_kwh",
        "pf_converged",
        "pf_error",
        *[str(meta.key) for meta in reward_fn.component_meta],
    }
    env.close()


def test_train_info_reports_power_flow_convergence() -> None:
    cfg = _make_cfg()
    grid_core = FakeGridCore(n_agents=int(cfg.env.num_agents))
    grid_core.converged = False
    grid_core.last_pf_error = "fake solver failed"
    env = _build_env(cfg, mode="train", grid_core=grid_core)

    try:
        env.reset()
        _, _, _, _, info = env.step(_zero_actions())

        assert info["pf_converged"] is False
        assert info["pf_error"] == "fake solver failed"
    finally:
        env.close()


def test_single_dim_actions_raise_fail_fast(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    with pytest.raises(ValueError, match="exactly two action dimensions"):
        grid_env.step(actions)


def test_second_action_dimension_controls_pv_curtailment(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0, -1.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = grid_env.step(actions)

    np.testing.assert_allclose(info["pv_effective"], 0.0, atol=1e-6)
    np.testing.assert_allclose(info["pv_curtail"], info["pv"], atol=1e-6)
    np.testing.assert_allclose(info["pv_utilization"], 0.0, atol=1e-6)


def test_locally_infeasible_charge_action_raises(grid_env) -> None:
    grid_env.reset()
    grid_env.soc = np.full((N_AGENTS,), grid_env.soc_max, dtype=np.float32)
    actions = [np.array([1.0, 1.0], dtype=np.float32) for _ in range(N_AGENTS)]

    with pytest.raises(ValueError, match="locally infeasible battery action"):
        grid_env.step(actions)
