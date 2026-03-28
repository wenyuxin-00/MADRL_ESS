"""Integration tests for GridEnv using a fake GridCore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.rewards import NormalReward
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
        self._return_violations = True

    def reset(self, base_load_kw, base_pv_kw) -> None:
        del base_load_kw, base_pv_kw

    def step(self, p_batt_kw, base_load_kw):
        del base_load_kw
        from envs.grid.core.grid_types import GridStepResult

        n = len(p_batt_kw)
        line_loading_pct = np.zeros(self.n_lines, dtype=np.float32)
        trafo_loading_pct = np.zeros(self.n_trafos, dtype=np.float32)
        if self._return_violations:
            v_violation = np.array([0.02, 0.00, 0.01], dtype=np.float32)
            bus_v_excess = np.zeros(self.n_buses, dtype=np.float32)
            bus_v_excess[3] = 0.03
            bus_v_excess[7] = 0.02
            line_excess = np.zeros(self.n_lines, dtype=np.float32)
            line_excess[5] = 0.05
            trafo_excess = np.zeros(self.n_trafos, dtype=np.float32)
            trafo_excess[0] = 0.04
            line_loading_pct[5] = 105.0
            trafo_loading_pct[0] = 104.0
        else:
            v_violation = np.zeros(n, dtype=np.float32)
            bus_v_excess = np.zeros(self.n_buses, dtype=np.float32)
            line_excess = np.zeros(self.n_lines, dtype=np.float32)
            trafo_excess = np.zeros(self.n_trafos, dtype=np.float32)

        psi_v_raw = float(np.sum(bus_v_excess ** 2))
        psi_line_raw = float(np.sum(line_excess ** 2))
        psi_trafo_raw = float(np.sum(trafo_excess ** 2))

        return GridStepResult(
            converged=True,
            vm_pu=np.ones(self.n_buses, dtype=np.float32),
            va_degree=np.zeros(self.n_buses, dtype=np.float32),
            line_loading_pct=line_loading_pct,
            trafo_loading_pct=trafo_loading_pct,
            p_mw_from=np.zeros(self.n_lines, dtype=np.float32),
            agent_vm_pu=1.0 - v_violation,
            v_violation=v_violation,
            line_violation=0.10,
            trafo_violation=0.05,
            l_violation=0.10,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
            n_trafos=self.n_trafos,
            bus_v_excess=bus_v_excess,
            line_excess=line_excess,
            trafo_excess=trafo_excess,
            psi_v_raw=psi_v_raw,
            psi_line_raw=psi_line_raw,
            psi_trafo_raw=psi_trafo_raw,
        )


def _make_cfg(n_agents: int = N_AGENTS, episode_limit: int = EPISODE_LIMIT):
    from configs import compose_experiment_config

    cfg = compose_experiment_config(
        profile="debug",
        algorithm="MADDPG",
        model_family="mlp",
        forecast_type="perfect",
    )
    cfg.env.num_agents = n_agents
    cfg.env.episode_limit = episode_limit
    cfg.obs.local_features = ["calendar_time", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.forecast.target_signals = ["price", "load", "pv"]
    cfg.reward.w_trafo_pen = 7.5
    cfg.grid.train_compact_info = False
    cfg.grid.agent_bus_ids = [10, 6, 12][:n_agents]
    cfg.data.agent_profiles = ["SFH12", "SFH14", "SFH16"][:n_agents]
    cfg.data.load_components = ["household", "heatpump"]
    cfg.data.pv_reference = "south"
    cfg.data.data_dir = _ensure_case_data(n_agents=n_agents, total_steps=episode_limit * 3)
    return cfg


def _ensure_case_data(*, n_agents: int, total_steps: int) -> Path:
    data_dir = Path(__file__).resolve().parent / ".tmp" / "grid_env_case" / "data"
    prosumer_dir = data_dir / "processed" / "prosumer"
    prosumer_dir.mkdir(parents=True, exist_ok=True)

    household_csv = prosumer_dir / "household.csv"
    if not household_csv.exists():
        write_prosumer_processed_dataset(
            data_dir,
            agent_profiles=["SFH12", "SFH14", "SFH16"][:n_agents],
            train_steps=total_steps,
            test_steps=total_steps,
        )
    return data_dir


def _build_env(cfg=None, *, mode: str = "test", grid_core: FakeGridCore | None = None):
    from data.loaders.registry import build_dataset
    from envs.grid_env import GridEnv
    from envs.observation.default_builder import DefaultObservationBuilder
    from models import validate_and_finalize_model_config
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
        adjacency_type=cfg.obs.adjacency_type,
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
    assert hasattr(env, "reward_fn")


def test_reset_returns_correct_obs_shape(grid_env) -> None:
    obs, reset_info = grid_env.reset()
    for key, shape in grid_env.observation_schema.items():
        assert key in obs
        assert obs[key].shape == shape
    assert "episode_idx" in reset_info
    assert "p_max" in reset_info


def test_from_pv_battery_mode_uses_episode_pv_peak(grid_env) -> None:
    _, reset_info = grid_env.reset(episode_idx=0)
    pv_peak_kw = np.asarray(reset_info["episode_meta"]["pv_peak_kw"], dtype=np.float32)
    expected_p_max = pv_peak_kw * np.float32(grid_env.from_pv_power_ratio)
    expected_capacity = expected_p_max * np.float32(grid_env.from_pv_duration_hours)

    assert grid_env.battery_mode == "from_pv"
    assert np.allclose(reset_info["p_max"], expected_p_max)
    assert np.allclose(reset_info["battery_capacity_kwh"], expected_capacity)


def test_fixed_battery_mode_uses_cfg_defaults() -> None:
    cfg = _make_cfg()
    cfg.env.battery_mode = "fixed"
    cfg.env.battery_capacity = [10.0, 12.0, 8.0]
    cfg.env.max_charge_rate = 0.5
    env = _build_env(cfg, mode="test")

    try:
        _, reset_info = env.reset(episode_idx=0)
        assert np.allclose(reset_info["p_max"], np.array([5.0, 6.0, 4.0], dtype=np.float32))
        assert np.allclose(reset_info["battery_capacity_kwh"], np.array([10.0, 12.0, 8.0], dtype=np.float32))
    finally:
        env.close()


def test_step_return_types(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    obs, reward_list, terminated_list, truncated_list, info = grid_env.step(actions)
    assert isinstance(obs, dict)
    assert isinstance(reward_list, list) and len(reward_list) == N_AGENTS
    assert isinstance(terminated_list, list) and len(terminated_list) == N_AGENTS
    assert isinstance(truncated_list, list) and len(truncated_list) == N_AGENTS
    assert isinstance(info, dict)


def test_info_contains_required_fields(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.1], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = grid_env.step(actions)

    required = [
        "price",
        "e_bat_req",
        "e_bat",
        "soc_next",
        "pf_converged",
        "pf_error",
        "vm_pu",
        "agent_vm_pu",
        "line_loading_pct",
        "trafo_loading_pct",
        "v_violation",
        "line_violation",
        "trafo_violation",
        "n_v_violations",
        "n_l_violations",
        "n_line_violations",
        "n_t_violations",
        "n_trafo_violations",
        "psi_v_raw",
        "psi_line_raw",
        "psi_trafo_raw",
    ]
    for key in required:
        assert key in info, f"Missing required info key: '{key}'"


def test_info_contains_reward_component_keys(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = grid_env.step(actions)
    for meta in grid_env.reward_fn.component_meta:
        assert meta.key in info


def test_soc_stays_in_bounds(grid_env) -> None:
    grid_env.reset()
    for _ in range(EPISODE_LIMIT):
        actions = [np.array([np.random.uniform(-1, 1)], dtype=np.float32) for _ in range(N_AGENTS)]
        _, _, terminated_list, truncated_list, _ = grid_env.step(actions)
        assert np.all(grid_env.soc >= grid_env.soc_min - 1e-5)
        assert np.all(grid_env.soc <= grid_env.soc_max + 1e-5)
        if any(np.logical_or(terminated_list, truncated_list)):
            break


def test_episode_recorder_compatible(grid_env) -> None:
    from scripts.recorders.episode_recorder import append_step_record, init_episode_record

    reward_metas = grid_env.reward_fn.component_meta
    history = init_episode_record(
        n_agents=N_AGENTS,
        init_soc=grid_env.init_soc,
        reward_metas=reward_metas,
    )

    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, reward_list, _, _, info = grid_env.step(actions)
    append_step_record(history, info, step_total=sum(reward_list), reward_metas=reward_metas)

    assert len(history["price"]) == 1
    assert len(history["base_net_load"][0]) == 1
    assert len(history["e_bat_exec"][0]) == 1
    assert len(history["r_total_per_agent"][0]) == 1
    assert "r_cost_sum" in history
    assert "r_throughput_sum" in history
    assert "r_action_pen_sum" in history
    assert "r_safe_v_per_agent" in history
    assert "r_safe_trafo_per_agent" in history
    assert history["r_safe_trafo_per_agent"][0][0] == history["r_safe_trafo_per_agent"][2][0]
    assert history["r_safe_v_per_agent"][0][0] != history["r_safe_v_per_agent"][2][0]


def test_reward_tracks_local_voltage_differences(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, reward_list, _, _, info = grid_env.step(actions)
    assert not np.allclose(info["v_violation"], info["v_violation"][0])
    assert info["r_safe_v"][0] > info["r_safe_v"][2]
    assert np.any(np.asarray(reward_list, dtype=np.float32) != reward_list[1])


def test_trafo_penalty_is_shared(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = grid_env.step(actions)
    assert info["r_safe_trafo"][0] == info["r_safe_trafo"][1] == info["r_safe_trafo"][2]


def test_grid_fields_shapes(grid_env) -> None:
    grid_env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = grid_env.step(actions)

    assert info["agent_vm_pu"].shape == (N_AGENTS,)
    assert info["v_violation"].shape == (N_AGENTS,)
    assert info["line_loading_pct"].shape == (30,)
    assert info["trafo_loading_pct"].shape == (2,)
    assert isinstance(info["line_violation"], float)
    assert isinstance(info["trafo_violation"], float)
    assert isinstance(info["n_v_violations"], int)
    assert isinstance(info["n_l_violations"], int)
    assert isinstance(info["psi_v_raw"], float)
    assert isinstance(info["psi_line_raw"], float)
    assert isinstance(info["psi_trafo_raw"], float)


def test_compact_info_omits_large_arrays() -> None:
    cfg = _make_cfg()
    cfg.grid.train_compact_info = True
    env = _build_env(cfg, mode="train")
    reward_fn = env.reward_fn

    env.reset()
    actions = [np.array([0.0], dtype=np.float32) for _ in range(N_AGENTS)]
    _, _, _, _, info = env.step(actions)

    assert set(info) == {"episode_done", *[str(meta.key) for meta in reward_fn.component_meta]}
    env.close()
