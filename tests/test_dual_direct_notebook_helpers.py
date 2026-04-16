from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from envs.rewards.NormalReward import NormalReward
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils import dual_direct_notebook_helpers as direct_nb

HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


def _has_working_gurobi_license() -> bool:
    if not HAS_GUROBI:
        return False
    try:
        gp, _ = direct_nb._load_gurobi()
        model = direct_nb._create_model(gp, "dual_direct_test")
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        return True
    except Exception:
        return False


HAS_WORKING_GUROBI_LICENSE = _has_working_gurobi_license()


def _make_cfg(
    *,
    dt: float = 0.25,
    episode_limit: int = 96,
    import_adder: float = 0.2,
    subsidy: float = 0.079,
    n_agents: int = 2,
    future_horizon: int = 4,
    test_start_date: str = "2020-06-01",
    test_end_date: str = "2020-06-01",
):
    return SimpleNamespace(
        env=SimpleNamespace(
            dt=float(dt),
            episode_limit=int(episode_limit),
            future_horizon=int(future_horizon),
            num_agents=int(n_agents),
        ),
        reward=SimpleNamespace(
            import_price_adder_eur_per_kwh=float(import_adder),
            export_subsidy_eur_per_kwh=float(subsidy),
            w_soc_pen=10.0,
            w_voltage_pen=10.0,
            w_line_pen=10.0,
            w_trafo_pen=10.0,
        ),
        grid=SimpleNamespace(
            line_max_loading_pct=100.0,
            v_min_pu=0.95,
            v_max_pu=1.05,
            agent_bus_ids=[idx + 1 for idx in range(int(n_agents))],
        ),
        forecast=SimpleNamespace(type="perfect", lstm_artifact_root=None),
        runtime=SimpleNamespace(shared_data_dir=None, shared_data_signature=None, forecast_ready=None),
        data=SimpleNamespace(
            test_start_date=str(test_start_date),
            test_end_date=str(test_end_date),
            agent_profiles=[f"agent_{idx}" for idx in range(int(n_agents))],
        ),
    )


def _make_fake_env():
    class _FakeEnv:
        def __init__(self):
            self.n = 2
            self.dt = 0.25
            self.num_available_episodes = 3
            self.agent_c_bat = np.asarray([4.0, 5.0], dtype=np.float32)
            self.agent_p_max = np.asarray([2.0, 2.5], dtype=np.float32)
            self.eff = 0.95
            self.soc_min = 0.1
            self.soc_max = 0.9
            self.init_soc = 0.5
            self._episodes = [
                self._make_episode("2020-06-03", base_value=30.0),
                self._make_episode("2020-06-01", base_value=10.0),
                self._make_episode("2020-06-02", base_value=20.0),
            ]
            self.ep_price = np.zeros((96,), dtype=np.float32)
            self.ep_load = np.zeros((96, self.n), dtype=np.float32)
            self.ep_pv = np.zeros((96, self.n), dtype=np.float32)

        def _make_episode(self, day: str, *, base_value: float):
            timestamps = pd.date_range(day, periods=96, freq="15min", tz="Europe/Berlin")
            price = np.linspace(0.05, 0.08, 96, dtype=np.float32) + np.float32(base_value / 1000.0)
            load = np.column_stack(
                [
                    np.full((96,), base_value + 1.0, dtype=np.float32),
                    np.full((96,), base_value + 2.0, dtype=np.float32),
                ]
            ).astype(np.float32)
            pv = np.column_stack(
                [
                    np.full((96,), base_value / 10.0, dtype=np.float32),
                    np.full((96,), base_value / 20.0, dtype=np.float32),
                ]
            ).astype(np.float32)
            return {
                "timestamps": [timestamp.isoformat() for timestamp in timestamps],
                "price": price,
                "load": load,
                "pv": pv,
            }

        def reset(self, episode_idx=0):
            episode = self._episodes[int(episode_idx)]
            self.ep_price = episode["price"].copy()
            self.ep_load = episode["load"].copy()
            self.ep_pv = episode["pv"].copy()
            return {}, {"episode_meta": {"timestamps": list(episode["timestamps"])}}

        def close(self):
            return None

    return _FakeEnv()


def _make_synthetic_problem(T: int = 16, n_agents: int = 3):
    timestamps = tuple(str(value) for value in pd.date_range("2020-06-01", periods=T, freq="15min"))
    load_kw = np.full((n_agents, T), 1.6, dtype=np.float32)
    pv_kw = np.zeros((n_agents, T), dtype=np.float32)
    pv_kw[:, 4:8] = np.asarray([[2.2], [2.0], [1.8]], dtype=np.float32)
    import_price = np.full((T,), 0.2, dtype=np.float32)
    data = direct_nb.DirectDayProblemData(
        timestamps=timestamps,
        wholesale_price_eur_per_kwh=(import_price - np.float32(0.1)).astype(np.float32),
        import_price_eur_per_kwh=import_price,
        load_kw=load_kw,
        pv_kw=pv_kw,
        battery_capacity_kwh=np.full((n_agents,), 4.0, dtype=np.float32),
        p_max_kw=np.full((n_agents,), 2.0, dtype=np.float32),
        eff_charge=np.full((n_agents,), 1.0, dtype=np.float32),
        eff_discharge=np.full((n_agents,), 1.0, dtype=np.float32),
        energy_init_kwh=np.full((n_agents,), 2.0, dtype=np.float32),
        energy_min_kwh=np.full((n_agents,), 0.4, dtype=np.float32),
        energy_max_kwh=np.full((n_agents,), 3.6, dtype=np.float32),
        export_subsidy_eur_per_kwh=0.05,
        import_price_adder_eur_per_kwh=0.1,
        dt_hours=0.25,
    )
    alpha = np.ones((n_agents, T), dtype=np.float32)
    baseline_root = np.sum(load_kw - pv_kw, axis=0).astype(np.float32)
    surrogate = direct_nb.DualTrafoSurrogate(
        trafo_limit_kw=1.0,
        alpha_netload_kw=np.ones((n_agents,), dtype=np.float32),
        alpha_netload_window_kw=alpha,
        baseline_root_p_kw=baseline_root,
        export_overload_mask=baseline_root < -1.0,
        import_overload_mask=baseline_root > 1.0,
    )
    return data, surrogate


def _make_fake_grid_env(*, n_agents: int, cfg=None):
    resolved_cfg = cfg or _make_cfg(n_agents=n_agents)

    class _FakeGridCore:
        def __init__(self):
            self.agent_bus_ids = [bus_id for bus_id in range(1, n_agents + 1)]
            self.net = SimpleNamespace(bus=pd.DataFrame(index=[0, *self.agent_bus_ids, n_agents + 1]))

        def reset(self, base_load_kw, base_pv_kw):
            self.last_reset = (
                np.asarray(base_load_kw, dtype=np.float32).copy(),
                np.asarray(base_pv_kw, dtype=np.float32).copy(),
            )

        def step(self, *, p_batt_kw, base_load_kw):
            base_load = np.asarray(base_load_kw, dtype=np.float32)
            p_batt = np.asarray(p_batt_kw, dtype=np.float32)
            total = float(np.sum(base_load + p_batt))
            agent_vm_pu = np.linspace(0.944, 1.058, num=n_agents, dtype=np.float32) + np.float32(0.0001 * total)
            vm_pu = np.asarray([1.0, *agent_vm_pu.tolist(), 1.01], dtype=np.float32)
            line_loading_pct = np.asarray([88.0 + 0.9 * abs(total), 75.0 + 0.4 * abs(total)], dtype=np.float32)
            trafo_loading_pct = np.asarray([92.0 + 0.8 * abs(total)], dtype=np.float32)
            v_min = float(resolved_cfg.grid.v_min_pu)
            v_max = float(resolved_cfg.grid.v_max_pu)
            v_violation = np.maximum(0.0, v_min - agent_vm_pu) + np.maximum(0.0, agent_vm_pu - v_max)
            bus_v_excess = np.maximum(0.0, v_min - vm_pu) + np.maximum(0.0, vm_pu - v_max)
            limit = float(resolved_cfg.grid.line_max_loading_pct)
            line_excess = np.maximum(0.0, line_loading_pct - limit) / 100.0
            trafo_excess = np.maximum(0.0, trafo_loading_pct - limit) / 100.0
            return SimpleNamespace(
                converged=True,
                vm_pu=vm_pu.astype(np.float32, copy=False),
                agent_vm_pu=agent_vm_pu.astype(np.float32, copy=False),
                line_loading_pct=line_loading_pct.astype(np.float32, copy=False),
                trafo_loading_pct=trafo_loading_pct.astype(np.float32, copy=False),
                trafo_p_signed_kw=np.asarray([total], dtype=np.float32),
                v_violation=v_violation.astype(np.float32, copy=False),
                line_violation=float(np.max(line_excess)) if line_excess.size else 0.0,
                trafo_violation=float(np.max(trafo_excess)) if trafo_excess.size else 0.0,
                l_violation=float(max(np.max(line_excess) if line_excess.size else 0.0, np.max(trafo_excess) if trafo_excess.size else 0.0)),
                bus_v_excess=bus_v_excess.astype(np.float32, copy=False),
                line_excess=line_excess.astype(np.float32, copy=False),
                trafo_excess=trafo_excess.astype(np.float32, copy=False),
                psi_v_raw=float(np.sum(bus_v_excess ** 2)),
                psi_line_raw=float(np.sum(line_excess ** 2)),
                psi_trafo_raw=float(np.sum(trafo_excess ** 2)),
                n_buses=len(self.net.bus.index),
                n_lines=int(line_loading_pct.size),
                n_trafos=int(trafo_loading_pct.size),
            )

    return SimpleNamespace(
        n=int(n_agents),
        dt=float(resolved_cfg.env.dt),
        reward_fn=NormalReward(resolved_cfg),
        _grid_core=_FakeGridCore(),
        close=lambda: None,
    )


def test_build_direct_day_problem_data_transposes_and_matches_dates(monkeypatch):
    fake_env = _make_fake_env()
    cfg = _make_cfg()
    monkeypatch.setattr(direct_nb, "build_env", lambda cfg_arg, mode: fake_env)

    data = direct_nb.build_direct_day_problem_data(
        cfg,
        test_start_date="2020-06-02",
        test_end_date="2020-06-03",
    )

    assert data.load_kw.shape == (2, 192)
    assert data.pv_kw.shape == (2, 192)
    assert data.timestamps[0].startswith("2020-06-02")
    assert data.timestamps[96].startswith("2020-06-03")
    assert data.load_kw[0, 0] == pytest.approx(21.0)
    assert data.load_kw[1, 0] == pytest.approx(22.0)
    assert data.load_kw[0, 96] == pytest.approx(31.0)
    assert data.import_price_eur_per_kwh[0] == pytest.approx(float(fake_env._episodes[2]["price"][0] + 0.2))


def test_build_direct_day_problem_data_rejects_non_day_episode_length():
    cfg = _make_cfg(dt=15.0, episode_limit=96)
    with pytest.raises(ValueError, match="one full day per episode"):
        direct_nb.build_direct_day_problem_data(
            cfg,
            test_start_date="2020-06-01",
            test_end_date="2020-06-01",
        )


def test_compute_direct_day_baseline_marks_reference_only():
    data, surrogate = _make_synthetic_problem()
    baseline = direct_nb.compute_direct_day_baseline(data, surrogate)

    assert baseline.reference_only is True
    assert np.allclose(baseline.charge_kw, 0.0)
    assert np.allclose(baseline.discharge_kw, 0.0)
    assert np.allclose(baseline.pv_curtail_kw, 0.0)
    assert np.allclose(baseline.pv_effective_kw, data.pv_kw)
    assert np.allclose(baseline.net_load_kw, data.load_kw - data.pv_kw)
    assert baseline.max_export_violation_kw >= 0.0
    assert "reference-only" in baseline.note


def test_compute_direct_day_grid_profile_shapes_and_balance_columns():
    data, surrogate = _make_synthetic_problem(T=16, n_agents=3)
    charge_kw = np.zeros((data.n_agents, data.horizon), dtype=np.float32)
    discharge_kw = np.zeros_like(charge_kw)
    pv_curtail_kw = np.zeros_like(charge_kw)
    charge_kw[0, :4] = 0.25
    discharge_kw[1, 8:12] = 0.15
    pv_curtail_kw[2, 4:8] = 0.10
    energy_kwh = np.repeat(data.energy_init_kwh[:, None], data.horizon + 1, axis=1).astype(np.float32)
    solution = direct_nb._build_solution(
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_kw,
        energy_kwh=energy_kwh,
        data=data,
        surrogate=surrogate,
        solver_status="test_solution",
    )
    fake_env = _make_fake_grid_env(n_agents=data.n_agents)

    step_df, grid_df = direct_nb.compute_direct_day_grid_profile(fake_env, data, solution)

    assert step_df.shape[0] == data.horizon
    assert grid_df.shape[0] == data.horizon * len(fake_env._grid_core.net.bus.index)
    assert "is_agent_bus" in grid_df.columns
    assert int(grid_df["is_agent_bus"].sum()) == data.horizon * data.n_agents
    np.testing.assert_allclose(
        step_df["baseline_net_load_total_kw"].to_numpy(dtype=np.float32),
        np.sum(data.load_kw - data.pv_kw, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        step_df["post_curtail_net_load_total_kw"].to_numpy(dtype=np.float32),
        np.sum(data.load_kw - solution.pv_effective_kw, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        step_df["post_action_net_load_total_kw"].to_numpy(dtype=np.float32),
        np.sum(solution.net_load_kw, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        step_df["pp_root_p_kw"].to_numpy(dtype=np.float32),
        np.sum(solution.net_load_kw, axis=0),
        atol=1e-6,
    )
    assert float(np.max(np.abs(step_df["power_balance_residual_kw"].to_numpy(dtype=np.float32)))) <= 1e-6


def test_plot_direct_day_net_load_does_not_draw_root_power_limits():
    data, surrogate = _make_synthetic_problem(T=16, n_agents=3)
    baseline = direct_nb.compute_direct_day_baseline(data, surrogate)

    figure, axis = direct_nb.plot_direct_day_net_load(
        baseline,
        data,
        surrogate,
        baseline_solution=baseline,
        label="ADMM",
    )

    labels = [line.get_label() for line in axis.get_lines()]
    assert all("Trafo limit" not in str(label) for label in labels)
    assert "root P" in axis.texts[0].get_text()
    figure.clf()


def test_collect_admm_direct_rollout_builds_compare_ready_rollout(monkeypatch):
    n_agents = 3
    cfg = _make_cfg(
        n_agents=n_agents,
        episode_limit=192,
        test_start_date="2020-06-01",
        test_end_date="2020-06-02",
    )
    data, surrogate = _make_synthetic_problem(T=96 * 2, n_agents=n_agents)
    charge_kw = np.zeros((n_agents, data.horizon), dtype=np.float32)
    discharge_kw = np.zeros_like(charge_kw)
    pv_curtail_kw = np.zeros_like(charge_kw)
    charge_kw[0, :8] = 0.25
    discharge_kw[1, 32:40] = 0.2
    pv_curtail_kw[2, 40:48] = 0.15
    energy_kwh = np.zeros((n_agents, data.horizon + 1), dtype=np.float32)
    energy_kwh[:, 0] = data.energy_init_kwh
    for step_idx in range(data.horizon):
        energy_kwh[:, step_idx + 1] = (
            energy_kwh[:, step_idx]
            + data.dt_hours * charge_kw[:, step_idx]
            - data.dt_hours * discharge_kw[:, step_idx]
        )
    solution = direct_nb._build_solution(
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_kw,
        energy_kwh=energy_kwh,
        data=data,
        surrogate=surrogate,
        solver_status="admm_converged",
    )
    admm_result = direct_nb.DirectDayAdmmResult(
        solution=solution,
        history_df=pd.DataFrame(
            {
                "iteration": [1, 2],
                "objective_estimate": [solution.objective_eur, solution.objective_eur],
                "primal_residual": [1e-2, 1e-4],
                "dual_residual": [1e-2, 1e-4],
                "export_violation_kw_max": [0.1, 0.0],
                "iter_runtime_sec": [0.01, 0.01],
                "rho_value": [0.01, 0.01],
            }
        ),
        converged=True,
        final_primal_residual=1e-4,
        final_dual_residual=1e-4,
        iterations=2,
        objective_gap_vs_centralized_eur=0.0,
        objective_gap_vs_centralized_pct=0.0,
    )
    fake_env = _make_fake_grid_env(n_agents=n_agents, cfg=cfg)
    global_oracle = grid_nb.RolloutResult(
        step_df=pd.DataFrame({"timestamp": pd.to_datetime(list(data.timestamps))}),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "Global MISOCP (single_window)"},
    )

    observed_episode_limits: list[int] = []

    monkeypatch.setattr(direct_nb, "build_env", lambda cfg_arg, mode: fake_env)

    def _fake_build_problem_data(cfg_arg, *, test_start_date, test_end_date):
        observed_episode_limits.append(int(cfg_arg.env.episode_limit))
        assert test_start_date == "2020-06-01"
        assert test_end_date == "2020-06-02"
        return data

    monkeypatch.setattr(direct_nb, "build_direct_day_problem_data", _fake_build_problem_data)
    monkeypatch.setattr(direct_nb, "build_direct_day_trafo_surrogate", lambda *args, **kwargs: surrogate)
    monkeypatch.setattr(direct_nb, "solve_direct_day_admm", lambda *args, **kwargs: admm_result)

    rollout = direct_nb.collect_admm_direct_rollout(cfg, prediction_mode="perfect")

    assert cfg.env.episode_limit == 192
    assert observed_episode_limits == [96]
    assert isinstance(rollout, grid_nb.RolloutResult)
    assert rollout.meta["controller"] == direct_nb.ADMM_DIRECT_PERFECT_LABEL
    assert rollout.meta["penalty_source"] == "replay_derived"
    assert rollout.step_df.shape[0] == data.horizon
    assert rollout.agent_df.shape[0] == data.horizon * n_agents
    assert rollout.grid_df["global_step"].nunique() == data.horizon
    assert pd.Timestamp(rollout.step_df["timestamp"].iloc[0]) == pd.Timestamp(global_oracle.step_df["timestamp"].iloc[0])
    assert pd.Timestamp(rollout.step_df["timestamp"].iloc[-1]) == pd.Timestamp(global_oracle.step_df["timestamp"].iloc[-1])
    required_step_columns = {
        "controller",
        "episode_idx",
        "step",
        "global_step",
        "timestamp",
        "price",
        "price_pred",
        "base_net_load_total",
        "base_net_load_effective_total",
        "net_load_total",
        "load_total",
        "pv_raw_total",
        "pv_effective_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "battery_charge_total",
        "battery_discharge_total",
        "purchase_cost_total",
        "export_subsidy_total",
        "objective_total",
        "soc_penalty_total",
        "voltage_penalty_total",
        "line_penalty_total",
        "trafo_penalty_total",
        "trafo_loading_pct_max",
        "n_trafo_violations",
        "pp_root_p_kw",
    }
    assert required_step_columns.issubset(set(rollout.step_df.columns))
    required_agent_columns = {
        "controller",
        "episode_idx",
        "step",
        "global_step",
        "timestamp",
        "agent_id",
        "agent_profile",
        "load",
        "load_pred",
        "pv",
        "pv_raw",
        "pv_effective",
        "pv_curtail",
        "pv_pred",
        "base_net_load",
        "base_net_load_effective",
        "net_load",
        "grid_import_kw",
        "grid_export_kw",
        "e_bat",
        "soc",
        "purchase_cost",
        "export_subsidy",
        "objective_total",
    }
    assert required_agent_columns.issubset(set(rollout.agent_df.columns))
    assert {"controller", "episode_idx", "step", "global_step", "timestamp", "bus_id", "vm_pu", "is_agent_bus"}.issubset(
        set(rollout.grid_df.columns)
    )
    np.testing.assert_allclose(
        rollout.step_df["pv_curtail_total"].to_numpy(dtype=np.float32),
        np.sum(solution.pv_curtail_kw, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rollout.step_df["pv_effective_total"].to_numpy(dtype=np.float32),
        np.sum(solution.pv_effective_kw, axis=0),
        atol=1e-6,
    )
    assert float(rollout.step_df["soc_penalty_total"].sum()) == pytest.approx(0.0)
    assert float(rollout.step_df["voltage_penalty_total"].sum()) >= 0.0
    assert float(rollout.step_df["line_penalty_total"].sum()) >= 0.0
    assert float(rollout.step_df["trafo_penalty_total"].sum()) >= 0.0
    metrics_df = grid_nb.compare_rollout_metrics(rollout)
    costs_df = grid_nb.compare_purchase_costs(rollout)
    banner = grid_nb.build_compare_warning_banner(rollout)
    figure = grid_nb.plot_power_balance_comparison(rollout)
    assert metrics_df.loc[0, "controller"] == direct_nb.ADMM_DIRECT_PERFECT_LABEL
    assert costs_df.loc[0, "controller"] == direct_nb.ADMM_DIRECT_PERFECT_LABEL
    assert "optional compare fields missing" not in banner.data
    assert len(figure.axes) >= 1
    figure.clf()


def test_collect_admm_direct_rollout_rejects_nonperfect_prediction_mode():
    cfg = _make_cfg()
    with pytest.raises(ValueError, match="only supports perfect forecasts"):
        direct_nb.collect_admm_direct_rollout(cfg, prediction_mode="normal")


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real direct-day solver tests",
)
def test_solve_direct_day_centralized_has_no_integer_vars_and_respects_constraints():
    data, surrogate = _make_synthetic_problem()
    solution = direct_nb.solve_direct_day_centralized(data, surrogate)

    assert solution.solver_metadata["num_binary_vars"] == 0
    assert solution.solver_metadata["num_integer_vars"] == 0
    assert np.all(solution.charge_kw >= -1e-6)
    assert np.all(solution.charge_kw <= data.p_max_kw[:, None] + 1e-6)
    assert np.all(solution.discharge_kw >= -1e-6)
    assert np.all(solution.discharge_kw <= data.p_max_kw[:, None] + 1e-6)
    assert np.all(solution.pv_curtail_kw >= -1e-6)
    assert np.all(solution.pv_curtail_kw <= data.pv_kw + 1e-6)
    assert np.all(solution.energy_kwh >= data.energy_min_kwh[:, None] - 1e-6)
    assert np.all(solution.energy_kwh <= data.energy_max_kwh[:, None] + 1e-6)
    np.testing.assert_allclose(solution.energy_kwh[:, 0], data.energy_init_kwh, atol=1e-6)
    np.testing.assert_allclose(solution.energy_kwh[:, -1], data.energy_init_kwh, atol=1e-6)


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real direct-day solver tests",
)
def test_solve_direct_day_admm_converges_on_small_synthetic_problem():
    data, surrogate = _make_synthetic_problem()
    centralized = direct_nb.solve_direct_day_centralized(data, surrogate)
    rho_init = float(
        np.mean(data.import_price_eur_per_kwh) * data.dt_hours
        / max(1.0, float(np.mean(np.abs(surrogate.alpha_netload_window_kw * (data.load_kw - data.pv_kw)))))
    )

    admm_result = direct_nb.solve_direct_day_admm(
        data,
        surrogate,
        rho_init=rho_init,
        rho_adaptation="residual_balancing",
        max_iters=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        centralized_objective_eur=centralized.objective_eur,
    )

    assert admm_result.history_df.shape[0] >= 1
    assert admm_result.final_primal_residual < 1e-3 or not admm_result.converged
    assert admm_result.final_dual_residual < 1e-3 or not admm_result.converged
    assert admm_result.objective_gap_vs_centralized_pct < 0.5
    np.testing.assert_allclose(admm_result.solution.energy_kwh[:, 0], data.energy_init_kwh, atol=1e-6)
    np.testing.assert_allclose(admm_result.solution.energy_kwh[:, -1], data.energy_init_kwh, atol=1e-6)
