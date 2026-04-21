from __future__ import annotations

import builtins
import json
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.utils import admm_mpc_notebook_helpers as admm_mpc_nb
from scripts.utils import grid_notebook_workflow as grid_nb


HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


def _has_working_gurobi_license() -> bool:
    if not HAS_GUROBI:
        return False
    try:
        gp, _ = admm_mpc_nb._load_gurobi()
        model = admm_mpc_nb._create_model(gp, "admm_mpc_test")
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        return True
    except Exception:
        return False


HAS_WORKING_GUROBI_LICENSE = _has_working_gurobi_license()


def _make_cfg(*, future_horizon: int = 4, n_agents: int = 2, test_start_date: str = "2020-06-01", test_end_date: str = "2020-06-05"):
    return SimpleNamespace(
        env=SimpleNamespace(
            dt=0.25,
            episode_limit=192,
            future_horizon=int(future_horizon),
            num_agents=int(n_agents),
            battery_capacity=[4.0] * int(n_agents),
            max_charge_rate=0.5,
            efficiency=0.95,
            init_soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            soc_target=0.5,
        ),
        reward=SimpleNamespace(
            import_price_markup_eur_per_kwh=0.2,
            export_subsidy_eur_per_kwh=0.079,
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
            load_scale=[1.0] * int(n_agents),
            pv_scale=[1.0] * int(n_agents),
        ),
    )


def _make_env(*, n_agents: int = 2, future_horizon: int = 4):
    class _ObsBuilder:
        def __init__(self):
            self.raw_obs = {
                "wholesale_price_seq": np.asarray([0.10, 0.11, 0.12, 0.13, 0.14], dtype=np.float32)[: future_horizon + 1],
                "load_seq": np.asarray(
                    [[1.0, 1.1, 1.2, 1.1, 1.0], [0.8, 0.9, 1.0, 0.9, 0.8]],
                    dtype=np.float32,
                )[:, : future_horizon + 1],
                "pv_seq": np.asarray(
                    [[0.0, 0.2, 0.4, 0.2, 0.0], [0.0, 0.1, 0.2, 0.1, 0.0]],
                    dtype=np.float32,
                )[:, : future_horizon + 1],
            }

        def build_raw(self, env):
            del env
            return {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in self.raw_obs.items()
            }

    return SimpleNamespace(
        n=int(n_agents),
        future_horizon=int(future_horizon),
        num_available_episodes=1,
        episode_length=96,
        cur_step=0,
        soc=np.full((n_agents,), 0.5, dtype=np.float32),
        agent_c_bat=np.full((n_agents,), 4.0, dtype=np.float32),
        agent_p_max=np.full((n_agents,), 2.0, dtype=np.float32),
        eff=1.0,
        soc_min=0.0,
        soc_max=1.0,
        dt=0.25,
        obs_builder=_ObsBuilder(),
    )


def _make_solver_problem(*, horizon: int = 8, n_agents: int = 2):
    return admm_mpc_nb.AdmmMpcWindowData(
        wholesale_price_seq=np.full((horizon,), 0.0, dtype=np.float32),
        wholesale_price_eur_per_kwh=np.full((horizon,), 0.0, dtype=np.float32),
        import_price_eur_per_kwh=np.full((horizon,), 0.2, dtype=np.float32),
        load_seq=np.zeros((n_agents, horizon), dtype=np.float32),
        pv_seq=np.zeros((n_agents, horizon), dtype=np.float32),
        battery_capacity_kwh=np.full((n_agents,), 4.0, dtype=np.float32),
        p_max_kw=np.full((n_agents,), 2.0, dtype=np.float32),
        eff_charge=np.full((n_agents,), 1.0, dtype=np.float32),
        eff_discharge=np.full((n_agents,), 1.0, dtype=np.float32),
        energy_init_kwh=np.full((n_agents,), 2.0, dtype=np.float32),
        energy_ref_kwh=np.full((n_agents,), 2.0, dtype=np.float32),
        energy_min_kwh=np.zeros((n_agents,), dtype=np.float32),
        energy_max_kwh=np.full((n_agents,), 4.0, dtype=np.float32),
        export_subsidy_eur_per_kwh=0.05,
        import_price_markup_eur_per_kwh=0.2,
        dt_hours=0.25,
    )


def _make_surrogate_cache(*, horizon: int = 8, n_agents: int = 2):
    return admm_mpc_nb.AdmmMpcSurrogateCache(
        trafo_limit_kw=100.0,
        trafo_base_kw=0.0,
        alpha_netload_kw=np.ones((n_agents,), dtype=np.float32),
        alpha_netload_window_kw=np.ones((n_agents, horizon), dtype=np.float32),
        horizon_steps=int(horizon),
    )


def _make_cached_rollout(
    *,
    controller: str = "ADMM MPC + LSTM Forecast",
    forecast_backend: str = "lstm",
) -> grid_nb.RolloutResult:
    timestamp = pd.Timestamp("2020-06-01 00:00:00")
    step_df = pd.DataFrame(
        [
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "wholesale_price": 0.1,
                "import_price": 0.3,
                "wholesale_price_pred": 0.3,
                "import_price_pred": 0.5,
                "purchase_cost_total": 0.33,
                "export_subsidy_total": 0.0,
                "objective_total": 0.33,
                "admm_converged": True,
                "admm_iterations": 7,
                "admm_final_primal_residual": 1e-4,
                "admm_final_dual_residual": 2e-4,
                "admm_solve_time_sec": 0.05,
                "admm_rho_final": 0.2,
            }
        ]
    )
    agent_df = pd.DataFrame(
        [
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "agent_id": 0,
                "agent_profile": "agent_0",
                "load": 1.2,
                "load_pred": 1.2,
                "pv": 0.5,
                "pv_pred": 0.5,
                "e_bat": 0.4,
                "soc": 0.525,
                "purchase_cost": 0.195,
                "export_subsidy": 0.0,
                "objective_total": 0.195,
            },
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "agent_id": 1,
                "agent_profile": "agent_1",
                "load": 1.2,
                "load_pred": 1.2,
                "pv": 0.0,
                "pv_pred": 0.0,
                "e_bat": 0.2,
                "soc": 0.5125,
                "purchase_cost": 0.135,
                "export_subsidy": 0.0,
                "objective_total": 0.135,
            },
        ]
    )
    grid_df = pd.DataFrame(
        [
            {"controller": controller, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": timestamp, "bus_id": 0, "vm_pu": 1.0, "is_agent_bus": False},
            {"controller": controller, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": timestamp, "bus_id": 1, "vm_pu": 0.99, "is_agent_bus": True},
            {"controller": controller, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": timestamp, "bus_id": 2, "vm_pu": 1.01, "is_agent_bus": True},
        ]
    )
    summary = pd.DataFrame(
        [
            {"controller": controller, "agent_profile": "agent_0", "purchase_cost": 0.195, "export_subsidy": 0.0, "objective_total": 0.195},
            {"controller": controller, "agent_profile": "agent_1", "purchase_cost": 0.135, "export_subsidy": 0.0, "objective_total": 0.135},
        ]
    )
    meta = {
        "controller": controller,
        "agent_profiles": ["agent_0", "agent_1"],
        "agent_bus_ids": [1, 2],
        "v_min_pu": 0.95,
        "v_max_pu": 1.05,
        "prediction_mode": "normal",
        "forecast_backend": forecast_backend,
        "import_price_markup_eur_per_kwh": 0.2,
        "export_subsidy_eur_per_kwh": 0.079,
        "economics_scope": "agent_only",
    }
    return grid_nb.RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary, meta=meta)


def test_build_admm_mpc_window_data_shapes_and_import_price():
    cfg = _make_cfg(future_horizon=4)
    env = _make_env(future_horizon=4)
    raw_obs = env.obs_builder.build_raw(env)

    data = admm_mpc_nb.build_admm_mpc_window_data(cfg, env, raw_obs)

    assert data.wholesale_price_seq.shape == (5,)
    assert data.load_seq.shape == (2, 5)
    assert data.pv_seq.shape == (2, 5)
    assert np.allclose(data.import_price_eur_per_kwh, data.wholesale_price_seq + 0.2)
    assert np.allclose(data.energy_ref_kwh, np.full((2,), 2.0, dtype=np.float32))


def test_resolve_default_terminal_cost_weight_matches_formula():
    data = _make_solver_problem(horizon=6, n_agents=2)
    weights = admm_mpc_nb.resolve_default_terminal_cost_weight(data, multiplier=1.5)

    expected = 1.5 * 2.0 * float(np.mean(data.import_price_eur_per_kwh)) / 4.0
    assert weights.shape == (2,)
    assert np.allclose(weights, np.full((2,), expected, dtype=np.float32))


def test_duplicate_last_shift_rules_for_primal_z_and_u():
    window_data = _make_solver_problem(horizon=4, n_agents=2)
    warm_start_cache = admm_mpc_nb.AdmmMpcWarmStartCache(
        horizon_steps=4,
        charge_kw=np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]], dtype=np.float32),
        discharge_kw=np.asarray([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
        pv_curtail_kw=np.asarray([[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32),
        net_load_kw=np.asarray([[5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0]], dtype=np.float32),
        n_pos_kw=np.asarray([[5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0]], dtype=np.float32),
        energy_kwh=np.asarray([[2.0, 2.5, 3.0, 3.5, 3.5], [2.0, 1.5, 1.0, 0.5, 0.5]], dtype=np.float32),
        z_kw=np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32),
        u_kw=np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32),
    )

    local_warm = admm_mpc_nb._make_local_warm_start_cache(warm_start_cache, window_data=window_data)
    assert np.allclose(local_warm[0].charge_kw, np.asarray([2.0, 3.0, 4.0, 4.0], dtype=np.float32))
    assert np.allclose(local_warm[0].net_load_kw, np.asarray([6.0, 7.0, 8.0, 8.0], dtype=np.float32))
    assert np.allclose(local_warm[0].energy_kwh, np.asarray([2.0, 3.0, 3.5, 3.5, 3.5], dtype=np.float32))
    assert np.allclose(admm_mpc_nb._duplicate_last_shift_2d(warm_start_cache.z_kw)[0], np.asarray([2.0, 3.0, 4.0, 4.0], dtype=np.float32))
    assert np.allclose(admm_mpc_nb._duplicate_last_shift_2d(warm_start_cache.u_kw)[1], np.asarray([0.3, 0.2, 0.1, 0.1], dtype=np.float32))


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_terminal_cost_prevents_terminal_emptying():
    window_data = _make_solver_problem(horizon=8, n_agents=1)
    surrogate_cache = _make_surrogate_cache(horizon=8, n_agents=1)

    no_terminal = admm_mpc_nb.solve_admm_mpc_window(
        window_data,
        surrogate_cache,
        terminal_cost_weight_eur_per_kwh2=np.zeros((1,), dtype=np.float32),
        warm_start_cache=None,
        rho_init=0.05,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=20,
        max_iters_first_step=40,
        primal_tol=1e-4,
        dual_tol=1e-4,
    )
    with_terminal = admm_mpc_nb.solve_admm_mpc_window(
        window_data,
        surrogate_cache,
        terminal_cost_weight_eur_per_kwh2=admm_mpc_nb.resolve_default_terminal_cost_weight(window_data),
        warm_start_cache=None,
        rho_init=0.05,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=20,
        max_iters_first_step=40,
        primal_tol=1e-4,
        dual_tol=1e-4,
    )

    no_terminal_last4_mean = float(np.mean(no_terminal.energy_kwh[:, -4:]))
    with_terminal_last4_mean = float(np.mean(with_terminal.energy_kwh[:, -4:]))

    assert no_terminal_last4_mean <= 0.4
    assert with_terminal_last4_mean >= 1.5
    assert with_terminal_last4_mean - no_terminal_last4_mean >= 1.0


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_rho_is_clamped_in_solver():
    window_data = _make_solver_problem(horizon=6, n_agents=1)
    surrogate_cache = _make_surrogate_cache(horizon=6, n_agents=1)

    result = admm_mpc_nb.solve_admm_mpc_window(
        window_data,
        surrogate_cache,
        terminal_cost_weight_eur_per_kwh2=admm_mpc_nb.resolve_default_terminal_cost_weight(window_data),
        warm_start_cache=None,
        rho_init=1e9,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=1,
        max_iters_first_step=1,
        primal_tol=1e-4,
        dual_tol=1e-4,
    )

    assert 1e-3 <= float(result.rho_final) <= 1e3


def test_collect_admm_mpc_rollout_sets_meta_and_step_diagnostics(monkeypatch):
    cfg = _make_cfg(future_horizon=4, n_agents=2)

    fake_step_result = admm_mpc_nb.AdmmMpcStepResult(
        executed_charge_kw=np.asarray([0.4, 0.2], dtype=np.float32),
        executed_discharge_kw=np.zeros((2,), dtype=np.float32),
        executed_pv_curtail_kw=np.asarray([0.1, 0.0], dtype=np.float32),
        executed_net_load_kw=np.asarray([1.3, 0.9], dtype=np.float32),
        executed_action_array=np.asarray([[0.2, 0.8], [0.1, 1.0]], dtype=np.float32),
        full_horizon_solution=admm_mpc_nb.AdmmMpcWindowResult(
            charge_kw=np.asarray([[0.4, 0.0], [0.2, 0.0]], dtype=np.float32),
            discharge_kw=np.zeros((2, 2), dtype=np.float32),
            pv_curtail_kw=np.asarray([[0.1, 0.0], [0.0, 0.0]], dtype=np.float32),
            pv_effective_kw=np.asarray([[0.4, 0.2], [0.2, 0.1]], dtype=np.float32),
            net_load_kw=np.asarray([[1.3, 1.0], [0.9, 0.8]], dtype=np.float32),
            grid_import_kw=np.asarray([[1.3, 1.0], [0.9, 0.8]], dtype=np.float32),
            grid_export_kw=np.zeros((2, 2), dtype=np.float32),
            energy_kwh=np.asarray([[2.0, 2.1, 2.1], [2.0, 2.05, 2.05]], dtype=np.float32),
            surrogate_root_p_kw=np.asarray([2.2, 1.8], dtype=np.float32),
            baseline_root_p_kw=np.asarray([2.0, 1.7], dtype=np.float32),
            objective_eur=0.11,
            converged=True,
            iterations=7,
            final_primal_residual=1e-4,
            final_dual_residual=2e-4,
            solve_time_sec=0.05,
            rho_final=0.2,
            solver_status="admm_mpc_converged",
            history_df=pd.DataFrame(),
        ),
        converged=True,
        iterations=7,
        final_primal_residual=1e-4,
        final_dual_residual=2e-4,
        solve_time_sec=0.05,
        rho_final=0.2,
        warm_start_cache=admm_mpc_nb.AdmmMpcWarmStartCache(horizon_steps=2),
    )

    monkeypatch.setattr(
        admm_mpc_nb,
        "build_admm_mpc_surrogate_cache",
        lambda cfg_arg, env, horizon_steps: admm_mpc_nb.AdmmMpcSurrogateCache(
            trafo_limit_kw=10.0,
            trafo_base_kw=0.0,
            alpha_netload_kw=np.ones((env.n,), dtype=np.float32),
            alpha_netload_window_kw=np.ones((env.n, horizon_steps), dtype=np.float32),
            horizon_steps=int(horizon_steps),
        ),
    )
    monkeypatch.setattr(admm_mpc_nb, "run_admm_mpc_step", lambda *args, **kwargs: fake_step_result)

    def _fake_collect_controller_rollout(cfg_arg, *, label, controller_builder, **kwargs):
        del kwargs
        env = _make_env(n_agents=2, future_horizon=cfg_arg.env.future_horizon)
        controller = controller_builder(env)
        controller.reset()
        controller.act(None)
        timestamp = pd.Timestamp("2020-06-01 00:00:00")
        step_df = pd.DataFrame(
            [
                {
                    "controller": label,
                    "episode_idx": 0,
                    "step": 0,
                    "global_step": 0,
                    "timestamp": timestamp,
                    "wholesale_price": 0.1,
                    "import_price": 0.3,
                    "wholesale_price_pred": 0.3,
                    "import_price_pred": 0.5,
                    "base_net_load_total": 2.0,
                    "base_net_load_effective_total": 1.9,
                    "net_load_total": 2.2,
                    "fixed_load_kw": 0.0,
                    "fixed_generation_kw": 0.0,
                    "feeder_raw_net_load_kw": 2.0,
                    "feeder_effective_net_load_kw": 1.9,
                    "feeder_post_action_net_load_kw": 2.2,
                    "load_total": 2.4,
                    "pv_raw_total": 0.5,
                    "pv_effective_total": 0.4,
                    "pv_curtail_total": 0.1,
                    "grid_import_total": 2.2,
                    "grid_export_total": 0.0,
                    "battery_charge_total": 0.6,
                    "battery_discharge_total": 0.0,
                    "purchase_cost_total": 0.33,
                    "export_subsidy_total": 0.0,
                    "objective_total": 0.33,
                    "pp_root_p_kw": 2.2,
                }
            ]
        )
        agent_df = pd.DataFrame(
            [
                {
                    "controller": label,
                    "episode_idx": 0,
                    "step": 0,
                    "global_step": 0,
                    "timestamp": timestamp,
                    "agent_id": 0,
                    "agent_profile": "agent_0",
                    "load": 1.2,
                    "load_pred": 1.2,
                    "pv": 0.5,
                    "pv_raw": 0.5,
                    "pv_effective": 0.4,
                    "pv_curtail": 0.1,
                    "pv_pred": 0.5,
                    "base_net_load": 0.7,
                    "base_net_load_effective": 0.8,
                    "net_load": 1.3,
                    "grid_import_kw": 1.3,
                    "grid_export_kw": 0.0,
                    "e_bat": 0.4,
                    "soc": 0.525,
                    "purchase_cost": 0.195,
                    "export_subsidy": 0.0,
                    "objective_total": 0.195,
                },
                {
                    "controller": label,
                    "episode_idx": 0,
                    "step": 0,
                    "global_step": 0,
                    "timestamp": timestamp,
                    "agent_id": 1,
                    "agent_profile": "agent_1",
                    "load": 1.2,
                    "load_pred": 1.2,
                    "pv": 0.0,
                    "pv_raw": 0.0,
                    "pv_effective": 0.0,
                    "pv_curtail": 0.0,
                    "pv_pred": 0.0,
                    "base_net_load": 1.2,
                    "base_net_load_effective": 1.1,
                    "net_load": 0.9,
                    "grid_import_kw": 0.9,
                    "grid_export_kw": 0.0,
                    "e_bat": 0.2,
                    "soc": 0.5125,
                    "purchase_cost": 0.135,
                    "export_subsidy": 0.0,
                    "objective_total": 0.135,
                },
            ]
        )
        grid_df = pd.DataFrame(
            [
                {"controller": label, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": timestamp, "bus_id": 0, "vm_pu": 1.0, "is_agent_bus": False},
                {"controller": label, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": timestamp, "bus_id": 1, "vm_pu": 0.99, "is_agent_bus": True},
                {"controller": label, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": timestamp, "bus_id": 2, "vm_pu": 1.01, "is_agent_bus": True},
            ]
        )
        summary = (
            agent_df.groupby(["controller", "agent_profile"], as_index=False)[["purchase_cost", "export_subsidy", "objective_total"]].sum()
        )
        meta = {
            "controller": label,
            "agent_profiles": ["agent_0", "agent_1"],
            "agent_bus_ids": [1, 2],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
            "import_price_markup_eur_per_kwh": 0.2,
            "trafo_limit_kw": 10.0,
        }
        return grid_nb.RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary, meta=meta)

    monkeypatch.setattr(grid_nb, "collect_controller_rollout", _fake_collect_controller_rollout)

    rollout = admm_mpc_nb.collect_admm_mpc_rollout(cfg, prediction_mode="normal")

    assert rollout.meta["controller"] == admm_mpc_nb.ADMM_MPC_LSTM_LABEL
    assert rollout.meta["prediction_mode"] == "normal"
    assert rollout.meta["forecast_backend"] == "lstm"
    assert rollout.meta["economics_scope"] == "agent_only"
    assert rollout.meta["admm_terminal_cost_mode"] == "quadratic_to_soc_target"
    assert bool(rollout.step_df.loc[0, "admm_converged"])
    assert int(rollout.step_df.loc[0, "admm_iterations"]) == 7
    assert float(rollout.step_df.loc[0, "wholesale_price_pred"]) == pytest.approx(0.3)
    assert float(rollout.step_df.loc[0, "import_price_pred"]) == pytest.approx(0.5)
    assert float(rollout.step_df.loc[0, "import_price"]) == pytest.approx(0.3)
    assert rollout.step_df.loc[0, "objective_total"] == pytest.approx(
        float(rollout.step_df.loc[0, "purchase_cost_total"] - rollout.step_df.loc[0, "export_subsidy_total"])
    )


def test_admm_mpc_controller_keeps_progress_bar_enabled_for_notebooks(monkeypatch):
    cfg = _make_cfg(future_horizon=4, n_agents=2)
    env = _make_env(n_agents=2, future_horizon=4)
    created_progress_bars = []

    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            del args
            self.kwargs = dict(kwargs)
            self.closed = False
            created_progress_bars.append(self)

        def update(self, amount):
            del amount

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "tqdm.auto", SimpleNamespace(tqdm=DummyTqdm))
    monkeypatch.setattr(
        admm_mpc_nb,
        "build_admm_mpc_surrogate_cache",
        lambda cfg_arg, env_arg, horizon_steps: admm_mpc_nb.AdmmMpcSurrogateCache(
            trafo_limit_kw=10.0,
            trafo_base_kw=0.0,
            alpha_netload_kw=np.ones((env_arg.n,), dtype=np.float32),
            alpha_netload_window_kw=np.ones((env_arg.n, horizon_steps), dtype=np.float32),
            horizon_steps=int(horizon_steps),
        ),
    )

    controller = admm_mpc_nb._AdmmMpcController(
        env,
        cfg,
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=100,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
        show_progress=True,
    )
    try:
        assert len(created_progress_bars) == 1
        assert created_progress_bars[0].kwargs["desc"] == "ADMM MPC rollout"
        assert created_progress_bars[0].kwargs["disable"] is False
        assert created_progress_bars[0].kwargs["total"] == env.num_available_episodes * env.episode_length
    finally:
        controller.close()
    assert created_progress_bars[0].closed is True
    assert created_progress_bars[0].kwargs["leave"] is True


def test_admm_mpc_controller_falls_back_to_stdout_progress_when_tqdm_unavailable(monkeypatch, capsys):
    cfg = _make_cfg(future_horizon=4, n_agents=2)
    env = _make_env(n_agents=2, future_horizon=4)
    env.episode_length = 1
    original_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm.auto":
            raise ImportError("tqdm disabled for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    monkeypatch.setattr(
        admm_mpc_nb,
        "build_admm_mpc_surrogate_cache",
        lambda cfg_arg, env_arg, horizon_steps: admm_mpc_nb.AdmmMpcSurrogateCache(
            trafo_limit_kw=10.0,
            trafo_base_kw=0.0,
            alpha_netload_kw=np.ones((env_arg.n,), dtype=np.float32),
            alpha_netload_window_kw=np.ones((env_arg.n, horizon_steps), dtype=np.float32),
            horizon_steps=int(horizon_steps),
        ),
    )
    monkeypatch.setattr(
        admm_mpc_nb,
        "build_admm_mpc_window_data",
        lambda *args, **kwargs: admm_mpc_nb.AdmmMpcWindowData(
            wholesale_price_seq=np.asarray([0.1], dtype=np.float32),
            wholesale_price_eur_per_kwh=np.asarray([0.1], dtype=np.float32),
            import_price_eur_per_kwh=np.asarray([0.3], dtype=np.float32),
            load_seq=np.zeros((2, 1), dtype=np.float32),
            pv_seq=np.zeros((2, 1), dtype=np.float32),
            battery_capacity_kwh=np.ones((2,), dtype=np.float32),
            p_max_kw=np.ones((2,), dtype=np.float32),
            eff_charge=np.ones((2,), dtype=np.float32),
            eff_discharge=np.ones((2,), dtype=np.float32),
            energy_init_kwh=np.ones((2,), dtype=np.float32),
            energy_ref_kwh=np.ones((2,), dtype=np.float32),
            energy_min_kwh=np.zeros((2,), dtype=np.float32),
            energy_max_kwh=np.ones((2,), dtype=np.float32),
            export_subsidy_eur_per_kwh=0.079,
            import_price_markup_eur_per_kwh=0.2,
            dt_hours=0.25,
        ),
    )
    monkeypatch.setattr(
        admm_mpc_nb,
        "run_admm_mpc_step",
        lambda *args, **kwargs: admm_mpc_nb.AdmmMpcStepResult(
            executed_charge_kw=np.zeros((2,), dtype=np.float32),
            executed_discharge_kw=np.zeros((2,), dtype=np.float32),
            executed_pv_curtail_kw=np.zeros((2,), dtype=np.float32),
            executed_net_load_kw=np.zeros((2,), dtype=np.float32),
            executed_action_array=np.zeros((2, 2), dtype=np.float32),
            full_horizon_solution=admm_mpc_nb.AdmmMpcWindowResult(
                charge_kw=np.zeros((2, 1), dtype=np.float32),
                discharge_kw=np.zeros((2, 1), dtype=np.float32),
                pv_curtail_kw=np.zeros((2, 1), dtype=np.float32),
                pv_effective_kw=np.zeros((2, 1), dtype=np.float32),
                net_load_kw=np.zeros((2, 1), dtype=np.float32),
                grid_import_kw=np.zeros((2, 1), dtype=np.float32),
                grid_export_kw=np.zeros((2, 1), dtype=np.float32),
                energy_kwh=np.zeros((2, 2), dtype=np.float32),
                surrogate_root_p_kw=np.zeros((1,), dtype=np.float32),
                baseline_root_p_kw=np.zeros((1,), dtype=np.float32),
                objective_eur=0.0,
                converged=True,
                iterations=1,
                final_primal_residual=0.0,
                final_dual_residual=0.0,
                solve_time_sec=0.01,
                rho_final=0.2,
                solver_status="optimal",
                history_df=pd.DataFrame(),
            ),
            converged=True,
            iterations=1,
            final_primal_residual=0.0,
            final_dual_residual=0.0,
            solve_time_sec=0.01,
            rho_final=0.2,
            warm_start_cache=admm_mpc_nb.AdmmMpcWarmStartCache(horizon_steps=1),
        ),
    )
    monkeypatch.setattr(admm_mpc_nb, "build_safety_local_numpy", lambda *args, **kwargs: np.zeros((2, 2), dtype=np.float32))
    monkeypatch.setattr(
        admm_mpc_nb,
        "compute_action_gap_metrics_numpy",
        lambda *args, **kwargs: {"action_gap_abs": np.zeros((2,), dtype=np.float32)},
    )

    controller = admm_mpc_nb._AdmmMpcController(
        env,
        cfg,
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=100,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
        show_progress=True,
    )
    try:
        controller.reset()
        env.cur_step = 0
        controller.act(None)
    finally:
        controller.close()
    captured = capsys.readouterr()
    assert "ADMM MPC rollout: starting episode/day" in captured.out
    assert "ADMM MPC rollout progress:" in captured.out
    assert "ADMM MPC rollout complete:" in captured.out


def test_plotting_helpers_accept_admm_mpc_style_rollout(monkeypatch):
    cfg = _make_cfg(future_horizon=4, n_agents=2)

    monkeypatch.setattr(
        admm_mpc_nb,
        "collect_admm_mpc_rollout",
        lambda *args, **kwargs: grid_nb.RolloutResult(
            step_df=pd.DataFrame(
                    [
                        {
                            "controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL,
                            "episode_idx": 0,
                            "step": 0,
                        "global_step": 0,
                        "timestamp": pd.Timestamp("2020-06-01 00:00:00"),
                        "wholesale_price": 0.1,
                        "import_price": 0.3,
                            "wholesale_price_pred": 0.3,
                            "import_price_pred": 0.5,
                            "base_net_load_total": 1.9,
                            "net_load_total": 2.2,
                            "agent_raw_net_load_kw": 1.9,
                            "agent_effective_net_load_kw": 1.9,
                            "agent_post_action_net_load_kw": 2.2,
                            "battery_charge_total": 0.6,
                            "battery_discharge_total": 0.0,
                            "load_total": 2.4,
                            "pv_raw_total": 0.5,
                        "grid_import_total": 2.2,
                        "grid_export_total": 0.0,
                        "pv_curtail_total": 0.1,
                        "feeder_raw_net_load_kw": 2.0,
                        "feeder_effective_net_load_kw": 1.9,
                        "feeder_post_action_net_load_kw": 2.2,
                        "pp_root_p_kw": 2.2,
                        "purchase_cost_total": 0.33,
                        "export_subsidy_total": 0.0,
                        "objective_total": 0.33,
                    }
                ]
            ),
            agent_df=pd.DataFrame(
                [
                    {
                        "controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL,
                        "episode_idx": 0,
                        "step": 0,
                        "global_step": 0,
                        "timestamp": pd.Timestamp("2020-06-01 00:00:00"),
                        "agent_id": 0,
                        "agent_profile": "agent_0",
                        "load": 1.2,
                        "load_pred": 1.2,
                        "pv": 0.5,
                        "pv_pred": 0.5,
                        "e_bat": 0.4,
                        "soc": 0.525,
                    },
                    {
                        "controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL,
                        "episode_idx": 0,
                        "step": 0,
                        "global_step": 0,
                        "timestamp": pd.Timestamp("2020-06-01 00:00:00"),
                        "agent_id": 1,
                        "agent_profile": "agent_1",
                        "load": 1.2,
                        "load_pred": 1.2,
                        "pv": 0.0,
                        "pv_pred": 0.0,
                        "e_bat": 0.2,
                        "soc": 0.5125,
                    },
                ]
            ),
            grid_df=pd.DataFrame(
                [
                    {"controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": pd.Timestamp("2020-06-01 00:00:00"), "bus_id": 0, "vm_pu": 1.0, "is_agent_bus": False},
                    {"controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": pd.Timestamp("2020-06-01 00:00:00"), "bus_id": 1, "vm_pu": 0.99, "is_agent_bus": True},
                    {"controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL, "episode_idx": 0, "step": 0, "global_step": 0, "timestamp": pd.Timestamp("2020-06-01 00:00:00"), "bus_id": 2, "vm_pu": 1.01, "is_agent_bus": True},
                ]
            ),
            summary=pd.DataFrame(
                [
                    {"controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL, "agent_profile": "agent_0", "purchase_cost": 0.195, "export_subsidy": 0.0, "objective_total": 0.195},
                    {"controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL, "agent_profile": "agent_1", "purchase_cost": 0.135, "export_subsidy": 0.0, "objective_total": 0.135},
                ]
            ),
            meta={
                "controller": admm_mpc_nb.ADMM_MPC_LSTM_LABEL,
                "agent_profiles": ["agent_0", "agent_1"],
                "agent_bus_ids": [1, 2],
                "v_min_pu": 0.95,
                "v_max_pu": 1.05,
                "prediction_mode": "normal",
                "forecast_backend": "lstm",
            },
        ),
    )

    rollout = admm_mpc_nb.collect_admm_mpc_rollout(cfg, prediction_mode="normal")
    fig_1 = grid_nb.plot_test_rollout(rollout)
    fig_2 = grid_nb.plot_net_load_comparison(rollout)
    fig_3 = grid_nb.plot_battery_power_and_soc_comparison(rollout)

    assert fig_1 is not None
    assert fig_2 is not None
    assert fig_3 is not None


def test_compare_helpers_accept_admm_mpc_and_local_mpc_rollouts(monkeypatch):
    import matplotlib.pyplot as plt

    cfg = _make_cfg(future_horizon=4, n_agents=2)
    timestamps = pd.date_range("2020-06-01", periods=2, freq="15min")

    def _fake_collect_controller_rollout(local_cfg, *, label, **kwargs):
        del kwargs
        if "ADMM MPC" in str(label):
            wholesale_price_pred = [0.10, 0.20]
            import_price_pred = [0.30, 0.40]
            wholesale_price = [-0.10, 0.00]
            import_price = [0.10, 0.20]
            controller = admm_mpc_nb.ADMM_MPC_LSTM_LABEL
        else:
            wholesale_price_pred = [0.10, 0.20]
            import_price_pred = [0.30, 0.40]
            wholesale_price = [-0.10, 0.00]
            import_price = [0.10, 0.20]
            controller = "Local MPC + LSTM Forecast"

        step_df = pd.DataFrame(
            {
                "controller": [controller, controller],
                "episode_idx": [0, 0],
                "step": [0, 1],
                "global_step": [0, 1],
                "timestamp": timestamps,
                "wholesale_price": wholesale_price,
                "import_price": import_price,
                "wholesale_price_pred": wholesale_price_pred,
                "import_price_pred": import_price_pred,
                "base_net_load_total": [2.0, 2.1],
                "base_net_load_effective_total": [1.9, 2.0],
                "net_load_total": [1.6, 1.7],
                "fixed_load_kw": [0.0, 0.0],
                "fixed_generation_kw": [0.0, 0.0],
                "feeder_raw_net_load_kw": [2.0, 2.1],
                "feeder_effective_net_load_kw": [1.9, 2.0],
                "feeder_post_action_net_load_kw": [1.6, 1.7],
                "load_total": [2.4, 2.5],
                "pv_raw_total": [0.6, 0.6],
                "pv_effective_total": [0.4, 0.45],
                "pv_curtail_total": [0.2, 0.15],
                "grid_import_total": [1.6, 1.7],
                "grid_export_total": [0.0, 0.0],
                "battery_charge_total": [0.3, 0.25],
                "battery_discharge_total": [0.0, 0.0],
                "purchase_cost_total": [0.48, 0.68],
                "export_subsidy_total": [0.0, 0.0],
                "objective_total": [0.48, 0.68],
                "pp_root_p_kw": [1.6, 1.7],
                "trafo_loading_pct_max": [55.0, 58.0],
                "n_trafo_violations": [0, 0],
                "soc_penalty_total": [0.0, 0.0],
                "voltage_penalty_total": [0.0, 0.0],
                "line_penalty_total": [0.0, 0.0],
                "trafo_penalty_total": [0.0, 0.0],
            }
        )
        agent_rows = []
        for step_idx, timestamp in enumerate(timestamps):
            for agent_idx in range(2):
                agent_rows.append(
                    {
                        "controller": controller,
                        "episode_idx": 0,
                        "step": step_idx,
                        "global_step": step_idx,
                        "timestamp": timestamp,
                        "agent_id": agent_idx,
                        "agent_profile": f"agent_{agent_idx}",
                        "load": 1.2 + 0.1 * step_idx,
                        "load_pred": 1.2 + 0.1 * step_idx,
                        "pv": 0.3,
                        "pv_raw": 0.3,
                        "pv_effective": 0.2,
                        "pv_curtail": 0.1,
                        "pv_pred": 0.3,
                        "base_net_load": 0.9,
                        "base_net_load_effective": 0.8,
                        "net_load": 0.8,
                        "grid_import_kw": 0.8,
                        "grid_export_kw": 0.0,
                        "e_bat": 0.15,
                        "soc": 0.5 + 0.01 * step_idx,
                        "purchase_cost": 0.24 + 0.01 * step_idx,
                        "export_subsidy": 0.0,
                        "objective_total": 0.24 + 0.01 * step_idx,
                    }
                )
        agent_df = pd.DataFrame(agent_rows)
        grid_df = pd.DataFrame(
            [
                {
                    "controller": controller,
                    "episode_idx": step_idx // 2,
                    "step": step_idx,
                    "global_step": step_idx,
                    "timestamp": timestamp,
                    "bus_id": bus_id,
                    "vm_pu": vm_pu,
                    "is_agent_bus": bus_id in {1, 2},
                }
                for step_idx, timestamp in enumerate(timestamps)
                for bus_id, vm_pu in [(0, 1.0), (1, 0.99), (2, 1.01)]
            ]
        )
        summary = (
            agent_df.groupby(["controller", "agent_profile"], as_index=False)[
                ["purchase_cost", "export_subsidy", "objective_total"]
            ].sum()
        )
        return grid_nb.RolloutResult(
            step_df=step_df,
            agent_df=agent_df,
            grid_df=grid_df,
            summary=summary,
            meta={
                "controller": controller,
                "agent_profiles": ["agent_0", "agent_1"],
                "agent_bus_ids": [1, 2],
                "v_min_pu": 0.95,
                "v_max_pu": 1.05,
                "prediction_mode": "normal",
                "forecast_backend": str(local_cfg.forecast.type),
                "trafo_limit_kw": 10.0,
                "import_price_markup_eur_per_kwh": 0.2,
            },
        )

    monkeypatch.setattr(grid_nb, "collect_controller_rollout", _fake_collect_controller_rollout)

    admm_rollout = admm_mpc_nb.collect_admm_mpc_rollout(cfg, prediction_mode="normal", show_progress=False)
    single_agent_rollout = grid_nb.collect_local_mpc_rollout(
        cfg,
        prediction_mode="normal",
        label="Local MPC + LSTM Forecast",
    )

    metrics_df = grid_nb.compare_rollout_metrics(admm_rollout, single_agent_rollout)
    banner = grid_nb.build_compare_warning_banner(admm_rollout, single_agent_rollout)
    fig_price = grid_nb.plot_price_prediction_comparison(admm_rollout, single_agent_rollout)
    fig_power = grid_nb.plot_power_balance_comparison(admm_rollout, single_agent_rollout)
    fig_voltage = grid_nb.plot_voltage_profile_comparison(admm_rollout, single_agent_rollout)
    fig_net = grid_nb.plot_net_load_comparison(admm_rollout, single_agent_rollout)
    fig_battery = grid_nb.plot_battery_power_and_soc_comparison(admm_rollout, single_agent_rollout)

    assert not metrics_df.empty
    assert banner is not None
    assert fig_price is not None
    assert fig_power is not None
    assert fig_voltage is not None
    assert fig_net is not None
    assert fig_battery is not None
    for figure in [fig_price, fig_power, fig_voltage, fig_net, fig_battery]:
        plt.close(figure)


def test_admm_mpc_rollout_package_round_trip(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")

    package = admm_mpc_nb.build_admm_mpc_rollout_package(
        rollout,
        controller_label=rollout.meta["controller"],
        cfg=cfg,
        prediction_mode="normal",
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=100,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
        extra_meta={"rollout_meta": dict(rollout.meta)},
        diagnostic_summary={"convergence_rate": 1.0},
    )
    saved_dir = admm_mpc_nb.save_admm_mpc_rollout_package(package, tmp_path / "cached_rollout")
    loaded = admm_mpc_nb.load_admm_mpc_rollout_package(saved_dir)

    assert int(loaded["manifest"]["rollout_package_version"]) == 2
    assert loaded["diagnostics"]["convergence_rate"] == pytest.approx(1.0)
    assert pd.api.types.is_datetime64_any_dtype(loaded["step_df"]["timestamp"])
    assert list(loaded["agent_df"]["agent_profile"]) == ["agent_0", "agent_1"]
    assert float(loaded["summary_df"]["objective_total"].sum()) == pytest.approx(0.33)


def test_replay_admm_mpc_rollout_package_returns_compare_ready_rollout(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = admm_mpc_nb.save_admm_mpc_rollout_package(
        admm_mpc_nb.build_admm_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=100,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
            extra_meta={"rollout_meta": dict(rollout.meta)},
            diagnostic_summary={"avg_admm_iterations": 7.0},
        ),
        tmp_path / "cached_rollout",
    )

    replay = admm_mpc_nb.replay_admm_mpc_rollout_package(
        cfg,
        saved_dir,
        label="Replayed ADMM MPC",
        prediction_mode="normal",
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=100,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
    )

    replay_rollout = replay["rollout"]
    assert replay_rollout.meta["loaded_from_cached_rollout"] is True
    assert replay_rollout.meta["rollout_package_dir"] == str(saved_dir.resolve())
    assert replay_rollout.meta["cached_rollout_diagnostics"]["avg_admm_iterations"] == pytest.approx(7.0)
    assert replay_rollout.meta["controller"] == "Replayed ADMM MPC"
    assert replay_rollout.step_df["controller"].unique().tolist() == ["Replayed ADMM MPC"]
    assert replay_rollout.summary["controller"].unique().tolist() == ["Replayed ADMM MPC"]


def test_replay_admm_mpc_rollout_package_rejects_cfg_snapshot_mismatch(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = admm_mpc_nb.save_admm_mpc_rollout_package(
        admm_mpc_nb.build_admm_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=100,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
            extra_meta={"rollout_meta": dict(rollout.meta)},
        ),
        tmp_path / "cached_rollout",
    )

    mismatched_cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-03")
    mismatched_cfg.env.episode_limit = 96
    mismatched_cfg.forecast.type = "lstm"
    with pytest.raises(ValueError, match="ADMM MPC rollout package mismatch"):
        admm_mpc_nb.replay_admm_mpc_rollout_package(
            mismatched_cfg,
            saved_dir,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=100,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
        )


def test_replay_admm_mpc_rollout_package_rejects_solver_fingerprint_mismatch(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = admm_mpc_nb.save_admm_mpc_rollout_package(
        admm_mpc_nb.build_admm_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=100,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
            extra_meta={"rollout_meta": dict(rollout.meta)},
        ),
        tmp_path / "cached_rollout",
    )

    with pytest.raises(ValueError, match="admm_solver_fingerprint mismatch at 'max_iters'"):
        admm_mpc_nb.replay_admm_mpc_rollout_package(
            cfg,
            saved_dir,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=10,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
        )


def test_load_admm_mpc_rollout_package_rejects_version_mismatch(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = admm_mpc_nb.save_admm_mpc_rollout_package(
        admm_mpc_nb.build_admm_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=100,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
            extra_meta={"rollout_meta": dict(rollout.meta)},
        ),
        tmp_path / "cached_rollout",
    )
    manifest_path = saved_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rollout_package_version"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported ADMM MPC rollout package version"):
        admm_mpc_nb.load_admm_mpc_rollout_package(saved_dir)


def test_resolve_latest_compatible_admm_mpc_rollout_package_dir_returns_exact_dir(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")
    package = admm_mpc_nb.build_admm_mpc_rollout_package(
        rollout,
        controller_label=rollout.meta["controller"],
        cfg=cfg,
        prediction_mode="normal",
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=100,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
        extra_meta={"rollout_meta": dict(rollout.meta)},
    )

    prefix = tmp_path / "2020-06-01_2020-06-02_agents2_normal_lstm"
    base_dir = admm_mpc_nb.save_admm_mpc_rollout_package(package, prefix)

    resolved = admm_mpc_nb.resolve_latest_compatible_admm_mpc_rollout_package_dir(
        prefix,
        cfg=cfg,
        prediction_mode="normal",
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=100,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
    )

    assert base_dir.exists()
    assert resolved == base_dir.resolve()


def test_resolve_latest_compatible_admm_mpc_rollout_package_dir_requires_exact_dir(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=admm_mpc_nb.ADMM_MPC_LSTM_LABEL, forecast_backend="lstm")
    prefix = tmp_path / "2020-06-01_2020-06-02_agents2_normal_lstm"

    incompatible_package = admm_mpc_nb.build_admm_mpc_rollout_package(
        rollout,
        controller_label=rollout.meta["controller"],
        cfg=cfg,
        prediction_mode="normal",
        rho_init=None,
        rho_min=1e-3,
        rho_max=1e3,
        rho_adaptation="residual_balancing",
        max_iters=10,
        max_iters_first_step=300,
        primal_tol=1e-3,
        dual_tol=1e-3,
        terminal_cost_multiplier=1.0,
        extra_meta={"rollout_meta": dict(rollout.meta)},
    )
    admm_mpc_nb.save_admm_mpc_rollout_package(incompatible_package, tmp_path / f"{prefix.name}_badsolver")

    with pytest.raises(FileNotFoundError, match="requires one exact package directory") as exc_info:
        admm_mpc_nb.resolve_latest_compatible_admm_mpc_rollout_package_dir(
            prefix,
            cfg=cfg,
            prediction_mode="normal",
            rho_init=None,
            rho_min=1e-3,
            rho_max=1e3,
            rho_adaptation="residual_balancing",
            max_iters=100,
            max_iters_first_step=300,
            primal_tol=1e-3,
            dual_tol=1e-3,
            terminal_cost_multiplier=1.0,
        )

    message = str(exc_info.value)
    assert "2020-06-01_2020-06-02_agents2_normal_lstm_badsolver" in message
    assert "exact rollout package directory" in message
