"""Rolling ADMM-MPC helpers for the ADMM_mpc notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from controllers.action_feasibility import build_safety_local_numpy, compute_action_gap_metrics_numpy
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.grid_surrogate_notebook_helpers import (
    _battery_to_netload_sensitivity,
    _get_grid_projector,
    _projector_to_numpy,
)
from scripts.utils.price_protocol import (
    IMPORT_PRICE_COLUMN,
    IMPORT_PRICE_MARKUP_KEY,
    IMPORT_PRICE_PRED_COLUMN,
    WHOLESALE_PRICE_PRED_COLUMN,
    WHOLESALE_PRICE_SEQ_FIELD,
    derive_import_price_seq,
    get_import_price_markup,
)
from scripts.utils.admm_mpc_rollout_packages import (
    _augment_rollout_with_diagnostics,
    build_admm_mpc_rollout_package,
    load_admm_mpc_rollout_package,
    replay_admm_mpc_rollout_package,
    resolve_latest_compatible_admm_mpc_rollout_package_dir,
    save_admm_mpc_rollout_package,
)
from scripts.utils.admm_mpc_solver import (
    AdmmMpcStepResult,
    AdmmMpcSurrogateCache,
    AdmmMpcWarmStartCache,
    AdmmMpcWindowData,
    AdmmMpcWindowResult,
    _create_model,
    _compute_default_rho,
    _day_episode_length,
    _duplicate_last_shift_2d,
    _load_gurobi,
    _make_local_warm_start_cache,
    build_admm_mpc_surrogate_cache,
    build_admm_mpc_window_data,
    resolve_default_terminal_cost_weight,
    run_admm_mpc_step,
    solve_admm_mpc_window,
)


_THROUGHPUT_TIEBREAKER_EUR_PER_KWH = 1e-8
_NPOS_TIEBREAKER_EUR_PER_KWH = 1e-9
_ADMM_RESIDUAL_BALANCING_MU = 10.0
_ADMM_RESIDUAL_BALANCING_TAU = 2.0
ADMM_MPC_LSTM_LABEL = "ADMM MPC + LSTM Forecast"
ADMM_MPC_PERFECT_LABEL = "ADMM MPC + Perfect Forecast"



class _AdmmMpcController:
    def __init__(
        self,
        env,
        cfg,
        *,
        rho_init: float | None,
        rho_min: float,
        rho_max: float,
        rho_adaptation: str | None,
        max_iters: int,
        max_iters_first_step: int,
        primal_tol: float,
        dual_tol: float,
        terminal_cost_multiplier: float,
        show_progress: bool,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.rho_init = rho_init
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.rho_adaptation = rho_adaptation
        self.max_iters = int(max_iters)
        self.max_iters_first_step = int(max_iters_first_step)
        self.primal_tol = float(primal_tol)
        self.dual_tol = float(dual_tol)
        self.terminal_cost_multiplier = float(terminal_cost_multiplier)
        self.surrogate_cache = build_admm_mpc_surrogate_cache(
            cfg,
            env,
            horizon_steps=int(getattr(env, "future_horizon", cfg.env.future_horizon) + 1),
        )
        self.warm_start_cache: AdmmMpcWarmStartCache | None = None
        self.last_action_info: dict[str, np.ndarray] | None = None
        self.diagnostic_rows: list[dict[str, object]] = []
        self.apply_action_penalty = False
        self._show_progress = bool(show_progress)
        self._episode_idx = -1
        self._global_step = 0
        self._progress_bar = None
        if self._show_progress:
            try:
                from tqdm.auto import tqdm

                self._progress_bar = tqdm(
                    total=int(env.num_available_episodes) * int(env.episode_length),
                    desc="ADMM MPC rollout",
                    unit="step",
                    # tqdm.auto already chooses a notebook-friendly renderer when available.
                    disable=not self._show_progress,
                    leave=True,
                )
            except Exception:
                self._progress_bar = None

    def reset(self) -> None:
        self.warm_start_cache = None
        self.last_action_info = None
        self._episode_idx += 1
        if self._progress_bar is None and self._show_progress:
            total_episodes = int(getattr(self.env, "num_available_episodes", 1))
            print(f"ADMM MPC rollout: starting episode/day {self._episode_idx + 1}/{total_episodes}")

    def close(self) -> None:
        if self._progress_bar is not None:
            try:
                self._progress_bar.close()
            except Exception:
                pass
            self._progress_bar = None
        if self.warm_start_cache is not None and self.warm_start_cache.local_solvers is not None:
            for solver in self.warm_start_cache.local_solvers:
                dispose = getattr(solver, "dispose", None)
                if callable(dispose):
                    dispose()
            self.warm_start_cache.local_solvers = None
        if self._progress_bar is None and self._show_progress:
            total_steps = int(getattr(self.env, "num_available_episodes", 1)) * int(getattr(self.env, "episode_length", 0))
            print(f"ADMM MPC rollout complete: processed {self._global_step}/{total_steps} steps.")

    def _record_step_diagnostics(self, step_result: AdmmMpcStepResult) -> None:
        self.diagnostic_rows.append(
            {
                "episode_idx": int(self._episode_idx),
                "step": int(getattr(self.env, "cur_step", 0)),
                "global_step": int(self._global_step),
                "admm_converged": bool(step_result.converged),
                "admm_iterations": int(step_result.iterations),
                "admm_final_primal_residual": float(step_result.final_primal_residual),
                "admm_final_dual_residual": float(step_result.final_dual_residual),
                "admm_solve_time_sec": float(step_result.solve_time_sec),
                "admm_rho_final": float(step_result.rho_final),
            }
        )

    def act(self, obs, deterministic: bool = True):
        del obs, deterministic
        raw_obs = self.env.obs_builder.build_raw(self.env) if hasattr(self.env.obs_builder, "build_raw") else {}
        window_data = build_admm_mpc_window_data(self.cfg, self.env, raw_obs)
        terminal_weight = resolve_default_terminal_cost_weight(
            window_data,
            multiplier=float(self.terminal_cost_multiplier),
        )
        rho_value = (
            float(self.rho_init)
            if self.rho_init is not None
            else _compute_default_rho(window_data, self.surrogate_cache)
        )
        step_result = run_admm_mpc_step(
            self.env,
            raw_obs,
            self.cfg,
            surrogate_cache=self.surrogate_cache,
            warm_start_cache=self.warm_start_cache,
            rho_init=float(rho_value),
            rho_min=float(self.rho_min),
            rho_max=float(self.rho_max),
            rho_adaptation=self.rho_adaptation,
            max_iters=int(self.max_iters),
            max_iters_first_step=int(self.max_iters_first_step),
            primal_tol=float(self.primal_tol),
            dual_tol=float(self.dual_tol),
            terminal_cost_weight_eur_per_kwh2=terminal_weight,
        )
        self.warm_start_cache = step_result.warm_start_cache
        self._record_step_diagnostics(step_result)

        safety_local = build_safety_local_numpy(
            soc=np.asarray(self.env.soc, dtype=np.float32),
            load_raw=np.asarray(window_data.load_seq[:, 0], dtype=np.float32),
            pv_raw=np.asarray(window_data.pv_seq[:, 0], dtype=np.float32),
            battery_capacity_kwh=np.asarray(self.env.agent_c_bat, dtype=np.float32),
            p_max_kw=np.asarray(self.env.agent_p_max, dtype=np.float32),
        )
        action_info = compute_action_gap_metrics_numpy(
            safety_local,
            step_result.executed_action_array,
            step_result.executed_action_array,
        )
        action_info["solve_time_sec"] = np.float32(step_result.solve_time_sec)
        self.last_action_info = action_info
        actions = [step_result.executed_action_array[agent_idx].copy() for agent_idx in range(self.env.n)]
        if self._progress_bar is not None:
            try:
                self._progress_bar.set_postfix(
                    {
                        "episode/day": f"{self._episode_idx + 1}/{int(getattr(self.env, 'num_available_episodes', 1))}",
                        "step": int(getattr(self.env, "cur_step", 0)) + 1,
                        "iters": int(step_result.iterations),
                        "converged": bool(step_result.converged),
                    },
                    refresh=False,
                )
            except Exception:
                pass
            self._progress_bar.update(1)
        elif self._show_progress:
            episode_length = max(int(getattr(self.env, "episode_length", 1)), 1)
            current_step_in_episode = int(getattr(self.env, "cur_step", 0)) + 1
            if current_step_in_episode >= episode_length:
                print(
                    "ADMM MPC rollout progress: "
                    f"episode/day {self._episode_idx + 1}/{int(getattr(self.env, 'num_available_episodes', 1))}, "
                    f"step {self._global_step + 1}, iters={int(step_result.iterations)}, "
                    f"converged={bool(step_result.converged)}"
                )
        self._global_step += 1
        return actions


def collect_admm_mpc_rollout(
    cfg,
    *,
    prediction_mode: str = "normal",
    label: str | None = None,
    show_progress: bool = True,
    rho_init: float | None = None,
    rho_min: float = 1e-3,
    rho_max: float = 1e3,
    rho_adaptation: str | None = "residual_balancing",
    max_iters: int = 100,
    max_iters_first_step: int = 300,
    primal_tol: float = 1e-3,
    dual_tol: float = 1e-3,
    terminal_cost_multiplier: float = 1.0,
) -> grid_nb.RolloutResult:
    resolved_mode = grid_nb.normalize_prediction_mode(prediction_mode)
    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=resolved_mode)
    comparison_cfg.env.episode_limit = _day_episode_length(float(comparison_cfg.env.dt))
    resolved_label = (
        str(label)
        if label is not None
        else (ADMM_MPC_PERFECT_LABEL if resolved_mode == grid_nb.PERFECT_PREDICTION_MODE else ADMM_MPC_LSTM_LABEL)
    )
    controller_ref: dict[str, _AdmmMpcController] = {}

    def _controller_builder(env):
        controller = _AdmmMpcController(
            env,
            comparison_cfg,
            rho_init=rho_init,
            rho_min=float(rho_min),
            rho_max=float(rho_max),
            rho_adaptation=rho_adaptation,
            max_iters=int(max_iters),
            max_iters_first_step=int(max_iters_first_step),
            primal_tol=float(primal_tol),
            dual_tol=float(dual_tol),
            terminal_cost_multiplier=float(terminal_cost_multiplier),
            show_progress=bool(show_progress),
        )
        controller_ref["controller"] = controller
        return controller

    rollout = grid_nb.collect_controller_rollout(
        comparison_cfg,
        label=resolved_label,
        controller_builder=_controller_builder,
    )
    controller = controller_ref.get("controller")
    try:
        augmented = _augment_rollout_with_diagnostics(
            rollout,
            controller.diagnostic_rows if controller is not None else [],
            forecast_backend=str(comparison_cfg.forecast.type),
        )
        augmented.meta.update(
            {
                "controller": resolved_label,
                "prediction_mode": resolved_mode,
                "economics_scope": "agent_only",
                "soc_mode": "continuous",
                "future_horizon": int(comparison_cfg.env.future_horizon),
            }
        )
        return augmented
    finally:
        if controller is not None:
            controller.close()


__all__ = [
    "ADMM_MPC_LSTM_LABEL",
    "ADMM_MPC_PERFECT_LABEL",
    "AdmmMpcStepResult",
    "AdmmMpcSurrogateCache",
    "AdmmMpcWarmStartCache",
    "AdmmMpcWindowData",
    "AdmmMpcWindowResult",
    "build_admm_mpc_rollout_package",
    "build_admm_mpc_surrogate_cache",
    "build_admm_mpc_window_data",
    "collect_admm_mpc_rollout",
    "load_admm_mpc_rollout_package",
    "replay_admm_mpc_rollout_package",
    "resolve_default_terminal_cost_weight",
    "resolve_latest_compatible_admm_mpc_rollout_package_dir",
    "run_admm_mpc_step",
    "save_admm_mpc_rollout_package",
    "solve_admm_mpc_window",
]
