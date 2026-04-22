from __future__ import annotations

import numpy as np
import pandas as pd
from controllers.madrl.safety_projector import build_safety_local_numpy, compute_action_gap_metrics_numpy
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY, IMPORT_PRICE_PRED_COLUMN, WHOLESALE_PRICE_PRED_COLUMN, derive_import_price_seq
from scripts.utils.admm_mpc_solver import AdmmMpcStepResult, AdmmMpcSurrogateCache, AdmmMpcWindowData, _compute_default_rho, _day_episode_length, build_admm_mpc_surrogate_cache, build_admm_mpc_window_data, resolve_default_terminal_cost_weight, run_admm_mpc_step

ADMM_MPC_LSTM_LABEL = 'ADMM MPC + LSTM Forecast'
ADMM_MPC_PERFECT_LABEL = 'ADMM MPC + Perfect Forecast'

class _AdmmMpcController:

    def __init__(self, env, cfg, *, rho_init: float | None, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float, terminal_cost_multiplier: float, show_progress: bool) -> None:
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
        self.surrogate_cache = build_admm_mpc_surrogate_cache(cfg, env, horizon_steps=int(getattr(env, 'future_horizon', cfg.env.future_horizon) + 1))
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
                self._progress_bar = tqdm(total=int(env.num_available_episodes) * int(env.episode_length), desc='ADMM MPC rollout', unit='step', disable=not self._show_progress, leave=True)
            except Exception:
                self._progress_bar = None

    def reset(self) -> None:
        self.last_action_info = None
        self._episode_idx += 1
        if self._progress_bar is None and self._show_progress:
            print(f"ADMM MPC rollout: starting episode/day {self._episode_idx + 1}/{int(getattr(self.env, 'num_available_episodes', 1))}")

    def close(self) -> None:
        if self._progress_bar is not None:
            try:
                self._progress_bar.close()
            except Exception:
                pass
        elif self._show_progress:
            total_steps = int(getattr(self.env, 'num_available_episodes', 1)) * int(getattr(self.env, 'episode_length', 0))
            print(f'ADMM MPC rollout complete: processed {self._global_step}/{total_steps} steps.')
        self._progress_bar = None

    def _record_step_diagnostics(self, step_result: AdmmMpcStepResult) -> None:
        self.diagnostic_rows.append({'episode_idx': int(self._episode_idx), 'step': int(getattr(self.env, 'cur_step', 0)), 'global_step': int(self._global_step), 'admm_converged': bool(step_result.converged), 'admm_iterations': int(step_result.iterations), 'admm_final_primal_residual': float(step_result.final_primal_residual), 'admm_final_dual_residual': float(step_result.final_dual_residual), 'admm_solve_time_sec': float(step_result.solve_time_sec), 'admm_rho_final': float(step_result.rho_final)})

    def act(self, obs, deterministic: bool=True):
        del obs, deterministic
        raw_obs = self.env.obs_builder.build_raw(self.env)
        window_data = build_admm_mpc_window_data(self.cfg, self.env, raw_obs)
        terminal_weight = resolve_default_terminal_cost_weight(window_data, multiplier=float(self.terminal_cost_multiplier))
        rho_value = float(self.rho_init) if self.rho_init is not None else _compute_default_rho(window_data, self.surrogate_cache)
        step_result = run_admm_mpc_step(self.env, window_data, surrogate_cache=self.surrogate_cache, rho_init=float(rho_value), rho_min=float(self.rho_min), rho_max=float(self.rho_max), rho_adaptation=self.rho_adaptation, max_iters=int(self.max_iters), max_iters_first_step=int(self.max_iters_first_step), primal_tol=float(self.primal_tol), dual_tol=float(self.dual_tol), terminal_cost_weight_eur_per_kwh2=terminal_weight)
        self._record_step_diagnostics(step_result)
        action_info = compute_action_gap_metrics_numpy(build_safety_local_numpy(soc=np.asarray(self.env.soc, dtype=np.float32), load_raw=np.asarray(window_data.load_seq[:, 0], dtype=np.float32), pv_raw=np.asarray(window_data.pv_seq[:, 0], dtype=np.float32), battery_capacity_kwh=np.asarray(self.env.agent_c_bat, dtype=np.float32), p_max_kw=np.asarray(self.env.agent_p_max, dtype=np.float32)), step_result.executed_action_array, step_result.executed_action_array)
        action_info['solve_time_sec'] = np.float32(step_result.solve_time_sec)
        self.last_action_info = action_info
        actions = [step_result.executed_action_array[agent_idx].copy() for agent_idx in range(self.env.n)]
        if self._progress_bar is not None:
            try:
                self._progress_bar.set_postfix({'episode/day': f"{self._episode_idx + 1}/{int(getattr(self.env, 'num_available_episodes', 1))}", 'step': int(getattr(self.env, 'cur_step', 0)) + 1, 'iters': int(step_result.iterations), 'converged': bool(step_result.converged)}, refresh=False)
            except Exception:
                pass
            self._progress_bar.update(1)
        elif self._show_progress:
            episode_length = max(int(getattr(self.env, 'episode_length', 1)), 1)
            current_step_in_episode = int(getattr(self.env, 'cur_step', 0)) + 1
            if current_step_in_episode >= episode_length:
                print(f"ADMM MPC rollout progress: episode/day {self._episode_idx + 1}/{int(getattr(self.env, 'num_available_episodes', 1))}, step {self._global_step + 1}, iters={int(step_result.iterations)}, converged={bool(step_result.converged)}")
        self._global_step += 1
        return actions

def collect_admm_mpc_rollout(cfg, *, prediction_mode: str='normal', label: str | None=None, show_progress: bool=True, rho_init: float | None=None, rho_min: float=0.001, rho_max: float=1000.0, rho_adaptation: str | None='residual_balancing', max_iters: int=100, max_iters_first_step: int=300, primal_tol: float=0.001, dual_tol: float=0.001, terminal_cost_multiplier: float=1.0) -> grid_nb.RolloutResult:
    resolved_mode = grid_nb.normalize_prediction_mode(prediction_mode)
    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=resolved_mode)
    comparison_cfg.env.episode_limit = _day_episode_length(float(comparison_cfg.env.dt))
    resolved_label = str(label) if label is not None else ADMM_MPC_PERFECT_LABEL if resolved_mode == grid_nb.PERFECT_PREDICTION_MODE else ADMM_MPC_LSTM_LABEL
    def _controller_builder(env):
        controller = _AdmmMpcController(env, comparison_cfg, rho_init=rho_init, rho_min=float(rho_min), rho_max=float(rho_max), rho_adaptation=rho_adaptation, max_iters=int(max_iters), max_iters_first_step=int(max_iters_first_step), primal_tol=float(primal_tol), dual_tol=float(dual_tol), terminal_cost_multiplier=float(terminal_cost_multiplier), show_progress=bool(show_progress))
        _controller_builder.controller = controller
        return controller
    rollout = grid_nb.collect_controller_rollout(comparison_cfg, label=resolved_label, controller_builder=_controller_builder)
    controller = getattr(_controller_builder, 'controller', None)
    try:
        augmented = _augment_rollout_with_diagnostics(rollout, controller.diagnostic_rows if controller is not None else [], forecast_backend=str(comparison_cfg.forecast.type))
        augmented.meta.update({'controller': resolved_label, 'prediction_mode': resolved_mode, 'economics_scope': 'agent_only', 'soc_mode': 'continuous', 'future_horizon': int(comparison_cfg.env.future_horizon)})
        return augmented
    finally:
        if controller is not None:
            controller.close()


def _augment_rollout_with_diagnostics(rollout: grid_nb.RolloutResult, diagnostic_rows: list[dict[str, object]], *, forecast_backend: str) -> grid_nb.RolloutResult:
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    if 'global_step' not in step_df.columns:
        step_df['global_step'] = np.arange(len(step_df), dtype=np.int64)
    step_index = step_df[['episode_idx', 'step', 'global_step']].drop_duplicates()
    if not agent_df.empty and 'global_step' not in agent_df.columns:
        agent_df = agent_df.merge(step_index, on=['episode_idx', 'step'], how='left')
    if not grid_df.empty and 'global_step' not in grid_df.columns:
        grid_df = grid_df.merge(step_index, on=['episode_idx', 'step'], how='left')
    if not diagnostic_df.empty:
        merge_keys = [column for column in ('episode_idx', 'step', 'global_step') if column in diagnostic_df.columns]
        step_df = step_df.merge(diagnostic_df, on=merge_keys, how='left')
    for column, default_value in [('admm_converged', False), ('admm_iterations', 0), ('admm_final_primal_residual', np.nan), ('admm_final_dual_residual', np.nan), ('admm_solve_time_sec', np.nan), ('admm_rho_final', np.nan)]:
        if column not in step_df.columns:
            step_df[column] = default_value
    step_df['admm_converged'] = step_df['admm_converged'].astype(bool)
    import_price_markup = float(rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0))
    if WHOLESALE_PRICE_PRED_COLUMN in step_df.columns and IMPORT_PRICE_PRED_COLUMN not in step_df.columns:
        step_df[IMPORT_PRICE_PRED_COLUMN] = derive_import_price_seq(step_df[WHOLESALE_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64), markup_eur_per_kwh=import_price_markup)
    step_df['objective_total'] = step_df['purchase_cost_total'].astype(float) - step_df['export_subsidy_total'].astype(float)
    meta = dict(rollout.meta)
    meta['forecast_backend'] = str(forecast_backend)
    meta['admm_terminal_cost_mode'] = 'quadratic_to_soc_target'
    return grid_nb.RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=rollout.summary.copy(), meta=meta)
