"""Dual two-pass notebook helpers for transformer export mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from controllers.action_feasibility import (
    build_safety_local_numpy,
    compute_action_gap_metrics_numpy,
    merge_action_info_into_step_info,
)
from controllers.madrl.safety_projector import JointGridSafetyProjector
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.grid_notebook_workflow import RolloutResult


@dataclass(frozen=True)
class LocalMPCBundle:
    price_seq: np.ndarray
    load_seq: np.ndarray
    pv_seq: np.ndarray
    soc_init: np.ndarray
    p_max_kw: np.ndarray
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    pv_effective_kw: np.ndarray
    pv_utilization: np.ndarray
    signed_battery_kw: np.ndarray
    net_load_kw: np.ndarray
    energy_kwh: np.ndarray
    objective_eur: np.ndarray
    solve_time_sec: np.ndarray
    feasible: np.ndarray
    net_load_floor_kw: np.ndarray | None = None


@dataclass(frozen=True)
class DualTrafoSurrogate:
    trafo_limit_kw: float
    alpha_netload_kw: np.ndarray
    alpha_netload_window_kw: np.ndarray
    baseline_root_p_kw: np.ndarray
    export_overload_mask: np.ndarray
    import_overload_mask: np.ndarray


@dataclass(frozen=True)
class PerStepNetLoadAllocation:
    delta_netload_max_kw: np.ndarray
    delta_battery_headroom_kw: np.ndarray
    delta_curtail_headroom_kw: np.ndarray
    export_deficit_kw: np.ndarray
    delta_netload_kw: np.ndarray
    dual_lambda: np.ndarray
    surrogate_trafo_relief_kw: np.ndarray
    net_load_floor_kw: np.ndarray
    infeasible_export_mask: np.ndarray
    import_only_unsupported: bool


@dataclass(frozen=True)
class FirstStepPFEvaluation:
    battery_power_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    line_loading_pct: np.ndarray
    trafo_loading_pct: np.ndarray
    trafo_p_signed_kw: np.ndarray
    root_import_kw: float
    root_export_kw: float
    max_line_loading_pct: float
    max_trafo_loading_pct: float
    export_over_limit_kw: float
    import_over_limit_kw: float


@dataclass(frozen=True)
class DualTwoPassStepResult:
    actions: list[np.ndarray]
    action_array: np.ndarray
    action_info: dict[str, np.ndarray]
    first_pass_bundle: LocalMPCBundle
    second_pass_bundle: LocalMPCBundle
    surrogate: DualTrafoSurrogate
    allocation: PerStepNetLoadAllocation
    round1_pf: FirstStepPFEvaluation
    final_pf: FirstStepPFEvaluation
    last_candidate_pf: FirstStepPFEvaluation | None
    beta: float
    backoff_iterations: int
    flexibility_insufficient: bool
    import_only_unsupported: bool

    def to_diagnostic_payload(self) -> dict[str, object]:
        dual_round1_root_p_kw = (
            float(self.round1_pf.trafo_p_signed_kw.reshape(-1)[0]) if self.round1_pf.trafo_p_signed_kw.size else float("nan")
        )
        dual_final_root_p_kw = (
            float(self.final_pf.trafo_p_signed_kw.reshape(-1)[0]) if self.final_pf.trafo_p_signed_kw.size else float("nan")
        )
        dual_pf_trafo_relief_kw = (
            float(dual_final_root_p_kw - dual_round1_root_p_kw)
            if np.isfinite(dual_round1_root_p_kw) and np.isfinite(dual_final_root_p_kw)
            else float("nan")
        )
        first_step_surrogate_relief_kw = (
            float(self.allocation.surrogate_trafo_relief_kw.reshape(-1)[0])
            if self.allocation.surrogate_trafo_relief_kw.size
            else float("nan")
        )
        payload = {
            "beta": float(self.beta),
            "backoff_iterations": int(self.backoff_iterations),
            "flexibility_insufficient": bool(self.flexibility_insufficient),
            "import_only_unsupported": bool(self.import_only_unsupported),
            "dual_round1_root_p_kw": dual_round1_root_p_kw,
            "dual_final_root_p_kw": dual_final_root_p_kw,
            "dual_round1_export_over_limit_kw": float(self.round1_pf.export_over_limit_kw),
            "dual_final_export_over_limit_kw": float(self.final_pf.export_over_limit_kw),
            "dual_round1_first_step_battery_kw": self.first_pass_bundle.signed_battery_kw[:, 0].astype(np.float32).tolist(),
            "dual_final_first_step_battery_kw": self.second_pass_bundle.signed_battery_kw[:, 0].astype(np.float32).tolist(),
            "dual_first_pass_net_load_kw": self.first_pass_bundle.net_load_kw.astype(np.float32).tolist(),
            "dual_second_pass_net_load_kw": self.second_pass_bundle.net_load_kw.astype(np.float32).tolist(),
            "dual_net_load_floor_kw": self.allocation.net_load_floor_kw.astype(np.float32).tolist(),
            "dual_delta_netload_kw": self.allocation.delta_netload_kw.astype(np.float32).tolist(),
            "dual_export_deficit_kw": self.allocation.export_deficit_kw.astype(np.float32).tolist(),
            "dual_delta_netload_max_kw": self.allocation.delta_netload_max_kw.astype(np.float32).tolist(),
            "dual_delta_battery_headroom_kw": self.allocation.delta_battery_headroom_kw.astype(np.float32).tolist(),
            "dual_delta_curtail_headroom_kw": self.allocation.delta_curtail_headroom_kw.astype(np.float32).tolist(),
            "dual_baseline_root_p_kw": self.surrogate.baseline_root_p_kw.astype(np.float32).tolist(),
            "dual_alpha_netload_kw": self.surrogate.alpha_netload_kw.astype(np.float32).tolist(),
            "dual_lambda": self.allocation.dual_lambda.astype(np.float32).tolist(),
            "dual_surrogate_trafo_relief_kw": self.allocation.surrogate_trafo_relief_kw.astype(np.float32).tolist(),
            "dual_infeasible_export_mask": self.allocation.infeasible_export_mask.astype(bool).tolist(),
            "dual_round1_pv_curtail_kw": self.first_pass_bundle.pv_curtail_kw.astype(np.float32).tolist(),
            "dual_round2_pv_curtail_kw": self.second_pass_bundle.pv_curtail_kw.astype(np.float32).tolist(),
            "dual_round1_pv_raw_kw": self.first_pass_bundle.pv_seq.astype(np.float32).tolist(),
            "dual_round2_pv_raw_kw": self.second_pass_bundle.pv_seq.astype(np.float32).tolist(),
            "dual_round1_pv_effective_kw": self.first_pass_bundle.pv_effective_kw.astype(np.float32).tolist(),
            "dual_round2_pv_effective_kw": self.second_pass_bundle.pv_effective_kw.astype(np.float32).tolist(),
            "dual_round1_pv_utilization": self.first_pass_bundle.pv_utilization.astype(np.float32).tolist(),
            "dual_round2_pv_utilization": self.second_pass_bundle.pv_utilization.astype(np.float32).tolist(),
            "dual_round1_line_loading_pct": self.round1_pf.line_loading_pct.astype(np.float32).tolist(),
            "dual_round1_trafo_loading_pct": self.round1_pf.trafo_loading_pct.astype(np.float32).tolist(),
            "dual_final_line_loading_pct": self.final_pf.line_loading_pct.astype(np.float32).tolist(),
            "dual_final_trafo_loading_pct": self.final_pf.trafo_loading_pct.astype(np.float32).tolist(),
            "dual_pf_trafo_relief_kw": dual_pf_trafo_relief_kw,
            "dual_round2_achieved_netload_lift_kw": (
                (self.second_pass_bundle.net_load_kw - self.first_pass_bundle.net_load_kw).astype(np.float32).tolist()
            ),
            "dual_surrogate_pf_relief_error_kw": (
                float(first_step_surrogate_relief_kw - dual_pf_trafo_relief_kw)
                if np.isfinite(first_step_surrogate_relief_kw) and np.isfinite(dual_pf_trafo_relief_kw)
                else float("nan")
            ),
        }
        if self.last_candidate_pf is not None:
            payload["dual_last_candidate_root_p_kw"] = (
                float(self.last_candidate_pf.trafo_p_signed_kw.reshape(-1)[0])
                if self.last_candidate_pf.trafo_p_signed_kw.size
                else float("nan")
            )
            payload["dual_last_candidate_export_over_limit_kw"] = float(self.last_candidate_pf.export_over_limit_kw)
        return payload


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    stage: str,
    **payload: object,
) -> None:
    if progress_cb is not None:
        progress_cb(stage, payload)


def _coerce_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _array_or_empty(payload: Mapping[str, object], key: str, *, dtype=np.float32) -> np.ndarray:
    value = payload.get(key)
    if value is None:
        return np.zeros((0,), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _projector_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _get_dual_projector(cfg, env) -> JointGridSafetyProjector:
    projector = getattr(env, "_dual_notebook_projector", None)
    if isinstance(projector, JointGridSafetyProjector):
        return projector
    projector = JointGridSafetyProjector.from_cfg(cfg, device="cpu")
    setattr(env, "_dual_notebook_projector", projector)
    return projector


def _battery_to_netload_sensitivity(projector: JointGridSafetyProjector) -> np.ndarray:
    """Map battery signed-power sensitivity into net-load sensitivity.

    Under fixed load/PV trajectories, increasing battery charging power by 1 kW
    increases net load by the same 1 kW, so the transformer sensitivity is
    numerically identical in the battery-signed and net-load domains.
    """

    trafo_sensitivity = _projector_to_numpy(projector.trafo_power_sensitivity)
    if trafo_sensitivity.size == 0:
        return np.zeros((int(projector.n_agents),), dtype=np.float32)
    if trafo_sensitivity.shape[0] != 1:
        raise ValueError(
            "Dual notebook helpers expect exactly one transformer sensitivity row, "
            f"got shape {trafo_sensitivity.shape}."
        )
    return np.asarray(trafo_sensitivity[0, : int(projector.n_agents)], dtype=np.float32)


def _compute_battery_headroom_kw(first_pass_bundle: LocalMPCBundle) -> np.ndarray:
    return np.maximum(
        np.asarray(first_pass_bundle.p_max_kw, dtype=np.float32)[:, None]
        - np.asarray(first_pass_bundle.signed_battery_kw, dtype=np.float32),
        0.0,
    ).astype(np.float32)


def _compute_curtail_headroom_kw(first_pass_bundle: LocalMPCBundle) -> np.ndarray:
    return np.maximum(
        np.maximum(np.asarray(first_pass_bundle.pv_seq, dtype=np.float32), 0.0)
        - np.asarray(first_pass_bundle.pv_curtail_kw, dtype=np.float32),
        0.0,
    ).astype(np.float32)


def _compute_netload_delta_max_kw(first_pass_bundle: LocalMPCBundle) -> np.ndarray:
    return (
        _compute_battery_headroom_kw(first_pass_bundle)
        + _compute_curtail_headroom_kw(first_pass_bundle)
    ).astype(np.float32)


def _empty_local_bundle(
    *,
    price_seq: np.ndarray,
    load_seq: np.ndarray,
    pv_seq: np.ndarray,
    soc_init: np.ndarray,
    p_max_kw: np.ndarray,
    net_load_floor_kw: np.ndarray | None = None,
) -> LocalMPCBundle:
    horizon = int(np.asarray(price_seq, dtype=np.float32).reshape(-1).size)
    n_agents = int(np.asarray(load_seq, dtype=np.float32).shape[0])
    return LocalMPCBundle(
        price_seq=np.asarray(price_seq, dtype=np.float32).reshape(-1).copy(),
        load_seq=np.asarray(load_seq, dtype=np.float32).copy(),
        pv_seq=np.asarray(pv_seq, dtype=np.float32).copy(),
        soc_init=np.asarray(soc_init, dtype=np.float32).reshape(-1).copy(),
        p_max_kw=np.asarray(p_max_kw, dtype=np.float32).reshape(-1).copy(),
        charge_kw=np.zeros((n_agents, horizon), dtype=np.float32),
        discharge_kw=np.zeros((n_agents, horizon), dtype=np.float32),
        pv_curtail_kw=np.zeros((n_agents, horizon), dtype=np.float32),
        pv_effective_kw=np.zeros((n_agents, horizon), dtype=np.float32),
        pv_utilization=np.ones((n_agents, horizon), dtype=np.float32),
        signed_battery_kw=np.zeros((n_agents, horizon), dtype=np.float32),
        net_load_kw=np.zeros((n_agents, horizon), dtype=np.float32),
        energy_kwh=np.zeros((n_agents, horizon + 1), dtype=np.float32),
        objective_eur=np.zeros((n_agents,), dtype=np.float32),
        solve_time_sec=np.zeros((n_agents,), dtype=np.float32),
        feasible=np.zeros((n_agents,), dtype=bool),
        net_load_floor_kw=None
        if net_load_floor_kw is None
        else np.asarray(net_load_floor_kw, dtype=np.float32).copy(),
    )


def solve_first_pass_local_mpc_bundle(
    env,
    raw_obs: dict[str, np.ndarray],
    *,
    show_progress: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> LocalMPCBundle:
    if show_progress and progress_cb is None:
        print(f"[dual] round1: solving {int(env.n)} local MPC problems")
    price_seq = np.asarray(raw_obs["price_seq"], dtype=np.float32).reshape(-1)
    load_seq = np.asarray(raw_obs["load_seq"], dtype=np.float32)
    pv_seq = np.asarray(raw_obs["pv_seq"], dtype=np.float32)
    soc_init = np.asarray(env.soc, dtype=np.float32).reshape(-1)
    p_max_kw = np.asarray(env.agent_p_max, dtype=np.float32).reshape(-1)
    bundle = _empty_local_bundle(
        price_seq=price_seq,
        load_seq=load_seq,
        pv_seq=pv_seq,
        soc_init=soc_init,
        p_max_kw=p_max_kw,
    )
    export_subsidy = float(getattr(env.reward_fn, "export_subsidy_eur_per_kwh", 0.079))

    charge_kw = np.zeros_like(bundle.charge_kw)
    discharge_kw = np.zeros_like(bundle.discharge_kw)
    pv_curtail_kw = np.zeros_like(bundle.pv_curtail_kw)
    pv_effective_kw = np.zeros_like(bundle.pv_effective_kw)
    pv_utilization = np.ones_like(bundle.pv_utilization)
    signed_battery_kw = np.zeros_like(bundle.signed_battery_kw)
    net_load_kw = np.zeros_like(bundle.net_load_kw)
    energy_kwh = np.zeros_like(bundle.energy_kwh)
    objective_eur = np.zeros_like(bundle.objective_eur)
    solve_time_sec = np.zeros_like(bundle.solve_time_sec)
    feasible = np.zeros_like(bundle.feasible)

    for agent_idx in range(env.n):
        _emit_progress(
            progress_cb,
            "round1",
            agent_idx=int(agent_idx + 1),
            total_agents=int(env.n),
        )
        solver, _ = grid_nb._get_single_agent_mpc_solver(
            env,
            agent_idx=agent_idx,
            price_seq=price_seq,
            battery_capacity_kwh=float(env.agent_c_bat[agent_idx]),
            p_max_kw=float(env.agent_p_max[agent_idx]),
            dt_hours=float(env.dt),
            efficiency=float(env.eff),
            soc_min=float(env.soc_min),
            soc_max=float(env.soc_max),
            export_subsidy_eur_per_kwh=export_subsidy,
        )
        result = solver.solve_full_horizon(
            price_seq=price_seq,
            load_seq=load_seq[agent_idx],
            pv_seq=pv_seq[agent_idx],
            soc=float(soc_init[agent_idx]),
            pv_curtail_upper_kw=np.zeros_like(pv_seq[agent_idx], dtype=np.float32),
        )
        charge_kw[agent_idx] = result.charge_kw
        discharge_kw[agent_idx] = result.discharge_kw
        pv_curtail_kw[agent_idx] = result.pv_curtail_kw
        pv_effective_kw[agent_idx] = result.pv_effective_kw
        pv_utilization[agent_idx] = result.pv_utilization
        signed_battery_kw[agent_idx] = result.signed_battery_kw
        net_load_kw[agent_idx] = result.net_load_kw
        energy_kwh[agent_idx] = result.energy_kwh
        objective_eur[agent_idx] = np.float32(result.objective_eur)
        solve_time_sec[agent_idx] = np.float32(result.solve_time_sec)
        feasible[agent_idx] = bool(result.feasible)

    infeasible_agents = np.nonzero(~feasible.astype(bool))[0].tolist()
    if infeasible_agents:
        raise RuntimeError(
            "Round-1 local MPC returned infeasible or unbounded solutions for "
            f"agents {infeasible_agents}. This usually means the relaxed local MPC "
            "formulation became unbounded after disabling guarded fallback."
        )

    return LocalMPCBundle(
        price_seq=price_seq.copy(),
        load_seq=load_seq.copy(),
        pv_seq=pv_seq.copy(),
        soc_init=soc_init.copy(),
        p_max_kw=p_max_kw.copy(),
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_kw,
        pv_effective_kw=pv_effective_kw,
        pv_utilization=pv_utilization,
        signed_battery_kw=signed_battery_kw,
        net_load_kw=net_load_kw,
        energy_kwh=energy_kwh,
        objective_eur=objective_eur,
        solve_time_sec=solve_time_sec,
        feasible=feasible,
        net_load_floor_kw=None,
    )


def build_dual_trafo_surrogate(
    cfg,
    env,
    first_pass_bundle: LocalMPCBundle,
    *,
    projector: JointGridSafetyProjector | None = None,
) -> DualTrafoSurrogate:
    projector = _get_dual_projector(cfg, env) if projector is None else projector
    alpha_netload_kw = _battery_to_netload_sensitivity(projector)
    alpha_window = np.repeat(alpha_netload_kw[:, None], first_pass_bundle.net_load_kw.shape[1], axis=1).astype(np.float32)

    trafo_base_kw = _projector_to_numpy(projector.trafo_power_base_kw).reshape(-1)
    if trafo_base_kw.size == 0:
        raise ValueError("Dual notebook helpers require a transformer baseline, but no trafo data was found.")
    baseline_root_p_kw = (
        float(trafo_base_kw[0])
        + np.sum(alpha_window * np.asarray(first_pass_bundle.net_load_kw, dtype=np.float32), axis=0)
    ).astype(np.float32)
    trafo_limit_kw = grid_nb._approx_trafo_limit_kw(env, loading_limit_pct=float(cfg.grid.line_max_loading_pct))
    if trafo_limit_kw is None or not np.isfinite(float(trafo_limit_kw)):
        raise ValueError("Unable to resolve a transformer power reference limit for the dual notebook helpers.")
    export_overload_mask = baseline_root_p_kw < -float(trafo_limit_kw) - 1e-6
    import_overload_mask = baseline_root_p_kw > float(trafo_limit_kw) + 1e-6
    return DualTrafoSurrogate(
        trafo_limit_kw=float(trafo_limit_kw),
        alpha_netload_kw=alpha_netload_kw.astype(np.float32),
        alpha_netload_window_kw=alpha_window,
        baseline_root_p_kw=baseline_root_p_kw.astype(np.float32),
        export_overload_mask=export_overload_mask.astype(bool),
        import_overload_mask=import_overload_mask.astype(bool),
    )


def _solve_single_step_identity_qp(
    alpha_netload_kw: np.ndarray,
    delta_max_kw: np.ndarray,
    export_deficit_kw: float,
    *,
    max_iter: int = 48,
    tol_kw: float = 1e-4,
) -> tuple[np.ndarray, float, float, bool]:
    alpha = np.maximum(np.asarray(alpha_netload_kw, dtype=np.float32).reshape(-1), 0.0)
    delta_max = np.maximum(np.asarray(delta_max_kw, dtype=np.float32).reshape(-1), 0.0)
    deficit = max(float(export_deficit_kw), 0.0)
    delta = np.zeros_like(delta_max, dtype=np.float32)
    if deficit <= tol_kw:
        return delta, 0.0, 0.0, True
    if alpha.size == 0:
        return delta, float("nan"), 0.0, False
    available = float(np.dot(alpha, delta_max))
    if available + tol_kw < deficit:
        saturated = delta_max.astype(np.float32)
        return saturated, float("nan"), float(np.dot(alpha, saturated)), False
    positive_mask = alpha > 1e-8
    if not np.any(positive_mask):
        return delta, float("nan"), 0.0, False
    high = float(np.max(delta_max[positive_mask] / alpha[positive_mask]))
    low = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        candidate = np.minimum(alpha * mid, delta_max)
        delivered = float(np.dot(alpha, candidate))
        if abs(delivered - deficit) <= tol_kw:
            return candidate.astype(np.float32), float(mid), delivered, True
        if delivered >= deficit:
            high = mid
        else:
            low = mid
    delta = np.minimum(alpha * high, delta_max).astype(np.float32)
    delivered = float(np.dot(alpha, delta))
    feasible = delivered + tol_kw >= deficit
    return delta, (float(high) if feasible else float("nan")), delivered, feasible


def solve_per_step_netload_qp_allocation(
    first_pass_bundle: LocalMPCBundle,
    surrogate: DualTrafoSurrogate,
    *,
    show_progress: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> PerStepNetLoadAllocation:
    if show_progress and progress_cb is None:
        print("[dual] dual: allocating per-step net-load floor")
    delta_battery_headroom_kw = _compute_battery_headroom_kw(first_pass_bundle)
    delta_curtail_headroom_kw = _compute_curtail_headroom_kw(first_pass_bundle)
    delta_netload_max_kw = (delta_battery_headroom_kw + delta_curtail_headroom_kw).astype(np.float32)
    export_deficit_kw = np.maximum(
        -float(surrogate.trafo_limit_kw) - np.asarray(surrogate.baseline_root_p_kw, dtype=np.float32),
        0.0,
    ).astype(np.float32)
    delta_netload_kw = np.zeros_like(first_pass_bundle.net_load_kw, dtype=np.float32)
    dual_lambda = np.zeros_like(export_deficit_kw, dtype=np.float32)
    surrogate_trafo_relief_kw = np.zeros_like(export_deficit_kw, dtype=np.float32)
    infeasible_export_mask = np.zeros_like(export_deficit_kw, dtype=bool)
    for step_idx, deficit_kw in enumerate(export_deficit_kw):
        _emit_progress(
            progress_cb,
            "dual",
            horizon_step=int(step_idx + 1),
            total_horizon_steps=int(export_deficit_kw.size),
        )
        delta_step, lambda_step, relief_step, feasible = _solve_single_step_identity_qp(
            surrogate.alpha_netload_kw,
            delta_netload_max_kw[:, step_idx],
            float(deficit_kw),
        )
        delta_netload_kw[:, step_idx] = delta_step
        dual_lambda[step_idx] = np.float32(lambda_step)
        surrogate_trafo_relief_kw[step_idx] = np.float32(relief_step)
        infeasible_export_mask[step_idx] = not bool(feasible)
    net_load_floor_kw = (first_pass_bundle.net_load_kw + delta_netload_kw).astype(np.float32)
    import_only_unsupported = bool(np.any(surrogate.import_overload_mask) and not np.any(surrogate.export_overload_mask))
    return PerStepNetLoadAllocation(
        delta_netload_max_kw=delta_netload_max_kw.astype(np.float32),
        delta_battery_headroom_kw=delta_battery_headroom_kw.astype(np.float32),
        delta_curtail_headroom_kw=delta_curtail_headroom_kw.astype(np.float32),
        export_deficit_kw=export_deficit_kw,
        delta_netload_kw=delta_netload_kw,
        dual_lambda=dual_lambda,
        surrogate_trafo_relief_kw=surrogate_trafo_relief_kw,
        net_load_floor_kw=net_load_floor_kw,
        infeasible_export_mask=infeasible_export_mask.astype(bool),
        import_only_unsupported=import_only_unsupported,
    )


def resolve_second_pass_local_mpc_bundle(
    env,
    first_pass_bundle: LocalMPCBundle,
    *,
    net_load_floor_kw: np.ndarray,
    show_progress: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> LocalMPCBundle:
    if show_progress and progress_cb is None:
        print(f"[dual] round2: solving {int(env.n)} constrained local MPC problems")
    floor = np.asarray(net_load_floor_kw, dtype=np.float32)
    if floor.shape != first_pass_bundle.net_load_kw.shape:
        raise ValueError(
            "net_load_floor_kw should match first_pass_bundle.net_load_kw shape, "
            f"got {floor.shape} vs {first_pass_bundle.net_load_kw.shape}."
        )

    export_subsidy = float(getattr(env.reward_fn, "export_subsidy_eur_per_kwh", 0.079))
    charge_kw = np.zeros_like(first_pass_bundle.charge_kw)
    discharge_kw = np.zeros_like(first_pass_bundle.discharge_kw)
    pv_curtail_kw = np.zeros_like(first_pass_bundle.pv_curtail_kw)
    pv_effective_kw = np.zeros_like(first_pass_bundle.pv_effective_kw)
    pv_utilization = np.ones_like(first_pass_bundle.pv_utilization)
    signed_battery_kw = np.zeros_like(first_pass_bundle.signed_battery_kw)
    net_load_kw = np.zeros_like(first_pass_bundle.net_load_kw)
    energy_kwh = np.zeros_like(first_pass_bundle.energy_kwh)
    objective_eur = np.zeros_like(first_pass_bundle.objective_eur)
    solve_time_sec = np.zeros_like(first_pass_bundle.solve_time_sec)
    feasible = np.zeros_like(first_pass_bundle.feasible)

    for agent_idx in range(env.n):
        _emit_progress(
            progress_cb,
            "round2",
            agent_idx=int(agent_idx + 1),
            total_agents=int(env.n),
        )
        solver, _ = grid_nb._get_single_agent_mpc_solver(
            env,
            agent_idx=agent_idx,
            price_seq=first_pass_bundle.price_seq,
            battery_capacity_kwh=float(env.agent_c_bat[agent_idx]),
            p_max_kw=float(env.agent_p_max[agent_idx]),
            dt_hours=float(env.dt),
            efficiency=float(env.eff),
            soc_min=float(env.soc_min),
            soc_max=float(env.soc_max),
            export_subsidy_eur_per_kwh=export_subsidy,
        )
        result = solver.solve_full_horizon_with_netload_floor(
            price_seq=first_pass_bundle.price_seq,
            load_seq=first_pass_bundle.load_seq[agent_idx],
            pv_seq=first_pass_bundle.pv_seq[agent_idx],
            soc=float(first_pass_bundle.soc_init[agent_idx]),
            net_load_floor_kw=floor[agent_idx],
            pv_curtail_upper_kw=np.maximum(first_pass_bundle.pv_seq[agent_idx], 0.0).astype(np.float32),
        )
        charge_kw[agent_idx] = result.charge_kw
        discharge_kw[agent_idx] = result.discharge_kw
        pv_curtail_kw[agent_idx] = result.pv_curtail_kw
        pv_effective_kw[agent_idx] = result.pv_effective_kw
        pv_utilization[agent_idx] = result.pv_utilization
        signed_battery_kw[agent_idx] = result.signed_battery_kw
        net_load_kw[agent_idx] = result.net_load_kw
        energy_kwh[agent_idx] = result.energy_kwh
        objective_eur[agent_idx] = np.float32(result.objective_eur)
        solve_time_sec[agent_idx] = np.float32(result.solve_time_sec)
        feasible[agent_idx] = bool(result.feasible)

    return LocalMPCBundle(
        price_seq=first_pass_bundle.price_seq.copy(),
        load_seq=first_pass_bundle.load_seq.copy(),
        pv_seq=first_pass_bundle.pv_seq.copy(),
        soc_init=first_pass_bundle.soc_init.copy(),
        p_max_kw=first_pass_bundle.p_max_kw.copy(),
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_kw,
        pv_effective_kw=pv_effective_kw,
        pv_utilization=pv_utilization,
        signed_battery_kw=signed_battery_kw,
        net_load_kw=net_load_kw,
        energy_kwh=energy_kwh,
        objective_eur=objective_eur,
        solve_time_sec=solve_time_sec,
        feasible=feasible,
        net_load_floor_kw=floor.copy(),
    )


def _evaluate_first_step_pf(
    env,
    battery_power_kw: np.ndarray,
    pv_curtail_kw: np.ndarray,
    *,
    trafo_limit_kw: float,
    loading_limit_pct: float,
) -> FirstStepPFEvaluation:
    del loading_limit_pct
    base_net_load_effective = (
        np.asarray(env.get_signal_step("load"), dtype=np.float32)
        - np.asarray(env.get_signal_step("pv"), dtype=np.float32)
        + np.asarray(pv_curtail_kw, dtype=np.float32)
    ).astype(np.float32)
    pf_result = env._grid_core.step(
        p_batt_kw=np.asarray(battery_power_kw, dtype=np.float32),
        base_load_kw=base_net_load_effective,
    )
    line_loading_pct = np.asarray(getattr(pf_result, "line_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    trafo_loading_pct = np.asarray(getattr(pf_result, "trafo_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    trafo_p_signed_kw = np.asarray(getattr(pf_result, "trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32)
    root_p_signed_kw = float(trafo_p_signed_kw.reshape(-1)[0]) if trafo_p_signed_kw.size else 0.0
    return FirstStepPFEvaluation(
        battery_power_kw=np.asarray(battery_power_kw, dtype=np.float32).copy(),
        pv_curtail_kw=np.asarray(pv_curtail_kw, dtype=np.float32).copy(),
        line_loading_pct=line_loading_pct.copy(),
        trafo_loading_pct=trafo_loading_pct.copy(),
        trafo_p_signed_kw=trafo_p_signed_kw.copy(),
        root_import_kw=max(root_p_signed_kw, 0.0),
        root_export_kw=max(-root_p_signed_kw, 0.0),
        max_line_loading_pct=float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0,
        max_trafo_loading_pct=float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0,
        export_over_limit_kw=max(-root_p_signed_kw - float(trafo_limit_kw), 0.0),
        import_over_limit_kw=max(root_p_signed_kw - float(trafo_limit_kw), 0.0),
    )


def _bundle_to_actions(
    env,
    battery_power_kw: np.ndarray,
    pv_curtail_kw: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    return grid_nb._assemble_global_oracle_actions(
        env,
        np.asarray(battery_power_kw, dtype=np.float32),
        np.asarray(pv_curtail_kw, dtype=np.float32),
    )


def _candidate_is_safe(
    candidate_pf: FirstStepPFEvaluation,
    round1_pf: FirstStepPFEvaluation,
    *,
    loading_limit_pct: float,
) -> bool:
    return bool(
        candidate_pf.max_line_loading_pct <= float(loading_limit_pct) + 1e-6
        and candidate_pf.max_trafo_loading_pct <= float(loading_limit_pct) + 1e-6
        and candidate_pf.export_over_limit_kw <= round1_pf.export_over_limit_kw + 1e-6
    )


def _build_action_info(
    env,
    action_array: np.ndarray,
    *,
    beta: float,
    flexibility_insufficient: bool,
    import_only_unsupported: bool,
    allocation: PerStepNetLoadAllocation,
    round1_pf: FirstStepPFEvaluation,
    final_pf: FirstStepPFEvaluation,
    backoff_iterations: int,
) -> dict[str, np.ndarray]:
    safety_local = build_safety_local_numpy(
        soc=np.asarray(env.soc, dtype=np.float32),
        load_raw=np.asarray(env.get_signal_step("load"), dtype=np.float32),
        pv_raw=np.asarray(env.get_signal_step("pv"), dtype=np.float32),
        battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32),
        p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32),
    )
    action_info = compute_action_gap_metrics_numpy(safety_local, action_array, action_array)
    action_info["dual_beta"] = np.asarray(float(beta), dtype=np.float32)
    action_info["dual_backoff_iterations"] = np.asarray(int(backoff_iterations), dtype=np.float32)
    action_info["flexibility_insufficient"] = np.asarray(float(bool(flexibility_insufficient)), dtype=np.float32)
    action_info["import_only_unsupported"] = np.asarray(float(bool(import_only_unsupported)), dtype=np.float32)
    action_info["dual_round1_root_p_kw"] = np.asarray(float(round1_pf.trafo_p_signed_kw.reshape(-1)[0]) if round1_pf.trafo_p_signed_kw.size else 0.0, dtype=np.float32)
    action_info["dual_final_root_p_kw"] = np.asarray(float(final_pf.trafo_p_signed_kw.reshape(-1)[0]) if final_pf.trafo_p_signed_kw.size else 0.0, dtype=np.float32)
    action_info["dual_round1_export_over_limit_kw"] = np.asarray(float(round1_pf.export_over_limit_kw), dtype=np.float32)
    action_info["dual_final_export_over_limit_kw"] = np.asarray(float(final_pf.export_over_limit_kw), dtype=np.float32)
    action_info["dual_delta_first_step_kw"] = allocation.delta_netload_kw[:, 0].astype(np.float32)
    action_info["dual_delta_window_total_kw"] = np.asarray(float(np.sum(allocation.delta_netload_kw)), dtype=np.float32)
    action_info["dual_round1_pv_curtail_first_step_kw"] = round1_pf.pv_curtail_kw.astype(np.float32)
    action_info["dual_final_pv_curtail_first_step_kw"] = final_pf.pv_curtail_kw.astype(np.float32)
    return action_info


def run_dual_two_pass_step(
    env,
    raw_obs: dict[str, np.ndarray],
    cfg,
    *,
    max_bisect_iters: int = 6,
    show_progress: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> DualTwoPassStepResult:
    loading_limit_pct = float(cfg.grid.line_max_loading_pct)
    first_pass_bundle = solve_first_pass_local_mpc_bundle(
        env,
        raw_obs,
        show_progress=show_progress,
        progress_cb=progress_cb,
    )
    surrogate = build_dual_trafo_surrogate(cfg, env, first_pass_bundle)
    allocation = solve_per_step_netload_qp_allocation(
        first_pass_bundle,
        surrogate,
        show_progress=show_progress,
        progress_cb=progress_cb,
    )
    round1_pf = _evaluate_first_step_pf(
        env,
        first_pass_bundle.signed_battery_kw[:, 0],
        first_pass_bundle.pv_curtail_kw[:, 0],
        trafo_limit_kw=surrogate.trafo_limit_kw,
        loading_limit_pct=loading_limit_pct,
    )

    selected_bundle = first_pass_bundle
    final_pf = round1_pf
    last_candidate_pf: FirstStepPFEvaluation | None = None
    flexibility_insufficient = False
    import_only_unsupported = bool(allocation.import_only_unsupported)
    beta = 0.0
    backoff_iterations = 0

    needs_export_repair = bool(np.any(surrogate.export_overload_mask))
    if needs_export_repair and not import_only_unsupported:
        attempt_betas = [1.0]
        attempt_betas.extend(0.5 ** iteration for iteration in range(1, int(max_bisect_iters) + 1))
        for iteration, candidate_beta in enumerate(attempt_betas):
            _emit_progress(
                progress_cb,
                "backoff",
                attempt=int(iteration + 1),
                total_attempts=int(len(attempt_betas)),
                beta=float(candidate_beta),
            )
            candidate_delta = allocation.delta_netload_kw * np.float32(candidate_beta)
            candidate_floor = (first_pass_bundle.net_load_kw + candidate_delta).astype(np.float32)
            candidate_bundle = resolve_second_pass_local_mpc_bundle(
                env,
                first_pass_bundle,
                net_load_floor_kw=candidate_floor,
                show_progress=show_progress,
                progress_cb=progress_cb,
            )
            backoff_iterations = int(iteration)
            selected_bundle = candidate_bundle
            if not bool(np.all(candidate_bundle.feasible)):
                continue
            candidate_pf = _evaluate_first_step_pf(
                env,
                candidate_bundle.signed_battery_kw[:, 0],
                candidate_bundle.pv_curtail_kw[:, 0],
                trafo_limit_kw=surrogate.trafo_limit_kw,
                loading_limit_pct=loading_limit_pct,
            )
            last_candidate_pf = candidate_pf
            if _candidate_is_safe(candidate_pf, round1_pf, loading_limit_pct=loading_limit_pct):
                final_pf = candidate_pf
                beta = float(candidate_beta)
                break
        else:
            flexibility_insufficient = True
            selected_bundle = first_pass_bundle
            final_pf = round1_pf
            beta = 0.0

    actions, action_array = _bundle_to_actions(
        env,
        selected_bundle.signed_battery_kw[:, 0],
        selected_bundle.pv_curtail_kw[:, 0],
    )
    action_info = _build_action_info(
        env,
        action_array,
        beta=beta,
        flexibility_insufficient=flexibility_insufficient,
        import_only_unsupported=import_only_unsupported,
        allocation=allocation,
        round1_pf=round1_pf,
        final_pf=final_pf,
        backoff_iterations=backoff_iterations,
    )
    return DualTwoPassStepResult(
        actions=actions,
        action_array=action_array,
        action_info=action_info,
        first_pass_bundle=first_pass_bundle,
        second_pass_bundle=selected_bundle,
        surrogate=surrogate,
        allocation=allocation,
        round1_pf=round1_pf,
        final_pf=final_pf,
        last_candidate_pf=last_candidate_pf,
        beta=float(beta),
        backoff_iterations=int(backoff_iterations),
        flexibility_insufficient=bool(flexibility_insufficient),
        import_only_unsupported=bool(import_only_unsupported),
    )


def _build_collect_progress_handler(
    *,
    rollout_label: str,
    show_progress: bool,
) -> tuple[ProgressCallback | None, Any]:
    progress_bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            tqdm = None
        if tqdm is not None:
            progress_bar = tqdm(desc=rollout_label, unit="step", dynamic_ncols=True)

    last_printed_stage: str | None = None

    def progress_handler(stage: str, payload: Mapping[str, object]) -> None:
        nonlocal last_printed_stage
        episode_idx = payload.get("episode_idx")
        step = payload.get("step")
        postfix: dict[str, object] = {
            "ep": episode_idx if episode_idx is not None else "-",
            "step": step if step is not None else "-",
            "stage": stage,
        }
        if stage in {"round1", "round2"}:
            postfix["agent"] = f"{payload.get('agent_idx', '-')}/{payload.get('total_agents', '-')}"
        elif stage == "dual":
            postfix["tau"] = f"{payload.get('horizon_step', '-')}/{payload.get('total_horizon_steps', '-')}"
        elif stage == "backoff":
            postfix["attempt"] = f"{payload.get('attempt', '-')}/{payload.get('total_attempts', '-')}"
            beta = payload.get("beta")
            postfix["beta"] = f"{float(beta):.3f}" if isinstance(beta, (float, int, np.floating, np.integer)) else beta

        if progress_bar is not None:
            progress_bar.set_postfix(postfix, refresh=False)
            if stage == "step_done":
                progress_bar.update(1)
            return

        if not show_progress:
            return
        if stage == "step_done":
            beta = payload.get("beta", 0.0)
            print(
                f"[dual] ep={episode_idx} step={step} stage=step_done "
                f"beta={float(beta):.3f} flexibility_insufficient={bool(payload.get('flexibility_insufficient', False))}"
            )
            last_printed_stage = stage
            return
        if stage != last_printed_stage:
            extras: list[str] = []
            if stage in {"round1", "round2"}:
                extras.append(f"agent={payload.get('agent_idx', '-')}/{payload.get('total_agents', '-')}")
            elif stage == "dual":
                extras.append(f"tau={payload.get('horizon_step', '-')}/{payload.get('total_horizon_steps', '-')}")
            elif stage == "backoff":
                extras.append(f"attempt={payload.get('attempt', '-')}/{payload.get('total_attempts', '-')}")
                extras.append(f"beta={float(payload.get('beta', 0.0)):.3f}")
            print(f"[dual] ep={episode_idx} step={step} stage={stage}" + (f" {' '.join(extras)}" if extras else ""))
            last_printed_stage = stage

    return (progress_handler if show_progress else None), progress_bar


def build_dual_window_diagnostic_frame(rollout: RolloutResult) -> pd.DataFrame:
    diagnostic_rows = list(rollout.meta.get("dual_diagnostic_log", []) or [])
    expanded_rows: list[dict[str, object]] = []
    for entry in diagnostic_rows:
        lambda_values = _array_or_empty(entry, "dual_lambda")
        export_deficit = _array_or_empty(entry, "dual_export_deficit_kw")
        surrogate_relief = _array_or_empty(entry, "dual_surrogate_trafo_relief_kw")
        baseline_root_p = _array_or_empty(entry, "dual_baseline_root_p_kw")
        infeasible_export_mask = _array_or_empty(entry, "dual_infeasible_export_mask", dtype=bool)
        battery_headroom = _array_or_empty(entry, "dual_delta_battery_headroom_kw")
        curtail_headroom = _array_or_empty(entry, "dual_delta_curtail_headroom_kw")
        achieved_lift = _array_or_empty(entry, "dual_round2_achieved_netload_lift_kw")
        delta_netload = _array_or_empty(entry, "dual_delta_netload_kw")
        if delta_netload.ndim == 1:
            delta_total = delta_netload.reshape(-1)
        elif delta_netload.ndim >= 2:
            delta_total = np.sum(delta_netload, axis=0, dtype=np.float32)
        else:
            delta_total = np.zeros((0,), dtype=np.float32)
        horizon = int(
            max(
                lambda_values.size,
                export_deficit.size,
                surrogate_relief.size,
                baseline_root_p.size,
                infeasible_export_mask.size,
                delta_total.size,
            )
        )
        timestamp = _coerce_timestamp(entry.get("timestamp"))
        dual_pf_trafo_relief_kw = float(entry.get("dual_pf_trafo_relief_kw", np.nan))
        dual_surrogate_pf_relief_error_kw = float(entry.get("dual_surrogate_pf_relief_error_kw", np.nan))
        for horizon_step in range(horizon):
            lambda_value = float(lambda_values[horizon_step]) if horizon_step < lambda_values.size else float("nan")
            deficit_value = float(export_deficit[horizon_step]) if horizon_step < export_deficit.size else float("nan")
            relief_value = float(surrogate_relief[horizon_step]) if horizon_step < surrogate_relief.size else float("nan")
            delta_total_value = float(delta_total[horizon_step]) if horizon_step < delta_total.size else float("nan")
            infeasible_value = bool(infeasible_export_mask[horizon_step]) if horizon_step < infeasible_export_mask.size else False
            expanded_rows.append(
                {
                    "controller": entry.get("controller", rollout.meta.get("controller", "unknown")),
                    "episode_idx": int(entry.get("episode_idx", 0)),
                    "step": int(entry.get("step", 0)),
                    "timestamp": timestamp,
                    "horizon_step": int(horizon_step),
                    "dual_lambda": lambda_value,
                    "dual_export_deficit_kw": deficit_value,
                    "dual_surrogate_trafo_relief_kw": relief_value,
                    "dual_baseline_root_p_kw": (
                        float(baseline_root_p[horizon_step]) if horizon_step < baseline_root_p.size else float("nan")
                    ),
                    "dual_delta_total_kw": delta_total_value,
                    "dual_infeasible_export": infeasible_value,
                    "dual_active": bool((deficit_value > 1e-6) or (abs(np.nan_to_num(lambda_value, nan=0.0)) > 1e-6)),
                    "dual_battery_headroom_total_kw": (
                        float(np.sum(battery_headroom[:, horizon_step], dtype=np.float32))
                        if battery_headroom.ndim >= 2 and battery_headroom.shape[1] > horizon_step
                        else float("nan")
                    ),
                    "dual_curtail_headroom_total_kw": (
                        float(np.sum(curtail_headroom[:, horizon_step], dtype=np.float32))
                        if curtail_headroom.ndim >= 2 and curtail_headroom.shape[1] > horizon_step
                        else float("nan")
                    ),
                    "dual_round2_achieved_netload_lift_total_kw": (
                        float(np.sum(achieved_lift[:, horizon_step], dtype=np.float32))
                        if achieved_lift.ndim >= 2 and achieved_lift.shape[1] > horizon_step
                        else float("nan")
                    ),
                    "dual_pf_trafo_relief_kw": dual_pf_trafo_relief_kw,
                    "dual_surrogate_pf_relief_error_kw": dual_surrogate_pf_relief_error_kw,
                }
            )
    if not expanded_rows:
        return pd.DataFrame(
            columns=[
                "controller",
                "episode_idx",
                "step",
                "timestamp",
                "horizon_step",
                "dual_lambda",
                "dual_export_deficit_kw",
                "dual_surrogate_trafo_relief_kw",
                "dual_baseline_root_p_kw",
                "dual_delta_total_kw",
                "dual_infeasible_export",
                "dual_active",
                "dual_battery_headroom_total_kw",
                "dual_curtail_headroom_total_kw",
                "dual_round2_achieved_netload_lift_total_kw",
                "dual_pf_trafo_relief_kw",
                "dual_surrogate_pf_relief_error_kw",
            ]
        )
    return pd.DataFrame(expanded_rows).sort_values(["episode_idx", "step", "horizon_step"]).reset_index(drop=True)


def build_dual_first_step_summary_frame(rollout: RolloutResult) -> pd.DataFrame:
    diagnostic_rows = list(rollout.meta.get("dual_diagnostic_log", []) or [])
    summary_rows: list[dict[str, object]] = []
    for entry in diagnostic_rows:
        lambda_values = _array_or_empty(entry, "dual_lambda")
        export_deficit = _array_or_empty(entry, "dual_export_deficit_kw")
        surrogate_relief = _array_or_empty(entry, "dual_surrogate_trafo_relief_kw")
        infeasible_export_mask = _array_or_empty(entry, "dual_infeasible_export_mask", dtype=bool)
        round1_pv_curtail = _array_or_empty(entry, "dual_round1_pv_curtail_kw")
        round2_pv_curtail = _array_or_empty(entry, "dual_round2_pv_curtail_kw")
        round1_pv_raw = _array_or_empty(entry, "dual_round1_pv_raw_kw")
        round2_pv_raw = _array_or_empty(entry, "dual_round2_pv_raw_kw")
        round1_pv_effective = _array_or_empty(entry, "dual_round1_pv_effective_kw")
        round2_pv_effective = _array_or_empty(entry, "dual_round2_pv_effective_kw")
        achieved_lift = _array_or_empty(entry, "dual_round2_achieved_netload_lift_kw")
        battery_headroom = _array_or_empty(entry, "dual_delta_battery_headroom_kw")
        curtail_headroom = _array_or_empty(entry, "dual_delta_curtail_headroom_kw")
        delta_netload = _array_or_empty(entry, "dual_delta_netload_kw")
        if delta_netload.ndim == 1:
            delta_first_step_total_kw = float(delta_netload[0]) if delta_netload.size else 0.0
        elif delta_netload.ndim >= 2 and delta_netload.shape[1] > 0:
            delta_first_step_total_kw = float(np.sum(delta_netload[:, 0], dtype=np.float32))
        else:
            delta_first_step_total_kw = 0.0
        lambda_first_step = float(lambda_values[0]) if lambda_values.size else 0.0
        export_deficit_first_step_kw = float(export_deficit[0]) if export_deficit.size else 0.0
        surrogate_relief_first_step_kw = float(surrogate_relief[0]) if surrogate_relief.size else 0.0

        def _first_step_total(values: np.ndarray) -> float:
            if values.ndim >= 2 and values.shape[1] > 0:
                return float(np.sum(values[:, 0], dtype=np.float32))
            if values.ndim == 1 and values.size:
                return float(np.sum(values, dtype=np.float32))
            return 0.0

        round1_pv_raw_first_step_total_kw = _first_step_total(round1_pv_raw)
        round2_pv_raw_first_step_total_kw = _first_step_total(round2_pv_raw)
        round1_pv_effective_first_step_total_kw = _first_step_total(round1_pv_effective)
        round2_pv_effective_first_step_total_kw = _first_step_total(round2_pv_effective)

        round1_pv_utilization_first_step = (
            float(np.clip(round1_pv_effective_first_step_total_kw / round1_pv_raw_first_step_total_kw, 0.0, 1.0))
            if round1_pv_raw_first_step_total_kw > 1e-6
            else 1.0
        )
        round2_pv_utilization_first_step = (
            float(np.clip(round2_pv_effective_first_step_total_kw / round2_pv_raw_first_step_total_kw, 0.0, 1.0))
            if round2_pv_raw_first_step_total_kw > 1e-6
            else 1.0
        )
        summary_rows.append(
            {
                "controller": entry.get("controller", rollout.meta.get("controller", "unknown")),
                "episode_idx": int(entry.get("episode_idx", 0)),
                "step": int(entry.get("step", 0)),
                "timestamp": _coerce_timestamp(entry.get("timestamp")),
                "beta": float(entry.get("beta", 0.0)),
                "backoff_iterations": int(entry.get("backoff_iterations", 0)),
                "flexibility_insufficient": bool(entry.get("flexibility_insufficient", False)),
                "import_only_unsupported": bool(entry.get("import_only_unsupported", False)),
                "dual_round1_root_p_kw": float(entry.get("dual_round1_root_p_kw", np.nan)),
                "dual_final_root_p_kw": float(entry.get("dual_final_root_p_kw", np.nan)),
                "dual_round1_export_over_limit_kw": float(entry.get("dual_round1_export_over_limit_kw", np.nan)),
                "dual_final_export_over_limit_kw": float(entry.get("dual_final_export_over_limit_kw", np.nan)),
                "dual_lambda_first_step": lambda_first_step,
                "dual_export_deficit_first_step_kw": export_deficit_first_step_kw,
                "dual_delta_first_step_total_kw": delta_first_step_total_kw,
                "dual_surrogate_trafo_relief_first_step_kw": surrogate_relief_first_step_kw,
                "dual_round1_pv_curtail_first_step_total_kw": (
                    float(np.sum(round1_pv_curtail[:, 0], dtype=np.float32))
                    if round1_pv_curtail.ndim >= 2 and round1_pv_curtail.shape[1] > 0
                    else 0.0
                ),
                "dual_round1_pv_raw_first_step_total_kw": round1_pv_raw_first_step_total_kw,
                "dual_round1_pv_effective_first_step_total_kw": round1_pv_effective_first_step_total_kw,
                "dual_round1_pv_utilization_first_step": round1_pv_utilization_first_step,
                "dual_round2_pv_curtail_first_step_total_kw": (
                    float(np.sum(round2_pv_curtail[:, 0], dtype=np.float32))
                    if round2_pv_curtail.ndim >= 2 and round2_pv_curtail.shape[1] > 0
                    else 0.0
                ),
                "dual_round2_pv_raw_first_step_total_kw": round2_pv_raw_first_step_total_kw,
                "dual_round2_pv_effective_first_step_total_kw": round2_pv_effective_first_step_total_kw,
                "dual_round2_pv_utilization_first_step": round2_pv_utilization_first_step,
                "dual_delta_battery_headroom_first_step_total_kw": (
                    float(np.sum(battery_headroom[:, 0], dtype=np.float32))
                    if battery_headroom.ndim >= 2 and battery_headroom.shape[1] > 0
                    else 0.0
                ),
                "dual_delta_curtail_headroom_first_step_total_kw": (
                    float(np.sum(curtail_headroom[:, 0], dtype=np.float32))
                    if curtail_headroom.ndim >= 2 and curtail_headroom.shape[1] > 0
                    else 0.0
                ),
                "dual_round2_achieved_netload_lift_first_step_total_kw": (
                    float(np.sum(achieved_lift[:, 0], dtype=np.float32))
                    if achieved_lift.ndim >= 2 and achieved_lift.shape[1] > 0
                    else 0.0
                ),
                "dual_pf_trafo_relief_kw": float(entry.get("dual_pf_trafo_relief_kw", np.nan)),
                "dual_surrogate_pf_relief_error_kw": float(entry.get("dual_surrogate_pf_relief_error_kw", np.nan)),
                "dual_first_step_infeasible_export": (
                    bool(infeasible_export_mask[0]) if infeasible_export_mask.size else False
                ),
                "dual_active_first_step": bool(
                    export_deficit_first_step_kw > 1e-6 or abs(np.nan_to_num(lambda_first_step, nan=0.0)) > 1e-6
                ),
            }
        )
    if not summary_rows:
        return pd.DataFrame(
            columns=[
                "controller",
                "episode_idx",
                "step",
                "timestamp",
                "beta",
                "backoff_iterations",
                "flexibility_insufficient",
                "import_only_unsupported",
                "dual_round1_root_p_kw",
                "dual_final_root_p_kw",
                "dual_round1_export_over_limit_kw",
                "dual_final_export_over_limit_kw",
                "dual_lambda_first_step",
                "dual_export_deficit_first_step_kw",
                "dual_delta_first_step_total_kw",
                "dual_surrogate_trafo_relief_first_step_kw",
                "dual_round1_pv_curtail_first_step_total_kw",
                "dual_round1_pv_raw_first_step_total_kw",
                "dual_round1_pv_effective_first_step_total_kw",
                "dual_round1_pv_utilization_first_step",
                "dual_round2_pv_curtail_first_step_total_kw",
                "dual_round2_pv_raw_first_step_total_kw",
                "dual_round2_pv_effective_first_step_total_kw",
                "dual_round2_pv_utilization_first_step",
                "dual_delta_battery_headroom_first_step_total_kw",
                "dual_delta_curtail_headroom_first_step_total_kw",
                "dual_round2_achieved_netload_lift_first_step_total_kw",
                "dual_pf_trafo_relief_kw",
                "dual_surrogate_pf_relief_error_kw",
                "dual_first_step_infeasible_export",
                "dual_active_first_step",
            ]
        )
    return pd.DataFrame(summary_rows).sort_values(["episode_idx", "step"]).reset_index(drop=True)


def collect_dual_two_pass_rollout(
    cfg,
    *,
    prediction_mode: str,
    label: str | None = None,
    max_bisect_iters: int = 6,
    show_progress: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> RolloutResult:
    from scripts.builder import build_env

    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=prediction_mode)
    resolved_mode = grid_nb.normalize_prediction_mode(prediction_mode)
    rollout_label = label or f"Dual 2-Pass MPC ({grid_nb.resolve_evaluation_mode(resolved_mode)})"

    if grid_nb.cache_only_forecast_enabled(comparison_cfg):
        comparison_cfg.runtime.forecast_ready = None
    else:
        comparison_cfg.runtime.forecast_ready = grid_nb.ensure_forecast_ready(comparison_cfg)

    env = build_env(comparison_cfg, mode="test")
    _ = _get_dual_projector(comparison_cfg, env)
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, "agent_bus_ids", comparison_cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    loading_limit_pct = float(comparison_cfg.grid.line_max_loading_pct)
    fixed_load_kw, fixed_generation_kw = grid_nb._fixed_feeder_components_kw(env)
    trafo_limit_kw = grid_nb._approx_trafo_limit_kw(env, loading_limit_pct=loading_limit_pct)
    collect_progress_cb, progress_bar = _build_collect_progress_handler(
        rollout_label=rollout_label,
        show_progress=bool(show_progress),
    )
    active_progress_cb = progress_cb if progress_cb is not None else collect_progress_cb

    try:
        for episode_idx in range(env.num_available_episodes):
            obs, reset_info = env.reset(episode_idx=episode_idx)
            raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
            previous_raw_obs = None
            done = False
            step_in_episode = 0

            while not done:
                _emit_progress(
                    active_progress_cb,
                    "step_start",
                    episode_idx=int(episode_idx),
                    step=int(step_in_episode),
                )
                price_pred = grid_nb._aligned_prediction(previous_raw_obs, raw_obs, "price_seq")
                load_pred = grid_nb._aligned_prediction(previous_raw_obs, raw_obs, "load_seq")
                pv_pred = grid_nb._aligned_prediction(previous_raw_obs, raw_obs, "pv_seq")
                timestamp = grid_nb._step_timestamp(reset_info, step_in_episode)
                dual_step = run_dual_two_pass_step(
                    env,
                    raw_obs,
                    comparison_cfg,
                    max_bisect_iters=max_bisect_iters,
                    show_progress=show_progress,
                    progress_cb=active_progress_cb,
                )

                next_obs, reward, terminated, truncated, info = env.step(dual_step.actions)
                reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
                info, action_penalty = merge_action_info_into_step_info(
                    info,
                    dual_step.action_info,
                    soc_pen_weight=float(comparison_cfg.reward.w_soc_pen),
                    apply_action_penalty=False,
                )
                reward_array = reward_array - np.asarray(action_penalty, dtype=np.float32)
                info["reward"] = reward_array.astype(np.float32)
                del terminated, truncated
                raw_next_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") and not bool(info.get("episode_done", False)) else next_obs

                purchase_cost_per_agent = grid_nb._purchase_cost_per_agent(info, float(env.dt))
                base_net_load = np.asarray(info.get("base_net_load", np.asarray(info["load"], dtype=np.float32) - np.asarray(info["pv"], dtype=np.float32)), dtype=np.float32)
                base_net_load_effective = np.asarray(info.get("base_net_load_effective", base_net_load), dtype=np.float32)
                net_load = np.asarray(info.get("net_load", base_net_load_effective + np.asarray(info["e_bat"], dtype=np.float32)), dtype=np.float32)
                pv_raw = np.asarray(info.get("pv_raw", info["pv"]), dtype=np.float32)
                pv_effective = np.asarray(info.get("pv_effective", pv_raw), dtype=np.float32)
                pv_curtail = np.asarray(info.get("pv_curtail", pv_raw - pv_effective), dtype=np.float32)
                pv_utilization = np.asarray(info.get("pv_utilization", np.ones_like(pv_raw, dtype=np.float32)), dtype=np.float32)
                grid_import = np.asarray(info.get("grid_import_kw", np.maximum(net_load, 0.0)), dtype=np.float32)
                grid_export = np.asarray(info.get("grid_export_kw", np.maximum(-net_load, 0.0)), dtype=np.float32)
                controller_action_gap = np.asarray(info.get("controller_action_gap", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                battery_action_req = np.asarray(info.get("battery_action_req", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                battery_action_exec = np.asarray(info.get("battery_action_exec", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                pv_action_req = np.asarray(info.get("pv_action_req", np.ones(env.n, dtype=np.float32)), dtype=np.float32)
                pv_action_exec = np.asarray(info.get("pv_action_exec", info.get("pv_action", np.ones(env.n, dtype=np.float32))), dtype=np.float32)
                soc_penalty_unweighted = np.asarray(info.get("soc_penalty_unweighted", info.get("action_penalty_unweighted", np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                soc_penalty = np.asarray(info.get("r_soc_pen", info.get("r_action_pen", np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                battery_power = np.asarray(info["e_bat"], dtype=np.float32)
                battery_power_req = np.asarray(info.get("e_bat_req", battery_power), dtype=np.float32)
                battery_charge = np.clip(battery_power, 0.0, None).astype(np.float32)
                battery_discharge = np.maximum(-battery_power, 0.0).astype(np.float32)
                battery_charge_req = np.clip(battery_power_req, 0.0, None).astype(np.float32)
                battery_discharge_req = np.maximum(-battery_power_req, 0.0).astype(np.float32)
                pv_curtail_req = np.asarray(info.get("pv_curtail_req", pv_curtail), dtype=np.float32)
                battery_request_gap_kw = np.abs(battery_power_req - battery_power).astype(np.float32)
                pv_curtail_request_gap_kw = np.abs(pv_curtail_req - pv_curtail).astype(np.float32)
                line_loading_pct = np.asarray(info.get("line_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                trafo_loading_pct = np.asarray(info.get("trafo_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                root_import_kw = float(info.get("root_import_kw", dual_step.final_pf.root_import_kw))
                root_export_kw = float(info.get("root_export_kw", dual_step.final_pf.root_export_kw))
                export_subsidy_per_agent = grid_nb._export_subsidy_per_agent(info, float(env.dt), float(getattr(comparison_cfg.reward, "export_subsidy_eur_per_kwh", 0.079)))
                voltage_penalty_per_agent = np.asarray(info.get("r_safe_v", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                line_penalty_per_agent = np.asarray(info.get("r_safe_line", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                trafo_penalty_per_agent = np.asarray(info.get("r_safe_trafo", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                objective_per_agent = (purchase_cost_per_agent - export_subsidy_per_agent + soc_penalty + voltage_penalty_per_agent + line_penalty_per_agent + trafo_penalty_per_agent).astype(np.float32)
                voltage_penalty_total = grid_nb._mean_component_total(info, "r_safe_v", env.n)
                line_penalty_total = grid_nb._mean_component_total(info, "r_safe_line", env.n)
                trafo_penalty_total = grid_nb._mean_component_total(info, "r_safe_trafo", env.n)
                soc_penalty_total = float(np.sum(soc_penalty))
                purchase_cost_total = float(np.sum(purchase_cost_per_agent))
                export_subsidy_total = float(np.sum(export_subsidy_per_agent))
                agent_raw_net_load_kw = float(np.sum(base_net_load))
                agent_effective_net_load_kw = float(np.sum(base_net_load_effective))
                agent_post_action_net_load_kw = float(np.sum(net_load))
                feeder_raw_net_load_kw = float(agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_effective_net_load_kw = float(agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_post_action_net_load_kw = float(agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw)

                step_rows.append({
                    "controller": rollout_label,
                    "episode_idx": episode_idx,
                    "step": step_in_episode,
                    "timestamp": timestamp,
                    "price": float(info["price"]),
                    "price_pred": float(price_pred),
                    "base_net_load_total": float(np.sum(base_net_load)),
                    "base_net_load_effective_total": float(np.sum(base_net_load_effective)),
                    "net_load_total": float(np.sum(net_load)),
                    "agent_raw_net_load_kw": agent_raw_net_load_kw,
                    "agent_effective_net_load_kw": agent_effective_net_load_kw,
                    "agent_post_action_net_load_kw": agent_post_action_net_load_kw,
                    "fixed_load_kw": fixed_load_kw,
                    "fixed_generation_kw": fixed_generation_kw,
                    "feeder_raw_net_load_kw": feeder_raw_net_load_kw,
                    "feeder_effective_net_load_kw": feeder_effective_net_load_kw,
                    "feeder_post_action_net_load_kw": feeder_post_action_net_load_kw,
                    "load_total": float(np.sum(np.asarray(info["load"], dtype=np.float32))),
                    "pv_raw_total": float(np.sum(pv_raw)),
                    "pv_effective_total": float(np.sum(pv_effective)),
                    "pv_curtail_total": float(np.sum(pv_curtail)),
                    "grid_import_total": float(np.sum(grid_import)),
                    "grid_export_total": float(np.sum(grid_export)),
                    "battery_charge_total": float(np.sum(battery_charge)),
                    "battery_charge_req_total": float(np.sum(battery_charge_req)),
                    "battery_discharge_total": float(np.sum(battery_discharge)),
                    "battery_discharge_req_total": float(np.sum(battery_discharge_req)),
                    "pv_curtail_req_total": float(np.sum(pv_curtail_req)),
                    "battery_request_gap_kw_total": float(np.sum(battery_request_gap_kw)),
                    "pv_curtail_request_gap_kw_total": float(np.sum(pv_curtail_request_gap_kw)),
                    "projector_adjustment_kw_total": float(np.sum(battery_request_gap_kw) + np.sum(pv_curtail_request_gap_kw)),
                    "purchase_cost_total": purchase_cost_total,
                    "export_subsidy_total": export_subsidy_total,
                    "soc_penalty_total": soc_penalty_total,
                    "voltage_penalty_total": voltage_penalty_total,
                    "line_penalty_total": line_penalty_total,
                    "trafo_penalty_total": trafo_penalty_total,
                    "psi_v_raw": float(info.get("psi_v_raw", 0.0)),
                    "psi_line_raw": float(info.get("psi_line_raw", 0.0)),
                    "psi_trafo_raw": float(info.get("psi_trafo_raw", 0.0)),
                    "line_loading_pct_max": float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0,
                    "trafo_loading_pct_max": float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0,
                    "line_violation": float(info.get("line_violation", info.get("l_violation", 0.0))),
                    "trafo_violation": float(info.get("trafo_violation", 0.0)),
                    "n_line_violations": int(info.get("n_line_violations", info.get("n_l_violations", int(np.any(line_loading_pct > loading_limit_pct))))),
                    "n_trafo_violations": int(info.get("n_trafo_violations", info.get("n_t_violations", int(np.any(trafo_loading_pct > loading_limit_pct))))),
                    "objective_total": purchase_cost_total - export_subsidy_total + soc_penalty_total + voltage_penalty_total + line_penalty_total + trafo_penalty_total,
                    "voltage_violation_count": int(((np.asarray(info.get("vm_pu", []), dtype=np.float32) < float(comparison_cfg.grid.v_min_pu)) | (np.asarray(info.get("vm_pu", []), dtype=np.float32) > float(comparison_cfg.grid.v_max_pu))).sum()),
                    "controller_action_gap_total": float(np.sum(controller_action_gap)),
                    "soc_penalty_step_total": soc_penalty_total,
                    "root_import_kw": root_import_kw,
                    "root_export_kw": root_export_kw,
                    "trafo_limit_reference_kw": trafo_limit_kw,
                    "dual_beta": float(dual_step.beta),
                    "dual_backoff_iterations": int(dual_step.backoff_iterations),
                    "flexibility_insufficient": float(dual_step.flexibility_insufficient),
                    "import_only_unsupported": float(dual_step.import_only_unsupported),
                    "dual_round1_root_p_kw": float(dual_step.round1_pf.trafo_p_signed_kw.reshape(-1)[0]) if dual_step.round1_pf.trafo_p_signed_kw.size else float("nan"),
                    "dual_final_root_p_kw": float(dual_step.final_pf.trafo_p_signed_kw.reshape(-1)[0]) if dual_step.final_pf.trafo_p_signed_kw.size else float("nan"),
                    "dual_round1_export_over_limit_kw": float(dual_step.round1_pf.export_over_limit_kw),
                    "dual_final_export_over_limit_kw": float(dual_step.final_pf.export_over_limit_kw),
                    "dual_lambda_first_step": float(dual_step.allocation.dual_lambda[0]) if dual_step.allocation.dual_lambda.size else 0.0,
                    "dual_export_deficit_first_step_kw": float(dual_step.allocation.export_deficit_kw[0]) if dual_step.allocation.export_deficit_kw.size else 0.0,
                    "dual_delta_first_step_total_kw": float(np.sum(dual_step.allocation.delta_netload_kw[:, 0])),
                    "dual_surrogate_trafo_relief_first_step_kw": float(dual_step.allocation.surrogate_trafo_relief_kw[0]) if dual_step.allocation.surrogate_trafo_relief_kw.size else 0.0,
                    "dual_pf_trafo_relief_kw": (
                        float(dual_step.final_pf.trafo_p_signed_kw.reshape(-1)[0] - dual_step.round1_pf.trafo_p_signed_kw.reshape(-1)[0])
                        if dual_step.final_pf.trafo_p_signed_kw.size and dual_step.round1_pf.trafo_p_signed_kw.size
                        else float("nan")
                    ),
                    "dual_surrogate_pf_relief_error_kw": (
                        float(dual_step.allocation.surrogate_trafo_relief_kw[0])
                        - float(dual_step.final_pf.trafo_p_signed_kw.reshape(-1)[0] - dual_step.round1_pf.trafo_p_signed_kw.reshape(-1)[0])
                        if dual_step.allocation.surrogate_trafo_relief_kw.size and dual_step.final_pf.trafo_p_signed_kw.size and dual_step.round1_pf.trafo_p_signed_kw.size
                        else float("nan")
                    ),
                    "dual_delta_window_total_kw": float(np.sum(dual_step.allocation.delta_netload_kw)),
                    "dual_delta_battery_headroom_first_step_total_kw": float(np.sum(dual_step.allocation.delta_battery_headroom_kw[:, 0])),
                    "dual_delta_curtail_headroom_first_step_total_kw": float(np.sum(dual_step.allocation.delta_curtail_headroom_kw[:, 0])),
                    "dual_round1_pv_curtail_first_step_total_kw": float(np.sum(dual_step.first_pass_bundle.pv_curtail_kw[:, 0])),
                    "dual_round2_pv_curtail_first_step_total_kw": float(np.sum(dual_step.second_pass_bundle.pv_curtail_kw[:, 0])),
                    "dual_round2_achieved_netload_lift_first_step_total_kw": float(
                        np.sum(dual_step.second_pass_bundle.net_load_kw[:, 0] - dual_step.first_pass_bundle.net_load_kw[:, 0])
                    ),
                    "dual_export_overload_steps": int(np.sum(dual_step.surrogate.export_overload_mask.astype(np.int32))),
                })

                for agent_idx, profile in enumerate(comparison_cfg.data.agent_profiles):
                    agent_rows.append({
                        "controller": rollout_label,
                        "episode_idx": episode_idx,
                        "step": step_in_episode,
                        "timestamp": timestamp,
                        "agent_id": agent_idx,
                        "agent_profile": str(profile),
                        "load": float(np.asarray(info["load"], dtype=np.float32)[agent_idx]),
                        "load_pred": float(np.asarray(load_pred, dtype=np.float32)[agent_idx]),
                        "pv": float(np.asarray(info["pv"], dtype=np.float32)[agent_idx]),
                        "pv_raw": float(pv_raw[agent_idx]),
                        "pv_effective": float(pv_effective[agent_idx]),
                        "pv_curtail": float(pv_curtail[agent_idx]),
                        "pv_curtail_req": float(pv_curtail_req[agent_idx]),
                        "pv_utilization": float(pv_utilization[agent_idx]),
                        "pv_pred": float(np.asarray(pv_pred, dtype=np.float32)[agent_idx]),
                        "base_net_load": float(base_net_load[agent_idx]),
                        "base_net_load_effective": float(base_net_load_effective[agent_idx]),
                        "net_load": float(net_load[agent_idx]),
                        "grid_import_kw": float(grid_import[agent_idx]),
                        "grid_export_kw": float(grid_export[agent_idx]),
                        "e_bat": float(battery_power[agent_idx]),
                        "e_bat_req": float(battery_power_req[agent_idx]),
                        "battery_action_req": float(battery_action_req[agent_idx]),
                        "battery_action_exec": float(battery_action_exec[agent_idx]),
                        "pv_action_req": float(pv_action_req[agent_idx]),
                        "pv_action_exec": float(pv_action_exec[agent_idx]),
                        "controller_action_gap": float(controller_action_gap[agent_idx]),
                        "battery_request_gap_kw": float(battery_request_gap_kw[agent_idx]),
                        "pv_curtail_request_gap_kw": float(pv_curtail_request_gap_kw[agent_idx]),
                        "soc_penalty_unweighted": float(soc_penalty_unweighted[agent_idx]),
                        "r_soc_pen": float(soc_penalty[agent_idx]),
                        "r_safe_v": float(voltage_penalty_per_agent[agent_idx]),
                        "r_safe_line": float(line_penalty_per_agent[agent_idx]),
                        "r_safe_trafo": float(trafo_penalty_per_agent[agent_idx]),
                        "soc": float(np.asarray(info["soc_next"], dtype=np.float32)[agent_idx]),
                        "purchase_cost": float(purchase_cost_per_agent[agent_idx]),
                        "export_subsidy": float(export_subsidy_per_agent[agent_idx]),
                        "objective_total": float(objective_per_agent[agent_idx]),
                        "dual_first_pass_battery_kw": float(dual_step.first_pass_bundle.signed_battery_kw[agent_idx, 0]),
                        "dual_second_pass_battery_kw": float(dual_step.second_pass_bundle.signed_battery_kw[agent_idx, 0]),
                        "dual_first_pass_pv_curtail_kw": float(dual_step.first_pass_bundle.pv_curtail_kw[agent_idx, 0]),
                        "dual_second_pass_pv_curtail_kw": float(dual_step.second_pass_bundle.pv_curtail_kw[agent_idx, 0]),
                        "dual_first_pass_pv_utilization": float(dual_step.first_pass_bundle.pv_utilization[agent_idx, 0]),
                        "dual_second_pass_pv_utilization": float(dual_step.second_pass_bundle.pv_utilization[agent_idx, 0]),
                        "dual_first_pass_net_load_kw": float(dual_step.first_pass_bundle.net_load_kw[agent_idx, 0]),
                        "dual_second_pass_net_load_kw": float(dual_step.second_pass_bundle.net_load_kw[agent_idx, 0]),
                        "dual_net_load_floor_kw": float(dual_step.allocation.net_load_floor_kw[agent_idx, 0]),
                        "dual_delta_first_step_kw": float(dual_step.allocation.delta_netload_kw[agent_idx, 0]),
                        "dual_delta_battery_headroom_first_step_kw": float(dual_step.allocation.delta_battery_headroom_kw[agent_idx, 0]),
                        "dual_delta_curtail_headroom_first_step_kw": float(dual_step.allocation.delta_curtail_headroom_kw[agent_idx, 0]),
                    })

                vm_pu = np.asarray(info.get("vm_pu", np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
                if vm_pu.shape[0] == len(bus_ids):
                    for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
                        grid_rows.append({
                            "controller": rollout_label,
                            "episode_idx": episode_idx,
                            "step": step_in_episode,
                            "timestamp": timestamp,
                            "bus_id": int(bus_id),
                            "vm_pu": float(vm_value),
                            "is_agent_bus": bool(int(bus_id) in agent_bus_set),
                        })

                diagnostic_entry = dual_step.to_diagnostic_payload()
                diagnostic_entry["controller"] = rollout_label
                diagnostic_entry["episode_idx"] = int(episode_idx)
                diagnostic_entry["step"] = int(step_in_episode)
                diagnostic_entry["timestamp"] = timestamp.isoformat()
                diagnostic_rows.append(diagnostic_entry)
                _emit_progress(
                    active_progress_cb,
                    "step_done",
                    episode_idx=int(episode_idx),
                    step=int(step_in_episode),
                    beta=float(dual_step.beta),
                    flexibility_insufficient=bool(dual_step.flexibility_insufficient),
                )

                done = bool(info.get("episode_done", False))
                previous_raw_obs = raw_obs
                obs = next_obs
                raw_obs = raw_next_obs
                step_in_episode += 1

        step_df = pd.DataFrame(step_rows).sort_values(["episode_idx", "step"]).reset_index(drop=True)
        agent_df = pd.DataFrame(agent_rows).sort_values(["episode_idx", "step", "agent_id"]).reset_index(drop=True)
        grid_df = pd.DataFrame(grid_rows).sort_values(["episode_idx", "step", "bus_id"]).reset_index(drop=True)
        summary = agent_df.groupby(["controller", "agent_profile"], as_index=False)[["purchase_cost", "export_subsidy", "objective_total"]].sum() if not agent_df.empty else pd.DataFrame(columns=["controller", "agent_profile", "purchase_cost", "export_subsidy", "objective_total"])
        return RolloutResult(
            step_df=step_df,
            agent_df=agent_df,
            grid_df=grid_df,
            summary=summary,
            meta={
                "controller": rollout_label,
                "n_agents": int(env.n),
                "agent_profiles": list(comparison_cfg.data.agent_profiles),
                "agent_bus_ids": agent_bus_ids,
                "bus_ids": bus_ids,
                "v_min_pu": float(comparison_cfg.grid.v_min_pu),
                "v_max_pu": float(comparison_cfg.grid.v_max_pu),
                "future_horizon": int(comparison_cfg.env.future_horizon),
                "prediction_mode": grid_nb.resolve_prediction_mode_from_forecast_backend(comparison_cfg.forecast.type),
                "evaluation_mode": grid_nb.resolve_evaluation_mode(grid_nb.resolve_prediction_mode_from_forecast_backend(comparison_cfg.forecast.type)),
                "dt_hours": float(comparison_cfg.env.dt),
                "trafo_limit_kw": trafo_limit_kw,
                "trafo_limit_note": "Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.",
                "loading_limit_pct": loading_limit_pct,
                "trafo_loading_limit_pct": loading_limit_pct,
                "export_subsidy_eur_per_kwh": float(getattr(comparison_cfg.reward, "export_subsidy_eur_per_kwh", 0.079)),
                "import_price_adder_eur_per_kwh": float(getattr(getattr(comparison_cfg, "reward", None), "import_price_adder_eur_per_kwh", 0.0)),
                "single_agent_mpc_solver_build_count": int(float(getattr(env, "_single_agent_mpc_stats", {}).get("solver_build_count", 0.0))),
                "single_agent_mpc_solver_reuse_count": int(float(getattr(env, "_single_agent_mpc_stats", {}).get("solver_reuse_count", 0.0))),
                "single_agent_mpc_solve_count": int(float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_count", 0.0))),
                "single_agent_mpc_total_solve_time_sec": float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_time_sec_total", 0.0)),
                "single_agent_mpc_avg_solve_time_sec": float((float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_time_sec_total", 0.0)) / max(float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_count", 0.0)), 1.0)) if float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_count", 0.0)) > 0.0 else 0.0),
                "single_agent_mpc_guarded_fallback_count": int(float(getattr(env, "_single_agent_mpc_stats", {}).get("guarded_fallback_count", 0.0))),
                "controller_diagnostic_log": diagnostic_rows,
                "dual_diagnostic_log": diagnostic_rows,
                "soc_mode": "reset",
            },
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
        env.close()


__all__ = [
    "DualTrafoSurrogate",
    "DualTwoPassStepResult",
    "FirstStepPFEvaluation",
    "LocalMPCBundle",
    "PerStepNetLoadAllocation",
    "_battery_to_netload_sensitivity",
    "_compute_netload_delta_max_kw",
    "build_dual_first_step_summary_frame",
    "build_dual_trafo_surrogate",
    "build_dual_window_diagnostic_frame",
    "collect_dual_two_pass_rollout",
    "resolve_second_pass_local_mpc_bundle",
    "run_dual_two_pass_step",
    "solve_first_pass_local_mpc_bundle",
    "solve_per_step_netload_qp_allocation",
]
