from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN, IMPORT_PRICE_MARKUP_KEY, IMPORT_PRICE_PRED_COLUMN, WHOLESALE_PRICE_PRED_COLUMN, derive_import_price_seq
if TYPE_CHECKING:
    from scripts.utils.grid_notebook_workflow import RolloutResult
NORMAL_PREDICTION_MODE = 'normal'
_TITLE, _LABEL, _TICK, _LEGEND = (20, 18, 16, 16)
_AGENT_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ea580c', '#7c3aed', '#0891b2']
_MUTED = '#cbd5e1'
_WIDTH = 0.008
_POS_SPECS = [('load_total', 'Load', '#111827'), ('battery_charge_total', 'Charge', '#dc2626'), ('grid_export_total', 'Grid export', '#f59e0b'), ('pv_curtail_total', 'Curtailment loss', '#fca5a5')]
_NEG_SPECS = [('pv_raw_total', 'PV raw', '#16a34a'), ('grid_import_total', 'Grid import', '#2563eb'), ('battery_discharge_total', 'Discharge', '#7c3aed')]

def _controller(rollout: RolloutResult) -> str:
    return str(rollout.meta.get('controller', 'unknown'))

def _compare_style(axis, *, title: str, ylabel: str, xlabel: str | None=None) -> None:
    axis.set_title(title, fontsize=_TITLE)
    axis.set_ylabel(ylabel, fontsize=_LABEL)
    axis.tick_params(axis='both', labelsize=_TICK)
    axis.grid(True, alpha=0.25)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=_LABEL)

def _require(df: pd.DataFrame, columns: set[str], label: str, message: str) -> None:
    if df.empty or not columns.issubset(df.columns):
        missing = sorted(columns.difference(df.columns))
        raise ValueError(message.format(label=label, missing=missing))

def _price_prediction(df: pd.DataFrame, *, price_markup: float) -> np.ndarray:
    if IMPORT_PRICE_PRED_COLUMN in df.columns:
        return df[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64)
    return derive_import_price_seq(df[WHOLESALE_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64), markup_eur_per_kwh=float(price_markup))

def _prepare_agent_power_balance_frame(step_df: pd.DataFrame, *, controller_label: str, tolerance_kw: float=1.0) -> tuple[pd.DataFrame, float]:
    _require(step_df, {'load_total', 'battery_charge_total', 'pv_curtail_total', 'grid_import_total', 'grid_export_total', 'battery_discharge_total'}, controller_label, "Rollout '{label}' is missing required agent power-balance columns: {missing}")
    frame = step_df.copy()
    if 'pv_raw_total' not in frame.columns:
        if 'pv_effective_total' not in frame.columns:
            raise ValueError(f"Rollout '{controller_label}' must contain either pv_raw_total or pv_effective_total.")
        frame['pv_raw_total'] = frame['pv_effective_total'].to_numpy(dtype=np.float32) + frame['pv_curtail_total'].to_numpy(dtype=np.float32)
    pos = sum((frame[col].to_numpy(dtype=np.float32) for col, *_ in _POS_SPECS))
    neg = sum((frame[col].to_numpy(dtype=np.float32) for col, *_ in _NEG_SPECS))
    residual = pos - neg
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    if max_abs > float(tolerance_kw):
        raise ValueError(f"Rollout '{controller_label}' violates agent-only power balance by {max_abs:.3f} kW (tolerance={float(tolerance_kw):.3f} kW).")
    frame['agent_power_balance_residual_kw'] = residual.astype(np.float32)
    return (frame, max_abs)

def _prepare_compare_net_load_frame(step_df: pd.DataFrame, *, controller_label: str) -> pd.DataFrame:
    if step_df.empty:
        raise ValueError(f"Rollout '{controller_label}' has no step_df.")
    frame = step_df.copy()
    if not {'agent_raw_net_load_kw', 'agent_effective_net_load_kw', 'agent_post_action_net_load_kw'}.issubset(frame.columns):
        if not {'base_net_load_total', 'net_load_total'}.issubset(frame.columns):
            raise ValueError(f"Rollout '{controller_label}' is missing agent net-load columns required for compare plotting.")
        frame['agent_raw_net_load_kw'] = frame['base_net_load_total'].to_numpy(dtype=np.float32)
        frame['agent_effective_net_load_kw'] = np.asarray(frame.get('base_net_load_effective_total', frame['base_net_load_total']), dtype=np.float32)
        frame['agent_post_action_net_load_kw'] = frame['net_load_total'].to_numpy(dtype=np.float32)
    if not {'feeder_raw_net_load_kw', 'feeder_effective_net_load_kw', 'feeder_post_action_net_load_kw'}.issubset(frame.columns):
        if not {'fixed_load_kw', 'fixed_generation_kw'}.issubset(frame.columns):
            raise ValueError(f"Rollout '{controller_label}' is missing feeder-total net-load columns required for compare plotting.")
        fixed = frame['fixed_load_kw'].to_numpy(dtype=np.float32) - frame['fixed_generation_kw'].to_numpy(dtype=np.float32)
        for src, dst in [('agent_raw_net_load_kw', 'feeder_raw_net_load_kw'), ('agent_effective_net_load_kw', 'feeder_effective_net_load_kw'), ('agent_post_action_net_load_kw', 'feeder_post_action_net_load_kw')]:
            frame[dst] = frame[src].to_numpy(dtype=np.float32) + fixed
    return frame

def _shared_price_frame(step_df: pd.DataFrame, *, controller_label: str, require_predicted: bool=False) -> pd.DataFrame:
    required = {'timestamp', 'wholesale_price', IMPORT_PRICE_COLUMN}
    if require_predicted:
        required.add(WHOLESALE_PRICE_PRED_COLUMN)
    _require(step_df, required, controller_label, "Rollout '{label}' is missing shared price columns required for compare plotting: {missing}")
    cols = ['timestamp', 'wholesale_price', IMPORT_PRICE_COLUMN]
    cols += [c for c in (WHOLESALE_PRICE_PRED_COLUMN, IMPORT_PRICE_PRED_COLUMN) if c in step_df]
    frame = step_df.loc[:, cols].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    return frame.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

def _shared_signal_frame(agent_df: pd.DataFrame, signal_name: str, *, controller_label: str, require_predicted: bool=False) -> pd.DataFrame:
    pred = f'{signal_name}_pred'
    required = {'timestamp', 'agent_profile', signal_name}
    if require_predicted:
        required.add(pred)
    _require(agent_df, required, controller_label, f"Rollout '{{label}}' is missing shared {signal_name} columns required for compare plotting: {{missing}}")
    cols = ['timestamp', 'agent_profile', signal_name] + ([pred] if pred in agent_df else [])
    frame = agent_df.loc[:, cols].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    frame['agent_profile'] = frame['agent_profile'].astype(str)
    return frame.sort_values(['timestamp', 'agent_profile']).drop_duplicates(['timestamp', 'agent_profile']).reset_index(drop=True)

def _align_shared_actual_frames(reference_frame: pd.DataFrame, candidate_frame: pd.DataFrame, *, key_columns: list[str], value_column: str, controller_label: str, signal_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = reference_frame.set_index(key_columns)
    cand = candidate_frame.set_index(key_columns)
    common = ref.index[ref.index.isin(cand.index)]
    if not len(common):
        raise ValueError(f"Rollout '{controller_label}' has no overlapping {signal_label} timestamps/profiles with the shared forecast-vs-actual reference rollout.")
    ref = ref.loc[common].reset_index()
    cand = cand.loc[common].reset_index()
    if not np.allclose(ref[value_column].to_numpy(dtype=np.float64), cand[value_column].to_numpy(dtype=np.float64), rtol=1e-06, atol=1e-08, equal_nan=True):
        raise ValueError(f"Rollout '{controller_label}' has different actual {signal_label} values; shared forecast-vs-actual plotting requires matching actual series across rollouts.")
    return (ref, cand)

def _prediction_rollout(*rollouts: RolloutResult) -> RolloutResult:
    needed_step = {'timestamp', 'wholesale_price', IMPORT_PRICE_COLUMN, WHOLESALE_PRICE_PRED_COLUMN}
    needed_agent = {'timestamp', 'agent_profile', 'load', 'load_pred', 'pv', 'pv_pred'}
    return next((rollout for rollout in rollouts if str(rollout.meta.get('prediction_mode', '')).strip().lower() == NORMAL_PREDICTION_MODE and needed_step.issubset(rollout.step_df.columns) and needed_agent.issubset(rollout.agent_df.columns)), rollouts[0])

def _plot_agent_forecasts(axes, step_df: pd.DataFrame, agent_df: pd.DataFrame, profiles, *, title: str) -> None:
    axes[0].plot(step_df['timestamp'], step_df[IMPORT_PRICE_COLUMN], color='#111827', linewidth=1.6, label='Actual')
    axes[0].plot(step_df['timestamp'], step_df[IMPORT_PRICE_PRED_COLUMN], color='#dc2626', linewidth=1.4, linestyle='--', label='Forecast')
    axes[0].set_ylabel('Import Price')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc='upper right')
    for idx, signal in enumerate(('pv', 'load'), start=1):
        for agent_idx, profile in enumerate(profiles):
            frame = agent_df.loc[agent_df['agent_profile'] == profile]
            color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
            axes[idx].plot(frame['timestamp'], frame[signal], color=color, linewidth=1.4, label=f'{profile} actual')
            axes[idx].plot(frame['timestamp'], frame[f'{signal}_pred'], color=color, linewidth=1.2, linestyle='--', label=f'{profile} forecast')
        axes[idx].set_ylabel(signal.upper())
        axes[idx].grid(True, alpha=0.25)
        axes[idx].legend(loc='upper right', ncol=2)

def _plot_battery_rows(axes, agent_df: pd.DataFrame, profiles) -> None:
    for axis, profile in zip(axes, profiles, strict=False):
        frame = agent_df.loc[agent_df['agent_profile'] == profile]
        power = frame['e_bat'].to_numpy(dtype=np.float32)
        axis.bar(frame['timestamp'], np.clip(power, 0.0, None), width=_WIDTH, color='#dc2626', alpha=0.7, label='Charge')
        axis.bar(frame['timestamp'], np.clip(power, None, 0.0), width=_WIDTH, color='#2563eb', alpha=0.7, label='Discharge')
        axis.set_ylabel(f'{profile}\nP_bat')
        axis.grid(True, alpha=0.25)
        soc_axis = axis.twinx()
        soc_axis.plot(frame['timestamp'], frame['soc'], color='#111827', linewidth=1.2, label='SoC')
        soc_axis.set_ylabel('SoC')
        soc_axis.set_ylim(0.0, 1.0)
        h1, l1 = axis.get_legend_handles_labels()
        h2, l2 = soc_axis.get_legend_handles_labels()
        axis.legend(h1 + h2, l1 + l2, loc='upper right')

def _plot_voltage(axis, rollout: RolloutResult, *, legend: bool) -> None:
    grid_df = rollout.grid_df
    if grid_df.empty:
        raise ValueError(f"Rollout '{_controller(rollout)}' has no grid_df.")
    for _, frame in grid_df.loc[~grid_df['is_agent_bus']].groupby('bus_id'):
        axis.plot(frame['timestamp'], frame['vm_pu'], color=_MUTED, linewidth=0.9, alpha=0.35, zorder=1)
    for idx, bus_id in enumerate(rollout.meta['agent_bus_ids']):
        frame = grid_df.loc[grid_df['bus_id'] == int(bus_id)]
        if not frame.empty:
            axis.plot(frame['timestamp'], frame['vm_pu'], color=_AGENT_COLORS[idx % len(_AGENT_COLORS)], linewidth=2.1, alpha=0.95, label=f'Agent bus {bus_id}' if legend else None, zorder=3)
    axis.axhline(float(rollout.meta['v_min_pu']), color='#dc2626', linestyle='--', linewidth=1.1, label='V min' if legend else None)
    axis.axhline(float(rollout.meta['v_max_pu']), color='#ea580c', linestyle='--', linewidth=1.1, label='V max' if legend else None)

def _stack_power(axis, step_df: pd.DataFrame, *, with_legend: bool) -> None:
    bottom = np.zeros(len(step_df), dtype=np.float32)
    for col, label, color in _POS_SPECS:
        extra = {'hatch': '//', 'edgecolor': '#dc2626', 'linewidth': 1.0} if col == 'pv_curtail_total' else {}
        values = step_df[col].to_numpy(dtype=np.float32)
        axis.bar(step_df['timestamp'], values, width=_WIDTH, bottom=bottom, color=color, alpha=0.78, label=label if with_legend else None, **extra)
        bottom += values
    bottom = np.zeros(len(step_df), dtype=np.float32)
    for col, label, color in _NEG_SPECS:
        values = -step_df[col].to_numpy(dtype=np.float32)
        axis.bar(step_df['timestamp'], values, width=_WIDTH, bottom=bottom, color=color, alpha=0.78, label=label if with_legend else None)
        bottom += values
    axis.axhline(0.0, color='#111827', linewidth=1.0)

def plot_global_misocp_validation(rollout: RolloutResult, *, figsize: tuple[float, float]=(11.0, 8.0)):
    validation_df = rollout.meta.get('misocp_validation_df')
    if not isinstance(validation_df, pd.DataFrame) or validation_df.empty:
        raise ValueError('Rollout does not contain MISOCP-vs-pandapower validation data.')
    from controllers.mpc.global_socp_mpc import _LINE_LOADING_ERR_TOL_PCT, _ROOT_POWER_ERR_TOL_KW, _TRAFO_LOADING_ERR_TOL_PCT, _VOLTAGE_ERR_TOL_PU
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    metrics = [('max_vm_abs_err_pu', _VOLTAGE_ERR_TOL_PU, 'Max |V_misocp - V_pp| [p.u.]'), ('max_line_loading_abs_err_pct', _LINE_LOADING_ERR_TOL_PCT, 'Max |Line Loading| Error [pct-point]'), ('trafo_loading_abs_err_pct', _TRAFO_LOADING_ERR_TOL_PCT, '|Trafo Loading| Error [pct-point]'), ('root_p_abs_err_kw', _ROOT_POWER_ERR_TOL_KW, '|P_root| Error [kW]')]
    for axis, (col, limit, ylabel) in zip(axes, metrics, strict=False):
        axis.plot(validation_df['timestamp'], validation_df[col], color='#1d4ed8', linewidth=1.8)
        axis.axhline(float(limit), color='#dc2626', linestyle='--', linewidth=1.2)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    title = f'{_controller(rollout)} Validation'
    warning = str(rollout.meta.get('misocp_health_warning', '') or '')
    axes[0].set_title(f'{title}\n{warning}' if warning else title)
    axes[-1].set_xlabel('Timestamp')
    return fig

def plot_shared_forecast_vs_actual(*rollouts: RolloutResult, figsize: tuple[float, float] | None=None):
    if not rollouts:
        raise ValueError('At least one rollout is required.')
    ref = rollouts[0]
    price_ref = _shared_price_frame(ref.step_df, controller_label=_controller(ref))
    load_ref = _shared_signal_frame(ref.agent_df, 'load', controller_label=_controller(ref))
    pv_ref = _shared_signal_frame(ref.agent_df, 'pv', controller_label=_controller(ref))
    for rollout in rollouts[1:]:
        label = _controller(rollout)
        price_ref, _ = _align_shared_actual_frames(price_ref, _shared_price_frame(rollout.step_df, controller_label=label), key_columns=['timestamp'], value_column=IMPORT_PRICE_COLUMN, controller_label=label, signal_label='import_price')
        load_ref, _ = _align_shared_actual_frames(load_ref, _shared_signal_frame(rollout.agent_df, 'load', controller_label=label), key_columns=['timestamp', 'agent_profile'], value_column='load', controller_label=label, signal_label='load')
        pv_ref, _ = _align_shared_actual_frames(pv_ref, _shared_signal_frame(rollout.agent_df, 'pv', controller_label=label), key_columns=['timestamp', 'agent_profile'], value_column='pv', controller_label=label, signal_label='pv')
    pred_rollout = _prediction_rollout(*rollouts)
    pred_label = _controller(pred_rollout)
    price_pred = _shared_price_frame(pred_rollout.step_df, controller_label=pred_label, require_predicted=True)
    load_pred = _shared_signal_frame(pred_rollout.agent_df, 'load', controller_label=pred_label, require_predicted=True)
    pv_pred = _shared_signal_frame(pred_rollout.agent_df, 'pv', controller_label=pred_label, require_predicted=True)
    price_ref, price_pred = _align_shared_actual_frames(price_ref, price_pred, key_columns=['timestamp'], value_column=IMPORT_PRICE_COLUMN, controller_label=pred_label, signal_label='import_price')
    load_ref, load_pred = _align_shared_actual_frames(load_ref, load_pred, key_columns=['timestamp', 'agent_profile'], value_column='load', controller_label=pred_label, signal_label='load')
    pv_ref, pv_pred = _align_shared_actual_frames(pv_ref, pv_pred, key_columns=['timestamp', 'agent_profile'], value_column='pv', controller_label=pred_label, signal_label='pv')
    fig, axes = plt.subplots(3, 1, figsize=figsize or (20.0, 12.0), sharex=True)
    axes[0].plot(price_ref['timestamp'], price_ref[IMPORT_PRICE_COLUMN], color='#111827', linewidth=1.9, label='Actual import price')
    axes[0].plot(price_pred['timestamp'], _price_prediction(price_pred, price_markup=float(pred_rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0))), color='#dc2626', linewidth=1.8, linestyle='--', label=f'Predicted import price ({pred_label})')
    _compare_style(axes[0], title='Import Price Forecast vs Actual', ylabel='EUR/kWh')
    axes[0].legend(loc='upper right', fontsize=_LEGEND)
    profiles = list(dict.fromkeys(load_ref['agent_profile'].astype(str)))
    handles = [Line2D([0], [0], color=_AGENT_COLORS[i % len(_AGENT_COLORS)], linewidth=2.0, label=profile) for i, profile in enumerate(profiles)]
    handles += [Line2D([0], [0], color='#111827', linewidth=2.0, linestyle='-', label='Actual'), Line2D([0], [0], color='#111827', linewidth=2.0, linestyle='--', label='Predicted')]
    for axis, signal, actual, pred in ((axes[1], 'load', load_ref, load_pred), (axes[2], 'pv', pv_ref, pv_pred)):
        for i, profile in enumerate(profiles):
            color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            act = actual.loc[actual['agent_profile'] == profile]
            prd = pred.loc[pred['agent_profile'] == profile]
            axis.plot(act['timestamp'], act[signal], color=color, linewidth=1.8)
            axis.plot(prd['timestamp'], prd[f'{signal}_pred'], color=color, linewidth=1.6, linestyle='--')
        _compare_style(axis, title=f'{signal.upper()} Forecast vs Actual', ylabel='kW', xlabel='Timestamp' if axis is axes[2] else None)
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 7), frameon=False, fontsize=_LEGEND, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig

def plot_rollout_dashboard(rollout: RolloutResult, *, figsize: tuple[float, float] | None=None):
    if rollout.step_df.empty or rollout.agent_df.empty:
        raise ValueError('Rollout is empty; nothing to plot.')
    profiles = list(rollout.meta['agent_profiles'])
    fig, axes = plt.subplots(6 + len(profiles), 1, figsize=figsize or (18.0, 2.8 * (6 + len(profiles))), sharex=True)
    axes = np.atleast_1d(axes)
    _plot_agent_forecasts(axes[:3], rollout.step_df, rollout.agent_df, profiles, title=f'Test Rollout Dashboard - {_controller(rollout)}')
    _plot_voltage(axes[3], rollout, legend=True)
    axes[3].set_ylabel('Voltage [p.u.]')
    axes[3].legend(loc='upper right', ncol=2)
    step_df = rollout.step_df
    axes[4].plot(step_df['timestamp'], step_df['base_net_load_total'], color='#111827', linewidth=1.6, label='Raw net load')
    if 'base_net_load_effective_total' in step_df.columns:
        axes[4].plot(step_df['timestamp'], step_df['base_net_load_effective_total'], color='#16a34a', linewidth=1.4, linestyle='-.', label='Post-curtail net load')
    axes[4].plot(step_df['timestamp'], step_df['net_load_total'], color='#2563eb', linewidth=1.5, linestyle='--', label='Post-action net load')
    if (limit := rollout.meta.get('trafo_limit_kw')) is not None and np.isfinite(float(limit)) and (float(limit) > 0.0):
        for sign in (1.0, -1.0):
            axes[4].axhline(sign * float(limit), color='#dc2626', linestyle=':', linewidth=1.2, label=f"Approx trafo {('+' if sign > 0 else '-')}limit ({float(limit):.1f} kW)")
    axes[4].set_ylabel('Net load')
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc='upper right')
    if 'pv_effective_total' in step_df.columns:
        axes[5].plot(step_df['timestamp'], step_df['pv_raw_total'], color='#ea580c', linewidth=1.5, label='Raw PV')
        axes[5].plot(step_df['timestamp'], step_df['pv_effective_total'], color='#16a34a', linewidth=1.5, linestyle='--', label='Effective PV')
        axes[5].bar(step_df['timestamp'], step_df['pv_curtail_total'], width=_WIDTH, color='#dc2626', alpha=0.35, label='Curtailment')
        axes[5].legend(loc='upper right')
    axes[5].set_ylabel('PV')
    axes[5].grid(True, alpha=0.25)
    _plot_battery_rows(axes[6:], rollout.agent_df, profiles)
    axes[-1].set_xlabel('Timestamp')
    fig.tight_layout()
    fig._dashboard_main_axes = list(axes)
    return fig

def plot_power_balance_bars(rollout: RolloutResult, *, figsize: tuple[float, float]=(18.0, 4.8)):
    step_df, _ = _prepare_agent_power_balance_frame(rollout.step_df, controller_label=_controller(rollout))
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    _stack_power(axis, step_df, with_legend=True)
    axis.set_title(f'Power Balance - {_controller(rollout)}')
    axis.set_ylabel('kW')
    axis.set_xlabel('Timestamp')
    axis.grid(True, axis='y', alpha=0.25)
    axis.legend(loc='upper right', ncol=4)
    fig.tight_layout()
    return fig

def plot_test_rollout(rollout: RolloutResult, *, figsize: tuple[float, float] | None=None):
    if rollout.step_df.empty or rollout.agent_df.empty:
        raise ValueError('Rollout is empty; nothing to plot.')
    profiles = list(rollout.meta['agent_profiles'])
    fig, axes = plt.subplots(3 + len(profiles), 1, figsize=figsize or (18.0, 2.8 * (3 + len(profiles))), sharex=True)
    axes = np.atleast_1d(axes)
    _plot_agent_forecasts(axes[:3], rollout.step_df, rollout.agent_df, profiles, title=f'Test Rollout - {_controller(rollout)}')
    _plot_battery_rows(axes[3:], rollout.agent_df, profiles)
    axes[-1].set_xlabel('Timestamp')
    fig.tight_layout()
    return fig

def plot_test_voltage_profile(rollout: RolloutResult, *, figsize: tuple[float, float]=(18.0, 4.8)):
    if rollout.grid_df.empty:
        raise ValueError('Rollout does not contain full-grid voltage traces.')
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    _plot_voltage(axis, rollout, legend=True)
    axis.set_title(f'Node Voltage Profile - {_controller(rollout)}')
    axis.set_ylabel('Voltage [p.u.]')
    axis.set_xlabel('Timestamp')
    axis.grid(True, alpha=0.25)
    axis.legend(loc='upper right', ncol=2)
    fig.tight_layout()
    return fig

def plot_voltage_profile_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None=None):
    if not rollouts:
        raise ValueError('At least one rollout is required.')
    fig, axes = plt.subplots(len(rollouts), 1, figsize=figsize or (20.0, max(4.0 * len(rollouts), 5.6)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        _plot_voltage(axis, rollout, legend=idx == 0)
        _compare_style(axis, title=_controller(rollout), ylabel='V [p.u.]')
        if idx == 0:
            axis.legend(loc='upper right', ncol=2, fontsize=_LEGEND)
    axes[-1].set_xlabel('Timestamp', fontsize=_LABEL)
    fig.tight_layout()
    return fig

def plot_price_prediction_comparison(*rollouts: RolloutResult, figsize: tuple[float, float]=(20.0, 5.0)):
    if not rollouts:
        raise ValueError('At least one rollout is required.')
    ref = next((r for r in rollouts if str(r.meta.get('prediction_mode', '')) == NORMAL_PREDICTION_MODE), rollouts[0])
    _require(ref.step_df, {'timestamp', IMPORT_PRICE_COLUMN, WHOLESALE_PRICE_PRED_COLUMN}, _controller(ref), 'Rollouts must contain timestamp, import_price, and wholesale_price_pred columns.')
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    axis.plot(ref.step_df['timestamp'], ref.step_df[IMPORT_PRICE_COLUMN], color='#111827', linewidth=1.8, label='Actual import price')
    groups: list[dict[str, object]] = []
    for rollout in rollouts:
        if str(rollout.meta.get('prediction_mode', '')) not in ('', NORMAL_PREDICTION_MODE):
            continue
        step_df = rollout.step_df
        if step_df.empty or WHOLESALE_PRICE_PRED_COLUMN not in step_df.columns:
            raise ValueError(f"Rollout '{_controller(rollout)}' is missing wholesale_price_pred for compare plotting.")
        pred = _price_prediction(step_df, price_markup=float(rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0)))
        stamps = pd.to_datetime(step_df['timestamp']).to_numpy()
        group = next((g for g in groups if g['timestamps'].shape == stamps.shape and np.array_equal(g['timestamps'], stamps) and (pred.shape == g['pred'].shape) and np.allclose(pred, g['pred'], equal_nan=True)), None)
        if group is None:
            groups.append({'timestamps': stamps, 'pred': pred, 'controllers': [_controller(rollout)]})
        else:
            group['controllers'].append(_controller(rollout))
    if not groups:
        groups = [{'timestamps': pd.to_datetime(ref.step_df['timestamp']).to_numpy(), 'pred': _price_prediction(ref.step_df, price_markup=float(ref.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0))), 'controllers': [_controller(ref)]}]
    for idx, group in enumerate(groups):
        controllers = list(dict.fromkeys(group['controllers']))
        label = 'Predicted import price' if len(groups) == 1 else f'Predicted import price ({controllers[0]})' if len(controllers) == 1 else f'Predicted import price ({controllers[0]} +{len(controllers) - 1})'
        axis.plot(group['timestamps'], group['pred'], color=_AGENT_COLORS[(idx + 1) % len(_AGENT_COLORS)], linewidth=1.8, linestyle='--', label=label)
    _compare_style(axis, title='Import Price And Derived Forecast', ylabel='EUR/kWh', xlabel='Timestamp')
    axis.legend(loc='upper right', ncol=2, fontsize=_LEGEND)
    fig.tight_layout()
    return fig

def plot_net_load_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None=None):
    if not rollouts:
        raise ValueError('At least one rollout is required.')
    fig, axes = plt.subplots(len(rollouts), 1, figsize=figsize or (20.0, max(4.2 * len(rollouts), 5.8)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df = _prepare_compare_net_load_frame(rollout.step_df, controller_label=_controller(rollout))
        axis.plot(step_df['timestamp'], step_df['feeder_raw_net_load_kw'], color='#111827', linewidth=1.6, label='Feeder raw net load')
        axis.plot(step_df['timestamp'], step_df['feeder_effective_net_load_kw'], color='#16a34a', linewidth=1.4, linestyle='-.', label='Feeder post-curtail net load')
        axis.plot(step_df['timestamp'], step_df['feeder_post_action_net_load_kw'], color='#2563eb', linewidth=1.5, linestyle='--', label='Feeder post-action net load')
        root_col = 'root_net_exchange_kw' if 'root_net_exchange_kw' in step_df.columns else 'pp_root_p_kw' if 'pp_root_p_kw' in step_df.columns else None
        if root_col:
            axis.plot(step_df['timestamp'], step_df[root_col], color='#7c3aed', linewidth=1.7, label='Root net exchange' if root_col == 'root_net_exchange_kw' else 'Pandapower root exchange')
        if (limit := rollout.meta.get('trafo_limit_kw')) is not None and np.isfinite(float(limit)) and (float(limit) > 0.0):
            axis.axhline(float(limit), color='#dc2626', linestyle=':', linewidth=1.2, label=f'Transformer S-limit ref (+P view) ({float(limit):.1f} kW)' if idx == 0 else None)
            axis.axhline(-float(limit), color='#dc2626', linestyle=':', linewidth=1.2, label=f'Transformer S-limit ref (-P view) ({float(limit):.1f} kW)' if idx == 0 else None)
        _compare_style(axis, title=f'{_controller(rollout)} - feeder total', ylabel='kW')
        if idx == 0:
            axis.legend(loc='upper right', fontsize=_LEGEND)
    axes[-1].set_xlabel('Timestamp', fontsize=_LABEL)
    fig.tight_layout()
    return fig

def plot_power_balance_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None=None):
    if not rollouts:
        raise ValueError('At least one rollout is required.')
    fig, axes = plt.subplots(len(rollouts), 1, figsize=figsize or (20.0, max(4.0 * len(rollouts), 5.6)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df, residual = _prepare_agent_power_balance_frame(rollout.step_df, controller_label=_controller(rollout))
        _stack_power(axis, step_df, with_legend=idx == 0)
        _compare_style(axis, title=f'{_controller(rollout)} - agent-only balance (residual<={residual:.3f} kW)', ylabel='kW')
        axis.grid(True, axis='y', alpha=0.25)
        if idx == 0:
            axis.legend(loc='upper right', ncol=4, fontsize=_LEGEND)
    axes[-1].set_xlabel('Timestamp', fontsize=_LABEL)
    fig.tight_layout()
    return fig

def plot_battery_power_and_soc_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None=None):
    if not rollouts:
        raise ValueError('At least one rollout is required.')
    fig, axes = plt.subplots(len(rollouts), 1, figsize=figsize or (20.0, max(4.2 * len(rollouts), 6.5)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df, agent_df = (rollout.step_df, rollout.agent_df)
        if step_df.empty:
            raise ValueError(f"Rollout '{_controller(rollout)}' has no step_df.")
        _require(step_df, {'battery_charge_total', 'battery_discharge_total', 'timestamp'}, _controller(rollout), "Rollout '{label}' is missing battery power columns required for compare plotting.")
        if agent_df.empty or 'soc' not in agent_df.columns:
            raise ValueError(f"Rollout '{_controller(rollout)}' is missing agent_df.soc required for compare plotting.")
        if 'agent_id' not in agent_df.columns:
            raise ValueError(f"Rollout '{_controller(rollout)}' is missing agent_df.agent_id required for compare plotting.")
        net_power = step_df['battery_discharge_total'].to_numpy(dtype=np.float32) - step_df['battery_charge_total'].to_numpy(dtype=np.float32)
        axis.bar(step_df['timestamp'], net_power, width=_WIDTH, color=['#dc2626' if v >= 0.0 else '#2563eb' for v in net_power], alpha=0.82)
        axis.axhline(0.0, color='#475569', linewidth=1.0)
        _compare_style(axis, title=_controller(rollout), ylabel='Battery Power [kW]')
        soc_axis = axis.twinx()
        groups = [c for c in ('episode_idx', 'step', 'timestamp') if c in agent_df.columns]
        if not groups:
            raise ValueError(f"Rollout '{_controller(rollout)}' is missing grouping columns required to aggregate SoC.")
        summary = agent_df.groupby(groups, as_index=False).agg(mean=('soc', 'mean'))
        first = True
        for _, frame in agent_df.sort_values(groups + ['agent_id']).groupby('agent_id', sort=True):
            soc_axis.plot(frame['timestamp'], frame['soc'], color='#94a3b8', linewidth=1.1, alpha=0.45, label='Agent SoC' if first and idx == 0 else None)
            first = False
        soc_axis.plot(summary['timestamp'], summary['mean'], color='#111827', linewidth=1.7, label='Mean SoC' if idx == 0 else None)
        soc_axis.set_ylabel('SoC', fontsize=_LABEL)
        soc_axis.set_ylim(0.0, 1.0)
        soc_axis.grid(True, alpha=0.25)
        soc_axis.tick_params(axis='both', labelsize=_TICK)
        if idx == 0:
            handles, labels = soc_axis.get_legend_handles_labels()
            axis.legend([Patch(facecolor='#dc2626', alpha=0.82, label='Discharge (+)'), Patch(facecolor='#2563eb', alpha=0.82, label='Charge (-)'), *handles], ['Discharge (+)', 'Charge (-)', *labels], loc='upper right', ncol=2, fontsize=_LEGEND)
    axes[-1].set_xlabel('Timestamp', fontsize=_LABEL)
    fig.tight_layout()
    return fig

def plot_rollout_comparison_dashboard(metrics_df: pd.DataFrame, *, figsize: tuple[float, float]=(16.0, 9.0)):
    if metrics_df.empty:
        raise ValueError('metrics_df is empty; nothing to plot.')
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    controllers = metrics_df['controller'].astype(str).tolist()
    x = np.arange(len(controllers))
    colors = [_AGENT_COLORS[i % len(_AGENT_COLORS)] for i in range(len(controllers))]
    metrics = [('purchase_cost_total', 'Purchase Cost'), ('export_subsidy_total', 'Export Subsidy'), ('objective_total', 'Objective Total'), ('voltage_violation_count', 'Voltage Violation Count'), ('trafo_penalty_total', 'Transformer Penalty'), ('line_penalty_total', 'Line Penalty')]
    for axis, (col, title) in zip(axes, metrics, strict=False):
        axis.bar(x, metrics_df[col].astype(float).to_numpy(), color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels(controllers, rotation=15, ha='right')
        axis.set_title(title)
        axis.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    return fig
