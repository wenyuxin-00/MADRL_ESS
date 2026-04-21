from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def build_misocp_validation_df(diagnostic_rows: list[dict[str, object]]) -> pd.DataFrame:
    from controllers.mpc.global_socp_mpc import _LINE_LOADING_ERR_TOL_PCT, _ROOT_POWER_ERR_TOL_KW, _TRAFO_LOADING_ERR_TOL_PCT, _VOLTAGE_ERR_TOL_PU
    rows = []
    for entry in diagnostic_rows:
        parts = [entry.get(name) for name in ('misocp_vm_pu', 'pp_vm_pu', 'misocp_line_loading_pct', 'pp_line_loading_pct', 'misocp_trafo_loading_pct', 'pp_trafo_loading_pct')]
        pp_root_p = entry.get('pp_root_p_kw', np.nan)
        root_ok = bool(entry.get('pp_root_p_available', np.isfinite(float(pp_root_p)) if pp_root_p is not None else False))
        if any((value is None for value in parts)):
            vm_err = line_err = trafo_err = root_err = np.nan
            within = False
        else:
            misocp_vm, pp_vm, misocp_line, pp_line, misocp_trafo, pp_trafo = (np.asarray(value, dtype=np.float32) for value in parts)
            vm_err = float(np.max(np.abs(misocp_vm - pp_vm)))
            line_err = float(np.max(np.abs(misocp_line - pp_line)))
            trafo_err = float(np.max(np.abs(misocp_trafo - pp_trafo)))
            root_err = float(abs(float(entry.get('root_p_kw', np.nan)) - float(pp_root_p))) if root_ok else float('nan')
            within = bool(vm_err < _VOLTAGE_ERR_TOL_PU and line_err < _LINE_LOADING_ERR_TOL_PCT and (trafo_err < _TRAFO_LOADING_ERR_TOL_PCT) and root_ok and (root_err < _ROOT_POWER_ERR_TOL_KW))
        rows.append({'controller': entry.get('controller', 'MISOCP Global Oracle'), 'episode_idx': int(entry.get('episode_idx', 0)), 'step': int(entry.get('step', entry.get('global_step', 0))), 'timestamp': entry.get('timestamp'), 'misocp_fallback': float(entry.get('misocp_fallback', 0.0)), 'misocp_time_limit_feasible': float(entry.get('misocp_time_limit_feasible', 0.0)), 'solve_time_sec': float(entry.get('solve_time_sec', np.nan)), 'max_vm_abs_err_pu': vm_err, 'max_line_loading_abs_err_pct': line_err, 'trafo_loading_abs_err_pct': trafo_err, 'root_p_abs_err_kw': root_err, 'pp_root_p_available': root_ok, 'root_power_validation_unavailable': not root_ok, 'misocp_root_p_kw': float(entry.get('root_p_kw', np.nan)), 'pp_root_p_kw': float(pp_root_p), 'root_q_kvar': float(entry.get('root_q_kvar', np.nan)), 'misocp_root_s_kva': float(entry.get('misocp_root_s_kva', np.nan)), 'pp_root_s_kva': float(entry.get('pp_root_s_kva', np.nan)), 'soc_slack_max': float(entry.get('soc_slack_max', np.nan)), 'soc_slack_mean': float(entry.get('soc_slack_mean', np.nan)), 'soc_slack_p95_global': float(entry.get('soc_slack_p95_global', np.nan)), 'background_q_base_total_kvar': float(entry.get('background_q_base_total_kvar', np.nan)), 'network_q_loss_proxy_kvar': float(entry.get('network_q_loss_proxy_kvar', np.nan)), 'root_q_residual_kvar': float(entry.get('root_q_residual_kvar', np.nan)), 'solver_feeder_gap_kw': float(entry.get('solver_feeder_gap_kw', np.nan)), 'replay_feeder_gap_kw': float(entry.get('replay_feeder_gap_kw', np.nan)), 'within_tolerance': within})
    return pd.DataFrame(rows)

def summarize_misocp_validation(validation_df: pd.DataFrame) -> pd.Series:
    from scripts.utils.misocp_notebook_helpers import _SOC_SLACK_TIGHT_P95_THRESHOLD, _interpret_linear_fit, _linear_fit_summary
    if validation_df.empty:
        return pd.Series({'validation_rows': 0, 'max_vm_abs_err_pu': np.nan, 'max_line_loading_abs_err_pct': np.nan, 'max_trafo_loading_abs_err_pct': np.nan, 'max_root_p_abs_err_kw': np.nan, 'steps_outside_tolerance': 0, 'root_power_validation_unavailable': True, 'pp_root_p_available_ratio': 0.0, 'max_root_q_kvar': np.nan, 'max_soc_slack': np.nan, 'mean_soc_slack': np.nan, 'p95_soc_slack': np.nan, 'soc_relaxation_is_tight': False, 'max_solver_feeder_gap_kw': np.nan, 'mean_abs_solver_feeder_gap_kw': np.nan, 'max_replay_feeder_gap_kw': np.nan, 'mean_abs_replay_feeder_gap_kw': np.nan, 'background_q_base_total_kvar': np.nan, 'max_network_q_loss_proxy_kvar': np.nan, 'max_abs_root_q_residual_kvar': np.nan, 'root_p_fit_k': np.nan, 'root_p_fit_b': np.nan, 'root_p_fit_r2': np.nan, 'root_p_fit_interpretation': 'insufficient_data', 'root_s_fit_k': np.nan, 'root_s_fit_b': np.nan, 'root_s_fit_r2': np.nan, 'root_s_fit_interpretation': 'insufficient_data'}, name='misocp_validation_summary')
    available = validation_df['pp_root_p_available'].fillna(False).astype(bool)
    root_p_fit = _linear_fit_summary(validation_df.loc[available, 'misocp_root_p_kw'].to_numpy(dtype=np.float64), validation_df.loc[available, 'pp_root_p_kw'].to_numpy(dtype=np.float64))
    root_s_fit = _linear_fit_summary(validation_df['misocp_root_s_kva'].to_numpy(dtype=np.float64), validation_df['pp_root_s_kva'].to_numpy(dtype=np.float64))
    p95_series = validation_df['soc_slack_p95_global'].dropna()
    p95 = float(p95_series.iloc[0]) if not p95_series.empty else float('nan')
    q_base = validation_df['background_q_base_total_kvar'].dropna()
    summary = {'validation_rows': int(len(validation_df)), 'max_vm_abs_err_pu': float(validation_df['max_vm_abs_err_pu'].max()), 'max_line_loading_abs_err_pct': float(validation_df['max_line_loading_abs_err_pct'].max()), 'max_trafo_loading_abs_err_pct': float(validation_df['trafo_loading_abs_err_pct'].max()), 'max_root_p_abs_err_kw': float(validation_df.loc[available, 'root_p_abs_err_kw'].max()) if bool(available.any()) else float('nan'), 'steps_outside_tolerance': int((~validation_df['within_tolerance'].fillna(False)).sum()), 'root_power_validation_unavailable': bool(not available.all()), 'pp_root_p_available_ratio': float(np.mean(available.to_numpy(dtype=np.float32))), 'max_root_q_kvar': float(validation_df['root_q_kvar'].max()), 'max_soc_slack': float(validation_df['soc_slack_max'].max()), 'mean_soc_slack': float(validation_df['soc_slack_mean'].mean()), 'p95_soc_slack': p95, 'soc_relaxation_is_tight': bool(np.isfinite(p95) and p95 < _SOC_SLACK_TIGHT_P95_THRESHOLD), 'max_solver_feeder_gap_kw': float(np.abs(validation_df['solver_feeder_gap_kw']).max()), 'mean_abs_solver_feeder_gap_kw': float(np.abs(validation_df['solver_feeder_gap_kw']).mean()), 'max_replay_feeder_gap_kw': float(np.abs(validation_df['replay_feeder_gap_kw']).max()), 'mean_abs_replay_feeder_gap_kw': float(np.abs(validation_df['replay_feeder_gap_kw']).mean()), 'background_q_base_total_kvar': float(q_base.iloc[0]) if not q_base.empty else float('nan'), 'max_network_q_loss_proxy_kvar': float(validation_df['network_q_loss_proxy_kvar'].max()), 'max_abs_root_q_residual_kvar': float(np.abs(validation_df['root_q_residual_kvar']).max())}
    for prefix, fit in (('root_p', root_p_fit), ('root_s', root_s_fit)):
        summary[f'{prefix}_fit_k'] = float(fit['k'])
        summary[f'{prefix}_fit_b'] = float(fit['b'])
        summary[f'{prefix}_fit_r2'] = float(fit['r2'])
        summary[f'{prefix}_fit_interpretation'] = _interpret_linear_fit(float(fit['k']), float(fit['b']), float(fit['r2']))
    return pd.Series(summary, name='misocp_validation_summary')

def plot_full_horizon_power_balance(step_df: pd.DataFrame, *, controller_label: str, dt_hours: float, figsize: tuple[float, float]=(18.0, 4.8)):
    if step_df.empty:
        raise ValueError('step_df is empty; nothing to plot.')
    fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    width = pd.Timedelta(hours=max(float(dt_hours) * 0.8, 0.001)) if pd.api.types.is_datetime64_any_dtype(pd.Index(step_df['timestamp']).dtype) or isinstance(pd.Index(step_df['timestamp']).dtype, pd.DatetimeTZDtype) else 0.8
    demand = [('agent_load_kw', 'Agent load', '#111827'), ('fixed_load_kw', 'Non-agent fixed load', '#4b5563'), ('battery_charge_kw', 'Battery charge', '#dc2626'), ('grid_export_kw', 'Grid export', '#f59e0b')]
    supply = [('pv_effective_kw', 'PV effective', '#16a34a'), ('fixed_generation_kw', 'Non-agent fixed generation', '#15803d'), ('grid_import_kw', 'Grid import', '#2563eb'), ('battery_discharge_kw', 'Battery discharge', '#7c3aed')]
    for specs, sign in ((demand, 1.0), (supply, -1.0)):
        bottom = np.zeros(len(step_df), dtype=np.float32)
        for col, label, color in specs:
            extra = {'hatch': '//', 'edgecolor': '#f59e0b'} if col == 'grid_export_kw' else {}
            values = sign * step_df[col].to_numpy(dtype=np.float32)
            axis.bar(step_df['timestamp'], values, width=width, bottom=bottom, color=color, alpha=0.82, label=label, linewidth=0.0, zorder=2, **extra)
            bottom += values
    axis.plot(step_df['timestamp'], step_df['pv_curtail_kw'].to_numpy(dtype=np.float32), color='#ef4444', linewidth=1.3, linestyle='--', label='PV curtailment', zorder=4)
    axis.plot(step_df['timestamp'], step_df['balance_residual_kw'].to_numpy(dtype=np.float32), color='#0f172a', linewidth=1.35, label='Balance residual', zorder=4)
    axis.axhline(0.0, color='#111827', linewidth=1.0, zorder=5)
    axis.set(xlabel='Timestamp', ylabel='Power [kW]', title=f'Power Balance - {controller_label}')
    axis.grid(True, axis='y', alpha=0.25)
    axis.legend(loc='upper right', ncol=3)
    fig._misocp_balance_axes = [axis]
    return fig

def plot_full_horizon_voltage(step_df: pd.DataFrame, grid_voltage_df: pd.DataFrame, *, agent_bus_ids: list[int] | tuple[int, ...], v_min_pu: float, v_max_pu: float, controller_label: str, figsize: tuple[float, float]=(18.0, 5.4)):
    if step_df.empty or grid_voltage_df.empty:
        raise ValueError('Voltage plotting requires non-empty step_df and grid_voltage_df.')
    fig, axis = plt.subplots(figsize=figsize)
    background = grid_voltage_df.loc[~grid_voltage_df['is_agent_bus'], ['timestamp', 'bus_id', 'vm_pu']].pivot_table(index='timestamp', columns='bus_id', values='vm_pu', aggfunc='first').sort_index()
    if background.shape[1] > 0:
        axis.plot(background.index.to_numpy(), background.to_numpy(dtype=np.float32), color='#cbd5e1', linewidth=0.9, alpha=0.35, zorder=1)
    for idx, bus_id in enumerate(agent_bus_ids):
        frame = grid_voltage_df.loc[grid_voltage_df['bus_id'] == int(bus_id)]
        if not frame.empty:
            axis.plot(frame['timestamp'], frame['vm_pu'], color=['#2563eb', '#dc2626', '#16a34a', '#ea580c', '#7c3aed', '#0891b2'][idx % 6], linewidth=1.9, alpha=0.95, label=f'Agent bus {int(bus_id)}', zorder=3)
    for col, label, style in (('min_vm_pu', 'Min vm', '--'), ('mean_vm_pu', 'Mean vm', '-.'), ('max_vm_pu', 'Max vm', ':')):
        axis.plot(step_df['timestamp'], step_df[col], color={'min_vm_pu': '#0f172a', 'mean_vm_pu': '#475569', 'max_vm_pu': '#334155'}[col], linewidth=1.4 if col != 'mean_vm_pu' else 1.3, linestyle=style, label=label)
    axis.axhline(float(v_min_pu), color='#dc2626', linestyle='--', linewidth=1.1, label='V min')
    axis.axhline(float(v_max_pu), color='#ea580c', linestyle='--', linewidth=1.1, label='V max')
    axis.set(title=f'Node Voltage Profile - {controller_label}', ylabel='Voltage [p.u.]', xlabel='Timestamp')
    axis.grid(True, alpha=0.25)
    axis.legend(loc='upper right', ncol=3)
    fig.tight_layout()
    fig._misocp_voltage_axis = axis
    return fig

def plot_full_horizon_net_load(step_df: pd.DataFrame, *, trafo_limit_kw: float | None, controller_label: str, figsize: tuple[float, float]=(18.0, 8.2)):
    from scripts.utils.misocp_notebook_helpers import _resolve_agent_net_load_columns, _resolve_feeder_net_load_columns
    if step_df.empty:
        raise ValueError('step_df is empty; nothing to plot.')
    agent_raw, agent_effective, agent_post = _resolve_agent_net_load_columns(step_df)
    feeder_raw, feeder_effective, feeder_post = _resolve_feeder_net_load_columns(step_df)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    feeder_axis, agent_axis = axes
    for values, label, color, style in ((feeder_raw, 'Feeder raw net load', '#111827', '-'), (feeder_effective, 'Feeder post-curtail net load', '#16a34a', '-.'), (feeder_post, 'Feeder post-action net load', '#2563eb', '--')):
        if values is not None:
            feeder_axis.plot(step_df['timestamp'], values, color=color, linewidth=1.6 if style == '-' else 1.4 if style == '-.' else 1.5, linestyle=style, label=label)
    for col, label, color, style in (('root_net_exchange_kw', 'MISOCP root net exchange', '#7c3aed', '-'), ('pp_root_p_kw', 'Pandapower root exchange', '#0f766e', ':')):
        if col in step_df.columns:
            feeder_axis.plot(step_df['timestamp'], step_df[col], color=color, linewidth=1.7 if style == '-' else 1.4, linestyle=style, label=label)
    if trafo_limit_kw is not None and np.isfinite(float(trafo_limit_kw)):
        feeder_axis.axhline(float(trafo_limit_kw), color='#b91c1c', linestyle=':', linewidth=1.2, label='Transformer S-limit ref (+P view)')
        feeder_axis.axhline(-float(trafo_limit_kw), color='#b91c1c', linestyle=':', linewidth=1.2, label='Transformer S-limit ref (-P view)')
    feeder_axis.set(title=f'Net Load - {controller_label}\nFeeder total (transformer S-limit shown as active-power reference)', ylabel='Power [kW]')
    feeder_axis.grid(True, alpha=0.25)
    feeder_axis.legend(loc='upper right', ncol=2)
    for values, label, color, style in ((agent_raw, 'Agent raw net load', '#111827', '-'), (agent_effective, 'Agent post-curtail net load', '#16a34a', '-.'), (agent_post, 'Agent post-action net load', '#2563eb', '--')):
        agent_axis.plot(step_df['timestamp'], values, color=color, linewidth=1.6 if style == '-' else 1.4 if style == '-.' else 1.5, linestyle=style, label=label)
    agent_axis.set(title='Agent-only aggregate', xlabel='Timestamp', ylabel='Power [kW]')
    agent_axis.grid(True, alpha=0.25)
    agent_axis.legend(loc='upper right')
    fig._misocp_net_load_axes = list(axes)
    return fig

def plot_root_exchange_alignment(step_df: pd.DataFrame, *, controller_label: str, figsize: tuple[float, float]=(14.0, 4.5)) -> plt.Figure:
    fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    axis.plot(step_df['timestamp'], step_df['root_net_exchange_kw'], color='#2563eb', linewidth=1.6, label='MISOCP root net exchange')
    if 'pp_root_p_kw' in step_df.columns:
        axis.plot(step_df['timestamp'], step_df['pp_root_p_kw'], color='#f97316', linewidth=1.4, linestyle='--', label='Pandapower root exchange')
    axis.plot(step_df['timestamp'], step_df['feeder_post_action_net_load_kw'], color='#16a34a', linewidth=1.4, linestyle='-.', label='Feeder post-action net load')
    axis.axhline(0.0, color='#111827', linewidth=0.8, alpha=0.6)
    axis.set(title=f'{controller_label}: Root Exchange Alignment', xlabel='Timestamp', ylabel='Power [kW]')
    axis.grid(True, alpha=0.25)
    axis.legend(loc='upper right')
    return fig

def plot_misocp_validation_scatter_panel(validation_df: pd.DataFrame, *, controller_label: str, figsize: tuple[float, float]=(12.0, 4.8)) -> plt.Figure:
    from scripts.utils.misocp_notebook_helpers import _interpret_linear_fit, _linear_fit_summary
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    specs = [('misocp_root_p_kw', 'pp_root_p_kw', 'Root P fit', 'MISOCP root P [kW]', 'Pandapower root P [kW]'), ('misocp_root_s_kva', 'pp_root_s_kva', 'Root S fit', 'MISOCP root S [kVA]', 'Pandapower root S [kVA]')]
    for axis, (x_col, y_col, title, xlabel, ylabel) in zip(axes, specs, strict=False):
        x = validation_df.get(x_col, pd.Series(dtype=np.float32)).to_numpy(dtype=np.float64)
        y = validation_df.get(y_col, pd.Series(dtype=np.float32)).to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[mask], y[mask], s=18, alpha=0.75, color='#2563eb')
        fit = _linear_fit_summary(x[mask], y[mask])
        note = _interpret_linear_fit(float(fit['k']), float(fit['b']), float(fit['r2']))
        if int(np.sum(mask)) >= 2 and np.isfinite(float(fit['k'])) and np.isfinite(float(fit['b'])):
            fit_x = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), num=100, dtype=np.float64)
            axis.plot(fit_x, float(fit['k']) * fit_x + float(fit['b']), color='#ef4444', linewidth=1.2, label='linear fit')
        if int(np.sum(mask)) >= 1:
            ident = np.concatenate([x[mask], y[mask]])
            axis.plot([float(np.min(ident)), float(np.max(ident))], [float(np.min(ident)), float(np.max(ident))], color='#6b7280', linewidth=1.0, linestyle=':')
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(True, alpha=0.25)
        axis.text(0.03, 0.97, f"k={fit['k']:.4g}\nb={fit['b']:.4g}\nR²={fit['r2']:.4g}\n{note}", transform=axis.transAxes, va='top', ha='left', fontsize=9, bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#d1d5db'})
    fig.suptitle(f'{controller_label}: Validation Fit Diagnostics', fontsize=12)
    return fig
