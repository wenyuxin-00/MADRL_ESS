from __future__ import annotations
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Mapping
import numpy as np,pandas as pd
from controllers.madrl.safety_projector import build_safety_local_numpy,compute_action_gap_metrics_numpy,merge_action_info_into_step_info,require_strict_local_action_feasibility,resolve_local_action_penalty_settings
from envs.grid.deployments import resolve_fixed_battery_spec
from predictors.lstm_forecaster import get_default_lstm_artifact_dir
from predictors.mainline_forecast import build_mainline_forecast_controls,ensure_mainline_forecast_ready,resolve_mainline_forecast_controls
from scripts.plots.grid_notebook_plotting import plot_battery_power_and_soc_comparison,plot_global_misocp_validation,plot_net_load_comparison,plot_power_balance_comparison,plot_price_prediction_comparison,plot_rollout_dashboard,plot_voltage_profile_comparison
from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN,IMPORT_PRICE_MARKUP_KEY,IMPORT_PRICE_PRED_COLUMN,IMPORT_PRICE_SEQ_FIELD,WHOLESALE_PRICE_PRED_COLUMN,WHOLESALE_PRICE_SEQ_FIELD,WHOLESALE_PRICE_SIGNAL,canonicalize_step_price_frame,derive_import_price,derive_import_price_seq,get_import_price_markup,require_import_price_markup
PERFECT_PREDICTION_MODE,NORMAL_PREDICTION_MODE='perfect','normal'
ORACLE_EVAL_MODE,FORECAST_EVAL_MODE='oracle_eval','forecast_eval'
def normalize_prediction_mode(prediction_mode:str)->str:
	normalized=str(prediction_mode).strip().lower()
	if normalized in{'perfect','perfect_prediction','perfect prediction'}:return PERFECT_PREDICTION_MODE
	if normalized in{'normal','normal_prediction','normal prediction'}:return NORMAL_PREDICTION_MODE
	raise ValueError(f"prediction_mode must be one of {{'perfect', 'normal'}}, got '{prediction_mode}'.")
def normalize_date_input(value:str|int|None)->str|None:
	if value in(None,''):return
	text_value=str(value).strip();return f"{text_value[:4]}-{text_value[4:6]}-{text_value[6:]}"if len(text_value)==8 and text_value.isdigit()else str(pd.Timestamp(text_value).date())
def normalize_agent_scale(scale:float|list[float]|tuple[float,...],*,n_agents:int,name:str)->list[float]:
	values=np.asarray(scale if isinstance(scale,(list,tuple,np.ndarray))else[scale],dtype=np.float32)
	if values.size==1:values=np.repeat(values,n_agents)
	if values.size!=n_agents:raise ValueError(f"{name} should provide {n_agents} value(s), got {values.size}.")
	if np.any(values<.0):raise ValueError(f"{name} must be non-negative, got {values.tolist()}.")
	return values.astype(np.float32).tolist()
def resolve_battery_controls(cfg,battery_controls:Mapping[str,object]|None=None)->dict[str,object]:
	controls=dict(battery_controls or{});capacity_kwh,c_rate,p_max_kw=resolve_fixed_battery_spec(controls.get('battery_capacity',cfg.env.battery_capacity),controls.get('max_charge_rate',cfg.env.max_charge_rate),n_agents=int(cfg.env.num_agents));resolved={'mode':'fixed','battery_capacity':list(capacity_kwh),'max_charge_rate':float(c_rate),'p_max_kw':list(p_max_kw),'efficiency':float(controls.get('efficiency',cfg.env.efficiency)),'init_soc':float(controls.get('init_soc',cfg.env.init_soc)),'soc_min':float(controls.get('soc_min',cfg.env.soc_min)),'soc_max':float(controls.get('soc_max',cfg.env.soc_max)),'soc_target':float(controls.get('soc_target',cfg.env.soc_target))}
	if not .0<resolved['efficiency']<=1.:raise ValueError(f"efficiency must be in (0, 1], got {resolved['efficiency']}.")
	if not .0<=resolved['soc_min']<=resolved['soc_max']<=1.:raise ValueError(f"Invalid SoC range: soc_min={resolved['soc_min']}, soc_max={resolved['soc_max']}.")
	for field_name in('init_soc','soc_target'):
		if not .0<=resolved[field_name]<=1.:raise ValueError(f"{field_name} must be in [0, 1], got {resolved[field_name]}.")
	return resolved
def resolve_forecast_backend(prediction_mode:str,future_horizon:int)->str:
	mode=normalize_prediction_mode(prediction_mode)
	if mode==PERFECT_PREDICTION_MODE:return'perfect'
	if int(future_horizon)<=0:raise ValueError('Normal prediction mode requires future_horizon > 0 for the LSTM forecaster.')
	return'lstm'
def resolve_evaluation_mode(prediction_mode:str)->str:mode=normalize_prediction_mode(prediction_mode);return ORACLE_EVAL_MODE if mode==PERFECT_PREDICTION_MODE else FORECAST_EVAL_MODE
def resolve_prediction_mode_from_forecast_backend(forecast_backend:str)->str:normalized=str(forecast_backend).strip().lower();return PERFECT_PREDICTION_MODE if normalized=='perfect'else NORMAL_PREDICTION_MODE
def apply_notebook_experiment_settings(cfg,*,prediction_mode:str,test_start_date:str|int|None,test_end_date:str|int|None,agent_profiles:list[str]|None=None,agent_bus_ids:list[int]|None=None,load_scale:float|list[float]|None=None,pv_scale:float|list[float]|None=None,future_horizon:int|None=None,battery_controls:Mapping[str,object]|None=None,forecast_controls:Mapping[str,object]|None=None,train_year:int|None=None,test_year:int|None=None)->dict[str,object]:
	signal_fields=[WHOLESALE_PRICE_SEQ_FIELD.removesuffix('_seq'),'load','pv'];cfg.obs.local_features=['calendar_time','soc'];cfg.obs.sequence_features=cfg.forecast.target_signals=signal_fields;cfg.data.agent_profiles=list(cfg.data.agent_profiles if agent_profiles is None else agent_profiles);cfg.env.num_agents=int(len(cfg.data.agent_profiles));cfg.env.future_horizon=int(cfg.env.future_horizon if future_horizon is None else future_horizon);resolved_agent_bus_ids=list(cfg.grid.agent_bus_ids if agent_bus_ids is None else agent_bus_ids)
	if len(resolved_agent_bus_ids)<cfg.env.num_agents:raise ValueError(f"grid.agent_bus_ids only provides {len(resolved_agent_bus_ids)} buses, but {cfg.env.num_agents} agents were requested.")
	cfg.grid.agent_bus_ids=[int(bus_id)for bus_id in resolved_agent_bus_ids[:cfg.env.num_agents]]
	for(field_name,value)in(('train_year',train_year),('test_year',test_year)):
		if value is not None:setattr(cfg.data,field_name,int(value))
	resolved_test_start_date, resolved_test_end_date = normalize_date_input(test_start_date), normalize_date_input(test_end_date)
	if resolved_test_start_date is not None:cfg.data.test_start_date=resolved_test_start_date
	if resolved_test_end_date is not None:cfg.data.test_end_date=resolved_test_end_date
	resolved_load_scale,resolved_pv_scale=cfg.data.resolved_load_scale(cfg.env.num_agents)if load_scale is None else load_scale,cfg.data.resolved_pv_scale(cfg.env.num_agents)if pv_scale is None else pv_scale;cfg.data.load_scale=normalize_agent_scale(resolved_load_scale,n_agents=cfg.env.num_agents,name='load_scale');cfg.data.pv_scale=normalize_agent_scale(resolved_pv_scale,n_agents=cfg.env.num_agents,name='pv_scale');resolved_battery_controls=resolve_battery_controls(cfg,battery_controls)
	for(field_name,value)in{'battery_capacity':list(resolved_battery_controls['battery_capacity']),'max_charge_rate':resolved_battery_controls['max_charge_rate'],'efficiency':resolved_battery_controls['efficiency'],'init_soc':resolved_battery_controls['init_soc'],'soc_min':resolved_battery_controls['soc_min'],'soc_max':resolved_battery_controls['soc_max'],'soc_target':resolved_battery_controls['soc_target']}.items():setattr(cfg.env,field_name,value)
	if cfg.data.pv_capacity_kw and len(cfg.data.pv_capacity_kw)!=cfg.env.num_agents:cfg.data.pv_capacity_kw=[]
	resolved_prediction_mode=normalize_prediction_mode(prediction_mode);cfg.forecast.type=resolve_forecast_backend(resolved_prediction_mode,cfg.env.future_horizon);resolved_forecast_controls=resolve_mainline_forecast_controls(cfg,forecast_controls)
	for(field_name,value)in{'target_signals':list(resolved_forecast_controls['target_signals']),'history_window':int(resolved_forecast_controls['history_window']),'load_model_mode':str(resolved_forecast_controls['load_model_mode']),'load_component_split':bool(resolved_forecast_controls['load_component_split']),'load_scaler_type':str(resolved_forecast_controls['load_scaler_type']),'load_time_feature_mode':str(resolved_forecast_controls['load_time_feature_mode']),'pv_time_feature_mode':str(resolved_forecast_controls['pv_time_feature_mode']),'load_hybrid_mode':str(resolved_forecast_controls['load_hybrid_mode']),'load_baseline_mode':str(resolved_forecast_controls['load_baseline_mode']),'pv_postprocess_mode':str(resolved_forecast_controls['pv_postprocess_mode']),'auto_train_missing':bool(resolved_forecast_controls['auto_train_missing']),'signal_training_overrides':dict(resolved_forecast_controls['signal_training_overrides'])}.items():setattr(cfg.forecast,field_name,value)
	cfg.runtime.observation_normalization_state=None;artifact_root=resolved_forecast_controls['artifact_root'];cfg.forecast.lstm_artifact_root=artifact_root or str(get_default_lstm_artifact_dir())if cfg.forecast.type=='lstm'else artifact_root;return{'prediction_mode':resolved_prediction_mode,'evaluation_mode':resolve_evaluation_mode(resolved_prediction_mode),'forecast_backend':cfg.forecast.type,'future_horizon':int(cfg.env.future_horizon),'test_start_date':cfg.data.test_start_date,'test_end_date':cfg.data.test_end_date,'agent_profiles':list(cfg.data.agent_profiles),'load_scale':list(cfg.data.load_scale),'pv_scale':list(cfg.data.pv_scale),'battery':dict(resolved_battery_controls),'forecast':{'artifact_root':cfg.forecast.lstm_artifact_root,'target_signals':list(cfg.forecast.target_signals),'history_window':int(cfg.forecast.history_window),'auto_train_missing':bool(cfg.forecast.auto_train_missing),'signal_training_overrides':dict(cfg.forecast.signal_training_overrides),'load_component_split':bool(cfg.forecast.load_component_split),'load_scaler_type':str(cfg.forecast.load_scaler_type)},'train_year':int(cfg.data.train_year),'test_year':int(cfg.data.test_year),'agent_bus_ids':list(cfg.grid.agent_bus_ids)}
def ensure_forecast_ready(cfg)->dict[str,object]|None:return ensure_mainline_forecast_ready(cfg)
def _require_normal_comparison_runtime_contract(cfg,*,prediction_mode:str)->None:
	if normalize_prediction_mode(prediction_mode)!=NORMAL_PREDICTION_MODE:return
	runtime_cfg,forecast_cfg=getattr(cfg,'runtime',None),getattr(cfg,'forecast',None);shared_data_dir=str(getattr(runtime_cfg,'shared_data_dir',None)or'').strip();shared_data_signature=str(getattr(runtime_cfg,'shared_data_signature',None)or'').strip();missing_fields=[]
	if not shared_data_dir:missing_fields.append('cfg.runtime.shared_data_dir')
	if not shared_data_signature:missing_fields.append('cfg.runtime.shared_data_signature')
	if getattr(forecast_cfg,'auto_train_missing',None)is not False:missing_fields.append('cfg.forecast.auto_train_missing=False')
	if missing_fields:raise ValueError("Normal comparison mode refuses the removed live-LSTM fallback path. The original cfg is missing the canonical forecast shared-data contract required before build_comparison_cfg(...): "+', '.join(missing_fields)+". Expected cfg.runtime.shared_data_dir and cfg.runtime.shared_data_signature loaded from 'notebooks/record/forecast/lstm/shared_data_record.json', with cfg.forecast.auto_train_missing=False. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun notebooks/madrl/local_MPC.ipynb or notebooks/madrl/ADMM_mpc.ipynb.")
def build_comparison_cfg(cfg,*,prediction_mode:str):
	_require_normal_comparison_runtime_contract(cfg,prediction_mode=prediction_mode)
	comparison_cfg=deepcopy(cfg);resolved_type=resolve_forecast_backend(prediction_mode,comparison_cfg.env.future_horizon)
	if resolved_type!=str(cfg.forecast.type):comparison_cfg.runtime.shared_data_dir=comparison_cfg.runtime.shared_data_signature=comparison_cfg.runtime.forecast_ready=None
	comparison_cfg.forecast.type=resolved_type
	if comparison_cfg.forecast.type=='lstm'and comparison_cfg.forecast.lstm_artifact_root is None:comparison_cfg.forecast.lstm_artifact_root=get_default_lstm_artifact_dir()
	return comparison_cfg
def _aligned_prediction(previous_obs:dict|None,current_obs:dict,field_name:str):
	current_value=np.asarray(current_obs[field_name],dtype=np.float32)
	if previous_obs is None:
		if current_value.ndim==1:return float(current_value[0])
		return current_value[...,0].astype(np.float32)
	previous_value=np.asarray(previous_obs[field_name],dtype=np.float32);forecast_index=1 if previous_value.shape[-1]>1 else 0
	if previous_value.ndim==1:return float(previous_value[forecast_index])
	return previous_value[...,forecast_index].astype(np.float32)
def _step_timestamp(reset_info:dict[str,object],step_idx:int)->pd.Timestamp:
	episode_meta=dict(reset_info.get('episode_meta',{}));timestamps=episode_meta.get('timestamps')or[]
	if step_idx<len(timestamps):return pd.Timestamp(timestamps[step_idx])
	return pd.Timestamp(step_idx,unit='m')
def _approx_trafo_limit_kw(env,*,loading_limit_pct:float|None=None)->float|None:
	net=getattr(getattr(env,'_grid_core',None),'net',None);trafo_table=getattr(net,'trafo',None)
	if trafo_table is None or len(trafo_table)==0:return
	if'sn_mva'not in trafo_table:return
	sn_mva=np.asarray(trafo_table['sn_mva'],dtype=np.float64).reshape(-1)
	if sn_mva.size==0:return
	limit_scale=float(getattr(getattr(env,'_grid_cfg',None),'line_max_loading_pct',1e2)if loading_limit_pct is None else loading_limit_pct)/1e2;return float(np.sum(sn_mva)*max(limit_scale,.0)*1e3)
def _fixed_feeder_components_kw(env)->tuple[float,float]:
	net=getattr(getattr(env,'_grid_core',None),'net',None)
	if net is None:return .0,.0
	agent_bus_set=set(int(bus_id)for bus_id in getattr(getattr(env,'_grid_core',None),'agent_bus_ids',[]));fixed_load_kw=.0;fixed_generation_kw=.0;load_table=getattr(net,'load',None)
	if load_table is not None and not load_table.empty and'bus'in load_table and'p_mw'in load_table:fixed_load_kw+=float(np.asarray(load_table.loc[~load_table['bus'].isin(agent_bus_set),'p_mw'],dtype=np.float64).sum()*1e3)
	sgen_table=getattr(net,'sgen',None)
	if sgen_table is not None and not sgen_table.empty and'bus'in sgen_table and'p_mw'in sgen_table:fixed_generation_kw+=float(np.asarray(sgen_table.loc[~sgen_table['bus'].isin(agent_bus_set),'p_mw'],dtype=np.float64).sum()*1e3)
	return max(fixed_load_kw,.0),max(fixed_generation_kw,.0)
@dataclass
class RolloutResult:step_df:pd.DataFrame;agent_df:pd.DataFrame;grid_df:pd.DataFrame;summary:pd.DataFrame;meta:dict[str,object]
def _build_rollout_records(*,controller_label:str,episode_idx:int,step_in_episode:int,timestamp,info:dict[str,object],load_pred,pv_pred,wholesale_price_pred:float,import_price_pred:float,agent_profiles:list[str],bus_ids:list[int],agent_bus_set:set[int],dt_hours:float,export_subsidy:float,fixed_load_kw:float,fixed_generation_kw:float,global_step:int|None=None)->dict[str,object]:
	load_values=np.asarray(info['load'],dtype=np.float32);pv_values=np.asarray(info['pv'],dtype=np.float32);base_net_load=np.asarray(info.get('base_net_load',load_values-pv_values),dtype=np.float32);base_net_load_effective=np.asarray(info.get('base_net_load_effective',base_net_load),dtype=np.float32);battery_power=np.asarray(info['e_bat'],dtype=np.float32);battery_power_req=np.asarray(info.get('e_bat_req',battery_power),dtype=np.float32);net_load=np.asarray(info.get('net_load',base_net_load_effective+battery_power),dtype=np.float32);pv_raw=np.asarray(info.get('pv_raw',pv_values),dtype=np.float32);pv_effective=np.asarray(info.get('pv_effective',pv_raw),dtype=np.float32);pv_curtail=np.asarray(info.get('pv_curtail',pv_raw-pv_effective),dtype=np.float32);grid_import=np.asarray(info.get('grid_import_kw',np.maximum(net_load,.0)),dtype=np.float32);grid_export=np.asarray(info.get('grid_export_kw',np.maximum(-net_load,.0)),dtype=np.float32);battery_charge=np.clip(battery_power,.0,None).astype(np.float32);battery_discharge=np.maximum(-battery_power,.0).astype(np.float32);line_loading_pct=np.asarray(info.get('line_loading_pct',np.zeros(0,dtype=np.float32)),dtype=np.float32);trafo_loading_pct=np.asarray(info.get('trafo_loading_pct',np.zeros(0,dtype=np.float32)),dtype=np.float32);pp_root_p_array=np.asarray(info.get('trafo_p_signed_kw',np.zeros(1,dtype=np.float32)),dtype=np.float32).reshape(-1);pp_root_p_kw=float(pp_root_p_array.sum())if pp_root_p_array.size else float('nan');actual_price=np.float32(info[IMPORT_PRICE_COLUMN]);purchase_cost_per_agent=(grid_import*np.float32(dt_hours)*actual_price).astype(np.float32);export_subsidy_per_agent=(grid_export*np.float32(dt_hours)*np.float32(export_subsidy)).astype(np.float32);storage_purchase_cost_per_agent=(battery_charge*np.float32(dt_hours)*actual_price).astype(np.float32);storage_sale_revenue_per_agent=(battery_discharge*np.float32(dt_hours)*actual_price).astype(np.float32);storage_total_profit_per_agent=(storage_sale_revenue_per_agent-storage_purchase_cost_per_agent).astype(np.float32);storage_objective_per_agent=(-storage_total_profit_per_agent).astype(np.float32);soc_penalty=np.asarray(info.get('r_soc_pen',info.get('r_action_pen',np.zeros(len(agent_profiles),dtype=np.float32))),dtype=np.float32);voltage_penalty_per_agent=np.asarray(info.get('r_safe_v',np.zeros(len(agent_profiles),dtype=np.float32)),dtype=np.float32);line_penalty_per_agent=np.asarray(info.get('r_safe_line',np.zeros(len(agent_profiles),dtype=np.float32)),dtype=np.float32);trafo_penalty_per_agent=np.asarray(info.get('r_safe_trafo',np.zeros(len(agent_profiles),dtype=np.float32)),dtype=np.float32);objective_per_agent=(purchase_cost_per_agent-export_subsidy_per_agent+soc_penalty+voltage_penalty_per_agent+line_penalty_per_agent+trafo_penalty_per_agent).astype(np.float32);agent_raw_net_load_kw,agent_effective_net_load_kw,agent_post_action_net_load_kw=float(np.sum(base_net_load)),float(np.sum(base_net_load_effective)),float(np.sum(net_load));feeder_raw_net_load_kw,feeder_effective_net_load_kw,feeder_post_action_net_load_kw=float(agent_raw_net_load_kw+fixed_load_kw-fixed_generation_kw),float(agent_effective_net_load_kw+fixed_load_kw-fixed_generation_kw),float(agent_post_action_net_load_kw+fixed_load_kw-fixed_generation_kw);voltage_penalty_total=float(np.mean(voltage_penalty_per_agent))if voltage_penalty_per_agent.size else .0;line_penalty_total=float(np.mean(line_penalty_per_agent))if line_penalty_per_agent.size else .0;trafo_penalty_total=float(np.mean(trafo_penalty_per_agent))if trafo_penalty_per_agent.size else .0;soc_penalty_total,purchase_cost_total,export_subsidy_total=float(np.sum(soc_penalty)),float(np.sum(purchase_cost_per_agent)),float(np.sum(export_subsidy_per_agent));storage_purchase_cost_eur=float(np.sum(storage_purchase_cost_per_agent));storage_sale_revenue_eur=float(np.sum(storage_sale_revenue_per_agent));storage_total_profit_eur=float(np.sum(storage_total_profit_per_agent));storage_objective_eur=float(-storage_total_profit_eur);objective_total=float(purchase_cost_total-export_subsidy_total+soc_penalty_total+voltage_penalty_total+line_penalty_total+trafo_penalty_total);vm_pu=np.asarray(info.get('vm_pu',np.zeros(len(bus_ids),dtype=np.float32)),dtype=np.float32);step_row={'controller':controller_label,'episode_idx':int(episode_idx),'step':int(step_in_episode),'timestamp':timestamp,WHOLESALE_PRICE_SIGNAL:float(info[WHOLESALE_PRICE_SIGNAL]),IMPORT_PRICE_COLUMN:float(info[IMPORT_PRICE_COLUMN]),WHOLESALE_PRICE_PRED_COLUMN:float(wholesale_price_pred),IMPORT_PRICE_PRED_COLUMN:float(import_price_pred),'base_net_load_total':agent_raw_net_load_kw,'base_net_load_effective_total':agent_effective_net_load_kw,'net_load_total':agent_post_action_net_load_kw,'agent_raw_net_load_kw':agent_raw_net_load_kw,'agent_effective_net_load_kw':agent_effective_net_load_kw,'agent_post_action_net_load_kw':agent_post_action_net_load_kw,'fixed_load_kw':fixed_load_kw,'fixed_generation_kw':fixed_generation_kw,'feeder_raw_net_load_kw':feeder_raw_net_load_kw,'feeder_effective_net_load_kw':feeder_effective_net_load_kw,'feeder_post_action_net_load_kw':feeder_post_action_net_load_kw,'root_net_exchange_kw':pp_root_p_kw,'pp_root_p_available':bool(pp_root_p_array.size),'load_total':float(np.sum(load_values)),'pv_raw_total':float(np.sum(pv_raw)),'pv_effective_total':float(np.sum(pv_effective)),'pv_curtail_total':float(np.sum(pv_curtail)),'grid_import_total':float(np.sum(grid_import)),'grid_export_total':float(np.sum(grid_export)),'battery_charge_total':float(np.sum(battery_charge)),'battery_discharge_total':float(np.sum(battery_discharge)),'purchase_cost_total':purchase_cost_total,'export_subsidy_total':export_subsidy_total,'objective_total':objective_total,'storage_purchase_cost_eur':storage_purchase_cost_eur,'storage_sale_revenue_eur':storage_sale_revenue_eur,'storage_total_profit_eur':storage_total_profit_eur,'storage_objective_eur':storage_objective_eur,'storage_purchase_cost_eur_step':storage_purchase_cost_eur,'storage_sale_revenue_eur_step':storage_sale_revenue_eur,'storage_total_profit_eur_step':storage_total_profit_eur,'storage_charge_cost_total_eur':storage_purchase_cost_eur,'storage_discharge_revenue_total_eur':storage_sale_revenue_eur,'storage_profit_total_eur':storage_total_profit_eur,'storage_objective_total_eur':storage_objective_eur,'soc_penalty_total':soc_penalty_total,'voltage_penalty_total':voltage_penalty_total,'line_penalty_total':line_penalty_total,'trafo_penalty_total':trafo_penalty_total,'line_loading_pct_max':float(np.max(line_loading_pct))if line_loading_pct.size else .0,'trafo_loading_pct_max':float(np.max(trafo_loading_pct))if trafo_loading_pct.size else .0,'n_trafo_violations':int(info.get('n_trafo_violations',0)),'misocp_fallback':float(info.get('misocp_fallback',.0)),'simultaneous_step_flag':float(info.get('simultaneous_step_flag',.0))}
	if global_step is not None:step_row['global_step']=int(global_step)
	agent_rows,grid_rows=[],[];load_pred_values,pv_pred_values,soc_next=np.asarray(load_pred,dtype=np.float32),np.asarray(pv_pred,dtype=np.float32),np.asarray(info['soc_next'],dtype=np.float32)
	for(agent_idx,profile)in enumerate(agent_profiles):
		agent_row={'controller':controller_label,'episode_idx':int(episode_idx),'step':int(step_in_episode),'timestamp':timestamp,'agent_id':agent_idx,'agent_profile':str(profile),'load':float(load_values[agent_idx]),'load_pred':float(load_pred_values[agent_idx]),'pv':float(pv_values[agent_idx]),'pv_pred':float(pv_pred_values[agent_idx]),'e_bat':float(battery_power[agent_idx]),'e_bat_req':float(battery_power_req[agent_idx]),'soc':float(soc_next[agent_idx]),'purchase_cost':float(purchase_cost_per_agent[agent_idx]),'export_subsidy':float(export_subsidy_per_agent[agent_idx]),'objective_total':float(objective_per_agent[agent_idx]),'storage_purchase_cost_eur':float(storage_purchase_cost_per_agent[agent_idx]),'storage_sale_revenue_eur':float(storage_sale_revenue_per_agent[agent_idx]),'storage_total_profit_eur':float(storage_total_profit_per_agent[agent_idx]),'storage_objective_eur':float(storage_objective_per_agent[agent_idx]),'storage_charge_cost_eur':float(storage_purchase_cost_per_agent[agent_idx]),'storage_discharge_revenue_eur':float(storage_sale_revenue_per_agent[agent_idx]),'storage_profit_eur':float(storage_total_profit_per_agent[agent_idx])}
		if global_step is not None:agent_row['global_step']=int(global_step)
		agent_rows.append(agent_row)
	if vm_pu.shape[0]==len(bus_ids):
		for(bus_id,vm_value)in zip(bus_ids,vm_pu,strict=False):
			grid_row={'controller':controller_label,'episode_idx':int(episode_idx),'step':int(step_in_episode),'timestamp':timestamp,'bus_id':int(bus_id),'vm_pu':float(vm_value),'is_agent_bus':bool(int(bus_id)in agent_bus_set)}
			if global_step is not None:grid_row['global_step']=int(global_step)
			grid_rows.append(grid_row)
	return{'step_row':step_row,'agent_rows':agent_rows,'grid_rows':grid_rows,'pp_vm_pu':vm_pu.copy(),'pp_line_loading_pct':line_loading_pct.copy(),'pp_trafo_loading_pct':trafo_loading_pct.copy(),'pp_root_p_kw':pp_root_p_kw}
def _normalize_rollout_soc_mode(soc_mode:str)->str:
	resolved=str(soc_mode).strip().lower()
	if resolved not in{'reset','continuous'}:raise ValueError(f"rollout soc_mode must be one of {{'reset', 'continuous'}}, got {soc_mode!r}.")
	return resolved
def _restore_carried_soc(env,carried_soc:np.ndarray|None):
	if carried_soc is None:return None
	soc=np.asarray(carried_soc,dtype=np.float32).reshape(-1)
	if soc.shape!=(int(env.n),):raise ValueError(f"continuous SoC rollout expected carried SoC shape {(int(env.n),)}, got {soc.shape}.")
	if np.any(soc<float(env.soc_min)-1e-06)or np.any(soc>float(env.soc_max)+1e-06):raise ValueError(f"continuous SoC rollout received carried SoC outside [{float(env.soc_min):.4f}, {float(env.soc_max):.4f}]: {soc.tolist()}.")
	env.soc=soc.astype(np.float32,copy=True)
	return env.obs_builder.build(env)
def _attach_agent_soc_boundaries(agent_rows:list[dict[str,object]],info:dict[str,object])->None:
	if not agent_rows:return
	for(column,key)in(('soc_start','soc_t'),('soc_end','soc_next')):
		values=np.asarray(info.get(key,[]),dtype=np.float32).reshape(-1)
		if values.size!=len(agent_rows):continue
		for(agent_idx,row)in enumerate(agent_rows):row[column]=float(values[agent_idx])
def collect_controller_rollout(cfg,*,label:str,controller=None,controller_builder=None,action_fn=None,episode_indices:list[int]|None=None,soc_mode:str='reset')->RolloutResult:
	provided=int(controller is not None)+int(controller_builder is not None)+int(action_fn is not None)
	if provided!=1:raise ValueError('Provide exactly one of controller, controller_builder, or action_fn.')
	from scripts.builder import build_env
	if getattr(getattr(cfg,'runtime',None),'shared_data_dir',None)not in(None,''):cfg.runtime.forecast_ready=None
	else:cfg.runtime.forecast_ready=ensure_forecast_ready(cfg)
	env=build_env(cfg,mode='test');step_rows:list[dict[str,object]]=[];agent_rows:list[dict[str,object]]=[];grid_rows:list[dict[str,object]]=[];bus_ids=[int(bus_id)for bus_id in env._grid_core.net.bus.index.tolist()];agent_bus_ids=[int(bus_id)for bus_id in getattr(env._grid_core,'agent_bus_ids',cfg.grid.agent_bus_ids)];agent_bus_set=set(agent_bus_ids);loading_limit_pct=float(cfg.grid.line_max_loading_pct);fixed_load_kw,fixed_generation_kw=_fixed_feeder_components_kw(env);trafo_limit_kw=_approx_trafo_limit_kw(env,loading_limit_pct=loading_limit_pct)
	try:
		active_controller=controller_builder(env)if controller_builder is not None else controller;local_action_penalty_enabled,local_action_penalty_weight,local_action_penalty_mode=resolve_local_action_penalty_settings(cfg);resolved_soc_mode=_normalize_rollout_soc_mode(soc_mode);effective_episode_indices=[int(index)for index in episode_indices]if episode_indices is not None else[int(index)for index in list(getattr(cfg.runtime,'selected_episode_indices',[])or[])]or list(range(int(env.num_available_episodes)))
		if resolved_soc_mode=='continuous'and effective_episode_indices!=sorted(effective_episode_indices):raise ValueError(f"continuous SoC rollout requires chronological episode_indices sorted in ascending order, got {effective_episode_indices}. Re-run the MPC notebook with sorted selected_episode_indices.")
		carried_soc:np.ndarray|None=None;global_step=0
		for episode_idx in effective_episode_indices:
			obs,reset_info=env.reset(episode_idx=episode_idx);restored_obs=_restore_carried_soc(env,carried_soc)if resolved_soc_mode=='continuous'else None
			if restored_obs is not None:obs=restored_obs
			raw_obs=env.obs_builder.build_raw(env)
			if active_controller is not None:active_controller.reset()
			previous_raw_obs=None;done=False;step_in_episode=0
			while not done:
				wholesale_price_pred=_aligned_prediction(previous_raw_obs,raw_obs,WHOLESALE_PRICE_SEQ_FIELD);import_price_pred=float(derive_import_price(wholesale_price_pred,markup_eur_per_kwh=float(getattr(env,'import_price_markup_eur_per_kwh',get_import_price_markup(env)))));load_pred=_aligned_prediction(previous_raw_obs,raw_obs,'load_seq');pv_pred=_aligned_prediction(previous_raw_obs,raw_obs,'pv_seq');timestamp=_step_timestamp(reset_info,step_in_episode)
				if active_controller is not None:actions=active_controller.act(obs,deterministic=True);action_info=getattr(active_controller,'last_action_info',None)
				else:
					action_result=action_fn(env,raw_obs)
					if isinstance(action_result,tuple)and len(action_result)==2:actions,action_info=action_result
					else:actions=action_result;action_info=None
				next_obs,reward,terminated,truncated,info=env.step(actions);reward_array=np.asarray(reward,dtype=np.float32).reshape(-1);apply_action_penalty=bool(getattr(active_controller,'apply_action_penalty',False)and local_action_penalty_enabled);require_strict_local_action_feasibility(action_info,mode=local_action_penalty_mode);info,action_penalty=merge_action_info_into_step_info(info,action_info,soc_pen_weight=local_action_penalty_weight,apply_action_penalty=apply_action_penalty);reward_array=reward_array-np.asarray(action_penalty,dtype=np.float32);info['reward']=reward_array.astype(np.float32);del terminated,truncated;raw_next_obs=env.obs_builder.build_raw(env)if not bool(info.get('episode_done',False))else next_obs;rollout_records=_build_rollout_records(controller_label=label,episode_idx=episode_idx,step_in_episode=step_in_episode,timestamp=timestamp,info=info,load_pred=load_pred,pv_pred=pv_pred,wholesale_price_pred=float(wholesale_price_pred),import_price_pred=import_price_pred,agent_profiles=list(cfg.data.agent_profiles),bus_ids=bus_ids,agent_bus_set=agent_bus_set,dt_hours=float(env.dt),export_subsidy=float(getattr(cfg.reward,'export_subsidy_eur_per_kwh',.079)),fixed_load_kw=fixed_load_kw,fixed_generation_kw=fixed_generation_kw,global_step=global_step);_attach_agent_soc_boundaries(rollout_records['agent_rows'],info);step_rows.append(rollout_records['step_row']);agent_rows.extend(rollout_records['agent_rows']);grid_rows.extend(rollout_records['grid_rows']);done=bool(info.get('episode_done',False));previous_raw_obs=raw_obs;obs=next_obs;raw_obs=raw_next_obs;step_in_episode+=1;global_step+=1
			carried_soc=np.asarray(env.soc,dtype=np.float32).copy()
		step_df=pd.DataFrame(step_rows).sort_values(['episode_idx','step']).reset_index(drop=True);agent_df=pd.DataFrame(agent_rows).sort_values(['episode_idx','step','agent_id']).reset_index(drop=True);grid_df=pd.DataFrame(grid_rows).sort_values(['episode_idx','step','bus_id']).reset_index(drop=True);summary_columns=['purchase_cost','export_subsidy','objective_total','storage_purchase_cost_eur','storage_sale_revenue_eur','storage_total_profit_eur','storage_objective_eur','storage_charge_cost_eur','storage_discharge_revenue_eur','storage_profit_eur'];summary=agent_df.groupby(['controller','agent_profile'],as_index=False)[[column for column in summary_columns if column in agent_df.columns]].sum()if not agent_df.empty else pd.DataFrame(columns=['controller','agent_profile',*summary_columns]);initial_soc=np.full((int(env.n),),float(env.init_soc),dtype=np.float32);final_soc=np.asarray(carried_soc if carried_soc is not None else initial_soc,dtype=np.float32);return RolloutResult(step_df=step_df,agent_df=agent_df,grid_df=grid_df,summary=summary,meta={'controller':label,'agent_profiles':list(cfg.data.agent_profiles),'agent_bus_ids':agent_bus_ids,'v_min_pu':float(cfg.grid.v_min_pu),'v_max_pu':float(cfg.grid.v_max_pu),'prediction_mode':resolve_prediction_mode_from_forecast_backend(cfg.forecast.type),'trafo_limit_kw':trafo_limit_kw,'loading_limit_pct':loading_limit_pct,'trafo_loading_limit_pct':loading_limit_pct,IMPORT_PRICE_MARKUP_KEY:float(getattr(getattr(cfg,'reward',None),IMPORT_PRICE_MARKUP_KEY,.0)),'economics_scope':'storage_only','objective_mode':str(getattr(getattr(cfg,'reward',None),'storage_objective_mode','max_storage_profit')),'reward_objective_mode':str(getattr(getattr(cfg,'reward',None),'storage_objective_mode','max_storage_profit')),'reward_price_mode':str(getattr(getattr(cfg,'reward',None),'storage_price_mode','real_time_price')),'storage_profit_weight':float(getattr(getattr(cfg,'reward',None),'storage_profit_weight',1.0)),'soc_mode':resolved_soc_mode,'initial_soc_by_agent':initial_soc.tolist(),'final_soc_by_agent':final_soc.tolist(),'selected_episode_indices':effective_episode_indices})
	finally:env.close()
def bootstrap_madrl_notebook_shared_data(cfg,*,root:str|Path|None=None,notebook_path:str|Path|None=None)->dict[str,str]:
	root_path=_project_root_from_notebook_helpers()if root is None else Path(root).resolve();notebook_hint=str(Path(notebook_path).as_posix())if notebook_path is not None else'notebooks/madrl/<training_notebook>.ipynb';shared_data_record_path=(root_path/'notebooks'/'record'/'forecast'/'lstm'/'shared_data_record.json').resolve()
	if not shared_data_record_path.exists():raise FileNotFoundError(f"Missing forecast shared-data object: '{shared_data_record_path}'. Expected canonical forecast shared-data contract with 'shared_data_dir' and 'signature_hash'. Re-run notebooks/forecast/forecast_lstm.ipynb before {notebook_hint}.")
	shared_data_record=json.loads(shared_data_record_path.read_text(encoding='utf-8'));shared_data_dir_raw=str(shared_data_record.get('shared_data_dir','')).strip()
	if not shared_data_dir_raw:raise ValueError(f"Forecast shared-data record is missing 'shared_data_dir': '{shared_data_record_path}'. Expected canonical forecast shared-data contract with a non-empty 'shared_data_dir'. Re-run notebooks/forecast/forecast_lstm.ipynb before {notebook_hint}.")
	shared_data_dir=Path(shared_data_dir_raw).resolve()
	if not shared_data_dir.exists():raise FileNotFoundError(f"Missing forecast shared-data package: '{shared_data_dir}'. Expected canonical forecast shared-data directory referenced by '{shared_data_record_path}'. Re-run notebooks/forecast/forecast_lstm.ipynb before {notebook_hint}.")
	shared_data_signature=str(shared_data_record.get('signature_hash','')).strip()
	if not shared_data_signature:raise ValueError(f"Forecast shared-data record is missing 'signature_hash': '{shared_data_record_path}'. Expected canonical forecast shared-data contract with a non-empty 'signature_hash'. Re-run notebooks/forecast/forecast_lstm.ipynb before {notebook_hint}.")
	cfg.runtime.shared_data_dir=str(shared_data_dir);cfg.runtime.shared_data_signature=shared_data_signature;return{'shared_data_record_path':str(shared_data_record_path),'shared_data_dir':str(shared_data_dir),'shared_data_signature':shared_data_signature}
def resolve_madrl_notebook_training(cfg,*,spec:Mapping[str,object],force_retrain_madrl:bool,root:str|Path|None=None,notebook_path:str|Path|None=None)->dict[str,object]:
	from scripts.mainline_madrl import load_madrl_training_result,run_external_train_mainline
	root_path=_project_root_from_notebook_helpers()if root is None else Path(root).resolve();spec_dict={str(key):value for(key,value)in dict(spec).items()};notebook_hint=str(Path(notebook_path).as_posix())if notebook_path is not None else f"notebooks/madrl/{spec_dict.get('experiment_name','madrl')}.ipynb";expected_saved_fields={'agent_profiles':list(cfg.data.agent_profiles),'agent_bus_ids':list(cfg.grid.agent_bus_ids),'load_scale':list(cfg.data.load_scale),'pv_scale':list(cfg.data.pv_scale),'future_horizon':int(cfg.env.future_horizon),'test_start_date':str(cfg.data.test_start_date),'test_end_date':str(cfg.data.test_end_date)};expected_shared_data_dir_raw=str(getattr(cfg.runtime,'shared_data_dir',None)or'').strip();expected_shared_data_signature=str(getattr(cfg.runtime,'shared_data_signature',None)or'').strip()
	if not expected_shared_data_dir_raw or not expected_shared_data_signature:raise ValueError(f"Missing forecast shared-data runtime contract for {notebook_hint}. Expected canonical forecast shared-data contract loaded from 'notebooks/record/forecast/lstm/shared_data_record.json' with both 'shared_data_dir' and 'signature_hash'. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun {notebook_hint}.")
	expected_shared_data_dir=str(Path(expected_shared_data_dir_raw).resolve())
	if not Path(expected_shared_data_dir).exists():raise FileNotFoundError(f"Missing forecast shared-data package: '{expected_shared_data_dir}'. Expected canonical forecast shared-data directory for {notebook_hint}. Re-run notebooks/forecast/forecast_lstm.ipynb before rerunning {notebook_hint}.")
	runtime_controls={'shared_data_dir':expected_shared_data_dir,'shared_data_signature':expected_shared_data_signature}
	reward_controls={'w_soc_pen':float(cfg.reward.w_soc_pen),'w_voltage_pen':float(cfg.reward.w_voltage_pen),'w_line_pen':float(cfg.reward.w_line_pen),'w_trafo_pen':float(cfg.reward.w_trafo_pen),'export_subsidy_eur_per_kwh':float(cfg.reward.export_subsidy_eur_per_kwh),'import_price_markup_eur_per_kwh':float(cfg.reward.import_price_markup_eur_per_kwh),'storage_objective_mode':str(cfg.reward.storage_objective_mode),'storage_price_mode':str(cfg.reward.storage_price_mode),'storage_profit_weight':float(cfg.reward.storage_profit_weight),'local_action_penalty_mode':str(cfg.reward.local_action_penalty_mode),'local_action_penalty_weight':float(cfg.reward.local_action_penalty_weight)}
	experiment_controls={'algorithm':str(cfg.algo.name),'seed':int(cfg.runtime.seed),'runtime_mode':str(cfg.runtime.execution_mode),'device_request':None if str(cfg.runtime.device)=='cpu'else str(cfg.runtime.device),'forecast_controls':build_mainline_forecast_controls(cfg),'reward_controls':reward_controls,'safety_controls':{'enabled':bool(cfg.safety.enabled),'projector_mode':str(cfg.safety.projector_mode),'projection_iters':int(cfg.safety.projection_iters),'voltage_margin_pu':float(cfg.safety.voltage_margin_pu),'line_margin_pct':float(cfg.safety.line_margin_pct),'trafo_margin_pct':float(cfg.safety.trafo_margin_pct),'linearization_delta_kw':float(cfg.safety.linearization_delta_kw),'record_diagnostics':bool(cfg.safety.record_diagnostics)}if cfg.safety.enabled else{}}
	experiment_controls['runtime_controls']=runtime_controls
	data_controls={'prediction_mode':'normal','agent_profiles':list(cfg.data.agent_profiles),'agent_bus_ids':list(cfg.grid.agent_bus_ids),'load_scale':list(cfg.data.load_scale),'pv_scale':list(cfg.data.pv_scale),'future_horizon':int(cfg.env.future_horizon),'train_year':int(cfg.data.train_year),'test_year':int(cfg.data.test_year),'test_start_date':str(cfg.data.test_start_date),'test_end_date':str(cfg.data.test_end_date)}
	battery_controls={'battery_capacity':list(cfg.env.battery_capacity),'max_charge_rate':float(cfg.env.max_charge_rate),'efficiency':float(cfg.env.efficiency),'init_soc':float(cfg.env.init_soc),'soc_min':float(cfg.env.soc_min),'soc_max':float(cfg.env.soc_max),'soc_target':float(cfg.env.soc_target)}
	train_controls={'profile':'base','model_family':str(cfg.model.family),'train_episodes':int(cfg.train.train_episodes),'max_train_steps':cfg.train.max_train_steps,'num_envs':int(cfg.train.num_envs),'vec_env_type':str(cfg.train.vec_env_type),'parallel_episode_sampling':str(cfg.train.parallel_episode_sampling),'batch_size':int(cfg.train.batch_size),'buffer_size':int(cfg.train.buffer_size),'update_interval':int(cfg.train.update_interval),'updates_per_step':int(cfg.train.updates_per_step),'policy_update_freq':int(cfg.algo.policy_update_freq),'actor_lr':float(cfg.train.actor_lr),'critic_lr':float(cfg.train.critic_lr),'noise_std_init':float(cfg.train.noise_std_init),'noise_std_min':float(cfg.train.noise_std_min),'noise_decay_steps':float(cfg.train.noise_decay_steps),'use_noise_decay':bool(cfg.train.use_noise_decay),'show_progress':bool(cfg.train.show_progress),'progress_episode_interval':int(cfg.train.progress_episode_interval)}
	checkpoint_controls={'experiment_name':str(spec_dict['experiment_name'])}
	if force_retrain_madrl:
		launch=run_external_train_mainline(project_root=root_path,experiment_controls=experiment_controls,data_controls=data_controls,train_controls=train_controls,battery_controls=battery_controls,checkpoint_controls=checkpoint_controls,data_dir=root_path/'data',env_name=str(spec_dict['env_name']));train_result=dict(launch['result']);result_json_path=str(launch['launch_info']['result_json_path']);model_root=str(train_result['model_root'])
	else:
		loaded=load_madrl_training_result(algorithm=str(spec_dict['algorithm']),prediction_mode='normal',experiment_name=str(spec_dict['experiment_name']),root=root_path);train_result=dict(loaded['result']);result_json_path=str(loaded['result_json_path']);model_root=str(loaded['model_root']);stored_data=dict(train_result.get('data_controls',{}));stored_reward_controls=dict(dict(train_result.get('experiment_controls',{})).get('reward_controls',{}));mismatches=[f"{field_name}: expected={expected_value!r}, got={stored_data.get(field_name)!r}"for(field_name,expected_value)in expected_saved_fields.items()if stored_data.get(field_name)!=expected_value]
		for field_name,expected_value in reward_controls.items():
			if stored_reward_controls.get(field_name)!=expected_value:mismatches.append(f"reward_controls.{field_name}: expected={expected_value!r}, got={stored_reward_controls.get(field_name)!r}")
		if str(train_result.get('prediction_mode'))!='normal':mismatches.append(f"prediction_mode: expected='normal', got={train_result.get('prediction_mode')!r}")
		if mismatches:raise ValueError(f"Saved MADRL run does not match the canonical notebook contract. Re-run {notebook_hint} with force_retrain_madrl=True.\n"+'\n'.join(mismatches))
		actual_shared_data_signature=str(train_result.get('shared_data_signature')or'').strip()
		if expected_shared_data_signature and actual_shared_data_signature!=expected_shared_data_signature:raise ValueError(f"Saved MADRL run does not match the canonical forecast shared-data package. Expected signature {expected_shared_data_signature!r}, got {actual_shared_data_signature!r}. Re-run {notebook_hint} with force_retrain_madrl=True.")
	return{'train_result':train_result,'model_root':model_root,'result_json_path':result_json_path}


NOTEBOOK_RECORD_SCHEMA_VERSION=1


def _project_root_from_notebook_helpers()->Path:return Path(__file__).resolve().parents[2]


def get_notebook_record_root(root:str|Path|None=None)->Path:
	base_dir=_project_root_from_notebook_helpers()if root is None else Path(root).resolve()
	return(base_dir/'notebooks'/'record').resolve()


def get_notebook_record_dir(*,category:str,scheme_name:str,root:str|Path|None=None)->Path:
	return(get_notebook_record_root(root)/str(category).strip()/str(scheme_name).strip()).resolve()


def _json_default(value:object):
	if isinstance(value,Path):return str(value)
	if isinstance(value,pd.Timestamp):return value.isoformat()
	if isinstance(value,pd.Series):return value.to_dict()
	if isinstance(value,np.ndarray):return value.tolist()
	if isinstance(value,(np.integer,)):return int(value)
	if isinstance(value,(np.floating,)):return float(value)
	if isinstance(value,(np.bool_,)):return bool(value)
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path:str|Path,payload:Mapping[str,object])->Path:
	target_path=Path(path).resolve();target_path.parent.mkdir(parents=True,exist_ok=True);target_path.write_text(json.dumps(dict(payload),indent=2,default=_json_default),encoding='utf-8');return target_path


def save_rollout_record(rollout:RolloutResult,*,category:str,scheme_name:str,root:str|Path|None=None,config_snapshot:Mapping[str,object]|None=None,extra_meta:Mapping[str,object]|None=None)->dict[str,object]:
	from scripts.mainline_compare import compare_rollout_metrics
	record_dir=get_notebook_record_dir(category=category,scheme_name=scheme_name,root=root);record_dir.mkdir(parents=True,exist_ok=True);step_path,agent_path,grid_path,summary_path,metrics_path,meta_path,manifest_path=(record_dir/name for name in('step.parquet','agent.parquet','grid.parquet','summary.parquet','metrics.parquet','meta.json','manifest.json'));markup_eur_per_kwh=require_import_price_markup(rollout.meta,context=f"Rollout '{scheme_name}' metadata");normalized_step_df=canonicalize_step_price_frame(rollout.step_df,markup_eur_per_kwh=markup_eur_per_kwh,context=f"Rollout '{scheme_name}' step_df",require_actual=not rollout.step_df.empty,require_prediction=False);normalized_rollout=RolloutResult(step_df=normalized_step_df,agent_df=rollout.agent_df.copy(),grid_df=rollout.grid_df.copy(),summary=rollout.summary.copy(),meta=dict(rollout.meta));normalized_rollout.step_df.to_parquet(step_path,index=False);normalized_rollout.agent_df.to_parquet(agent_path,index=False);normalized_rollout.grid_df.to_parquet(grid_path,index=False);normalized_rollout.summary.to_parquet(summary_path,index=False);metrics_df=compare_rollout_metrics(normalized_rollout);metrics_df.to_parquet(metrics_path,index=False);meta_payload=dict(normalized_rollout.meta);meta_tables={key:value for(key,value)in list(meta_payload.items())if isinstance(value,pd.DataFrame)};meta_series={key:value for(key,value)in list(meta_payload.items())if isinstance(value,pd.Series)}
	for key in list(meta_tables)+list(meta_series):meta_payload.pop(key,None)
	if extra_meta:meta_payload.update(dict(extra_meta))
	meta_table_files={}
	for(key,value)in meta_tables.items():table_name=f"meta_{key}.parquet";value.to_parquet(record_dir/table_name,index=False);meta_table_files[str(key)]=table_name
	meta_series_files={}
	for(key,value)in meta_series.items():series_name=f"meta_{key}.json";_write_json(record_dir/series_name,value.to_dict());meta_series_files[str(key)]=series_name
	meta_payload.setdefault('economics_scope','storage_only');meta_payload.setdefault('objective_mode','max_storage_profit');meta_payload.setdefault('storage_price_mode','real_time_price')
	_write_json(meta_path,meta_payload)
	config_snapshot_path=None
	if config_snapshot is not None:config_snapshot_path=str(_write_json(record_dir/'config_snapshot.json',dict(config_snapshot)).name)
	manifest={'schema_version':NOTEBOOK_RECORD_SCHEMA_VERSION,'category':str(category),'scheme_name':str(scheme_name),'record_dir':str(record_dir),'files':{'step_df':step_path.name,'agent_df':agent_path.name,'grid_df':grid_path.name,'summary_df':summary_path.name,'metrics_df':metrics_path.name,'meta':meta_path.name,'config_snapshot':config_snapshot_path},'meta_tables':meta_table_files,'meta_series':meta_series_files}
	_write_json(manifest_path,manifest);return manifest


def load_rollout_record(*,category:str,scheme_name:str,root:str|Path|None=None)->RolloutResult:
	record_dir=get_notebook_record_dir(category=category,scheme_name=scheme_name,root=root);manifest=json.loads((record_dir/'manifest.json').read_text(encoding='utf-8'));step_df,agent_df,grid_df,summary_df=(pd.read_parquet(record_dir/manifest['files'][key])for key in('step_df','agent_df','grid_df','summary_df'));meta=dict(json.loads((record_dir/manifest['files']['meta']).read_text(encoding='utf-8')))
	for(key,filename)in dict(manifest.get('meta_tables',{})).items():meta[str(key)]=pd.read_parquet(record_dir/filename)
	for(key,filename)in dict(manifest.get('meta_series',{})).items():meta[str(key)]=pd.Series(json.loads((record_dir/filename).read_text(encoding='utf-8')))
	markup_eur_per_kwh=require_import_price_markup(meta,context=f"Saved rollout '{scheme_name}' metadata");step_df=canonicalize_step_price_frame(step_df,markup_eur_per_kwh=markup_eur_per_kwh,context=f"Saved rollout '{scheme_name}' step_df",require_actual=not step_df.empty,require_prediction=False);return RolloutResult(step_df=step_df,agent_df=agent_df,grid_df=grid_df,summary=summary_df,meta=meta)
