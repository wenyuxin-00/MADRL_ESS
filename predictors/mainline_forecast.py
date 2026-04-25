from __future__ import annotations
from pathlib import Path
from typing import Mapping
from configs.experiment_config import ExperimentConfig
from predictors.lstm_forecaster import get_default_lstm_artifact_dir
from predictors.training import ensure_lstm_artifacts
_FORECAST_FIELDS='target_signals','history_window','load_model_mode','load_component_split','load_scaler_type','load_time_feature_mode','pv_time_feature_mode','load_hybrid_mode','load_baseline_mode','pv_postprocess_mode','auto_train_missing'
def _normalize_signal_training_overrides(overrides:Mapping[str,Mapping[str,object]]|None)->dict[str,dict[str,object]]:
	normalized={}
	for(signal_name,signal_overrides)in dict(overrides or{}).items():
		if not isinstance(signal_overrides,Mapping):raise ValueError(f"signal_training_overrides['{signal_name}'] must be a mapping, got {signal_overrides!r}.")
		normalized[str(signal_name).strip().lower()]={str(field_name):value for(field_name,value)in dict(signal_overrides).items()}
	return normalized
def build_mainline_forecast_controls(cfg)->dict[str,object]:artifact_root=getattr(cfg.forecast,'lstm_artifact_root',None);controls={name:getattr(cfg.forecast,name)for name in _FORECAST_FIELDS};controls.update({'artifact_root':None if artifact_root in(None,'')else str(Path(artifact_root).resolve()),'target_signals':[str(value)for value in cfg.forecast.target_signals],'future_horizon':int(cfg.env.future_horizon),'signal_training_overrides':_normalize_signal_training_overrides(getattr(cfg.forecast,'signal_training_overrides',{}))});return controls
def resolve_mainline_forecast_controls(cfg,forecast_controls:Mapping[str,object]|None=None)->dict[str,object]:
	controls,canonical=dict(forecast_controls or{}),build_mainline_forecast_controls(cfg)
	if controls.get('future_horizon')not in(None,int(cfg.env.future_horizon)):raise ValueError(f"forecast_controls.future_horizon must match cfg.env.future_horizon, got {controls.get('future_horizon')} vs {cfg.env.future_horizon}.")
	incoming=_normalize_signal_training_overrides(controls.pop('signal_training_overrides',{}));merged={**canonical,**controls,'signal_training_overrides':{name:{**dict(canonical['signal_training_overrides'].get(name,{})or{}),**dict(incoming.get(name,{})or{})}for name in set(canonical['signal_training_overrides'])|set(incoming)}};artifact_root=merged.get('artifact_root',cfg.forecast.lstm_artifact_root);return{'target_signals':[str(value)for value in merged.get('target_signals',cfg.forecast.target_signals)],**{name:type(getattr(cfg.forecast,name))(merged.get(name,getattr(cfg.forecast,name)))for name in _FORECAST_FIELDS},'artifact_root':None if artifact_root in(None,'')else str(Path(artifact_root).resolve()),'signal_training_overrides':_normalize_signal_training_overrides(merged.get('signal_training_overrides',getattr(cfg.forecast,'signal_training_overrides',{})))}
def get_mainline_forecast_controls(*,artifact_root:str|Path|None=None,auto_train_missing:bool=False)->dict[str,object]:return{**build_mainline_forecast_controls(ExperimentConfig()),'artifact_root':str(Path(artifact_root).resolve())if artifact_root is not None else str(get_default_lstm_artifact_dir()),'auto_train_missing':bool(auto_train_missing)}
def ensure_mainline_forecast_ready(cfg)->dict[str,object]|None:return None if cfg.forecast.type!='lstm'else ensure_lstm_artifacts(cfg,device=cfg.runtime.device)
