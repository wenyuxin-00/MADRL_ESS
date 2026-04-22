from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
from predictors.lstm_forecaster import LSTMForecaster,get_default_lstm_artifact_dir
from predictors.training import collect_available_lstm_artifacts,ensure_lstm_artifacts,required_forecast_signals
from scripts.utils.price_protocol import WHOLESALE_PRICE_SIGNAL
class PerfectForecaster:
	def __init__(self,episode_signals:dict[str,np.ndarray]|np.ndarray|None=None):self._episode_signals={};episode_signals is not None and self.set_episode(episode_signals)
	def reset(self)->None:return None
	def set_episode(self,episode_signals:dict[str,np.ndarray]|np.ndarray,episode_meta:dict[str,object]|None=None)->None:del episode_meta;self._episode_signals={name:np.asarray(values,dtype=np.float32)for(name,values)in episode_signals.items()}if isinstance(episode_signals,dict)else{WHOLESALE_PRICE_SIGNAL:np.asarray(episode_signals,dtype=np.float32)}
	def predict(self,history:np.ndarray,horizon:int,*,signal_name:str=WHOLESALE_PRICE_SIGNAL,history_timestamps:list[str|pd.Timestamp]|None=None)->np.ndarray:
		del history_timestamps
		if horizon<=0:return np.zeros((0,),dtype=np.float32)
		if signal_name not in self._episode_signals:raise RuntimeError(f"PerfectForecaster has no episode signal '{signal_name}'. Available signals: {sorted(self._episode_signals)}")
		full_signal=np.asarray(self._episode_signals[signal_name],dtype=np.float32);time_index=max(0,np.asarray(history,dtype=np.float32).shape[0]-1)
		if full_signal.ndim==1:chunk=full_signal[time_index:time_index+horizon];return np.concatenate([chunk,np.zeros((max(horizon-chunk.size,0),),dtype=np.float32)],axis=0)[:horizon].astype(np.float32)
		if full_signal.ndim!=2:raise ValueError(f"PerfectForecaster expects 1D or 2D episode signals, got {full_signal.shape}")
		chunk=full_signal[time_index:time_index+horizon,:];return np.concatenate([chunk,np.zeros((max(horizon-chunk.shape[0],0),chunk.shape[1]),dtype=np.float32)],axis=0)[:horizon,:].T.astype(np.float32)
def build_forecaster(cfg):
	forecast_cfg=cfg.forecast;forecaster_type=str(forecast_cfg.type).strip().lower()
	if forecaster_type=='perfect':return PerfectForecaster()
	if forecaster_type!='lstm':raise ValueError(f"Unknown forecaster_type '{forecaster_type}', available: ['perfect', 'lstm']")
	ensure_result=ensure_lstm_artifacts(cfg,device=cfg.runtime.device);artifact_map=dict(ensure_result.get('artifacts')or collect_available_lstm_artifacts(cfg))
	if not artifact_map:artifact_root=Path(forecast_cfg.lstm_artifact_root or get_default_lstm_artifact_dir());raise FileNotFoundError(f"No managed LSTM forecast artifacts are available. Checked root: {artifact_root} for future_horizon={cfg.env.future_horizon}.")
	active_signals=required_forecast_signals(cfg);missing_required=[signal_name for signal_name in active_signals if signal_name not in artifact_map]
	if missing_required:raise FileNotFoundError(f"Missing required LSTM forecast artifacts for observation signals: {missing_required}. Available managed signals: {sorted(artifact_map)}.")
	if ensure_result.get('trained_signals'):print(f"[forecast] trained new artifacts for signals: {', '.join(ensure_result['trained_signals'])}")
	if ensure_result.get('retrained_signals'):print(f"[forecast] refreshed incompatible artifacts for signals: {', '.join(ensure_result['retrained_signals'])}")
	return LSTMForecaster.from_signal_artifacts({signal_name:artifact_map[signal_name]for signal_name in active_signals},device=cfg.runtime.device)
