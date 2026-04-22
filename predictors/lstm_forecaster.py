from __future__ import annotations
import json,pickle
from pathlib import Path
from typing import Sequence
import numpy as np,pandas as pd,torch,torch.nn as nn
from predictors.time_features import TIME_FEATURE_MODE_NONE,coerce_timestamp_index,encode_forecast_time_features,infer_timestamp_step,normalize_time_feature_mode
from scripts.utils.project_paths import get_forecast_artifact_root
from scripts.utils.price_protocol import PRICE_PROTOCOL_VERSION,WHOLESALE_PRICE_SIGNAL,normalize_internal_signal_name
LSTM_ARTIFACT_FORMAT,LSTM_LOAD_HYBRID_ARTIFACT_FORMAT='lstm_forecaster_v4','lstm_forecaster_v5'
PHYSICAL_NORMALIZATION_NONE,PHYSICAL_NORMALIZATION_LOAD_SCALE,PHYSICAL_NORMALIZATION_PV_PEAK='none','load_scale','pv_peak_kw'
POSTPROCESS_MODE_NONE,POSTPROCESS_MODE_BASELINE_BLEND,POSTPROCESS_MODE_PHYSICAL_CLIP='none','baseline_blend','physical_clip'
BASELINE_MODE_NONE,BASELINE_MODE_LAST_VALUE='none','last_value'
PHYSICAL_SCALE_EPS=np.float32(1e-06)
LSTM_META_SUFFIX,LSTM_SCALER_SUFFIX='_meta.json','_scaler.pkl'
LSTM_REQUIRED_META_FIELDS='seq_len','pred_len','hidden_size','num_layers','dropout','input_size','time_feature_mode','model_mode','price_protocol_version'
DEFAULT_TIMESTAMP_START=pd.Timestamp('2000-01-01 00:00:00+00:00')
DEFAULT_LSTM_ARTIFACT_DIR=Path('lstm')
DEFAULT_SUPPORTED_FORECAST_SIGNALS='wholesale_price','load','pv'
def get_default_lstm_artifact_dir(root:str|Path|None=None)->Path:return Path(root)if root is not None else get_forecast_artifact_root(root)/DEFAULT_LSTM_ARTIFACT_DIR
def get_default_lstm_artifact_paths(root:str|Path|None=None,*,signal_name:str='wholesale_price',future_horizon:int=24,agent_index:int|None=None,component:str|None=None)->dict[str,Path]:
	signal_name=str(signal_name);artifact_dir=get_default_lstm_artifact_dir(root)/f"h{int(future_horizon)}"/signal_name;stem=f"{signal_name}_lstm_h{int(future_horizon)}"
	if agent_index is not None:artifact_dir,stem=artifact_dir/f"agent_{int(agent_index)}",f"{signal_name}_agent{int(agent_index)}_lstm_h{int(future_horizon)}"
	if component is not None:artifact_dir,stem=artifact_dir/str(component),f"{stem}_{str(component)}"
	return{'artifact_dir':artifact_dir,'model_path':artifact_dir/f"{stem}.pt",'meta_path':artifact_dir/f"{stem}_meta.json",'scaler_path':artifact_dir/f"{stem}_scaler.pkl"}
def _normalize_physical_mode(mode:str|None)->str:
	normalized=str(mode or PHYSICAL_NORMALIZATION_NONE).strip().lower()
	if normalized not in{PHYSICAL_NORMALIZATION_NONE,PHYSICAL_NORMALIZATION_LOAD_SCALE,PHYSICAL_NORMALIZATION_PV_PEAK}:raise ValueError(f"Unsupported physical_normalization_mode '{mode}'.")
	return normalized
def _coerce_physical_scale_array(scale_by_column,*,expected_size:int|None=None)->np.ndarray|None:
	if scale_by_column is None:return
	scale=np.asarray(scale_by_column,dtype=np.float32).reshape(-1)
	if scale.size==0:return
	if expected_size is not None:
		if scale.size==1 and expected_size>1:scale=np.repeat(scale,expected_size)
		elif scale.size!=expected_size:raise ValueError(f"physical_scale_by_column size mismatch: expected {expected_size}, got {scale.size}")
	return scale.astype(np.float32,copy=True)
def resolve_lstm_artifact_paths(model_path,meta_path=None,scaler_path=None)->tuple[Path,Path,Path]:model_path=Path(model_path);stem=model_path.stem;meta_path=Path(meta_path)if meta_path is not None else model_path.with_name(f"{stem}{LSTM_META_SUFFIX}");scaler_path=Path(scaler_path)if scaler_path is not None else model_path.with_name(f"{stem}{LSTM_SCALER_SUFFIX}");return model_path,meta_path,scaler_path
def save_lstm_forecaster_artifacts(model_path,state_dict,scaler,*,seq_len:int,pred_len:int,hidden_size:int,num_layers:int,dropout:float,signal_name:str=WHOLESALE_PRICE_SIGNAL,future_horizon:int|None=None,normalization_mode:str=PHYSICAL_NORMALIZATION_NONE,source_signature:dict[str,object]|None=None,input_size:int=1,time_feature_mode:str=TIME_FEATURE_MODE_NONE,model_mode:str='shared',agent_index:int|None=None,agent_profile:str|None=None,artifact_format:str|None=None,postprocess_mode:str=POSTPROCESS_MODE_NONE,baseline_mode:str=BASELINE_MODE_NONE,blend_weight:float|None=None,optimized_metric:str|None=None,component:str|None=None,scaler_type:str='standard')->dict[str,str]:
	if scaler is None:raise ValueError('LSTM forecaster artifacts require a fitted scaler object.')
	model_path,meta_path,scaler_path=resolve_lstm_artifact_paths(model_path);model_path.parent.mkdir(parents=True,exist_ok=True);torch.save(state_dict,model_path);meta={'artifact_format':str(artifact_format or LSTM_ARTIFACT_FORMAT),'signal_name':normalize_internal_signal_name(signal_name),'price_protocol_version':int(PRICE_PROTOCOL_VERSION),'future_horizon':int(pred_len if future_horizon is None else future_horizon),'seq_len':int(seq_len),'pred_len':int(pred_len),'hidden_size':int(hidden_size),'num_layers':int(num_layers),'dropout':float(dropout),'input_size':int(input_size),'time_feature_mode':normalize_time_feature_mode(time_feature_mode),'model_mode':str(model_mode),'normalization_mode':_normalize_physical_mode(normalization_mode),'source_signature':dict(source_signature or{}),'agent_index':None if agent_index is None else int(agent_index),'agent_profile':None if agent_profile is None else str(agent_profile),'component':None if component is None else str(component),'scaler_type':str(scaler_type)}
	if meta['artifact_format']==LSTM_LOAD_HYBRID_ARTIFACT_FORMAT or str(postprocess_mode)!=POSTPROCESS_MODE_NONE:meta.update({'postprocess_mode':str(postprocess_mode),'baseline_mode':str(baseline_mode),'blend_weight':None if blend_weight is None else float(blend_weight),'optimized_metric':None if optimized_metric is None else str(optimized_metric)})
	meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
	with scaler_path.open('wb')as handle:pickle.dump(scaler,handle)
	return{'model_path':str(model_path),'meta_path':str(meta_path),'scaler_path':str(scaler_path)}
class LSTMForecastModel(nn.Module):
	def __init__(self,hidden_size:int=128,num_layers:int=2,dropout:float=.23,pred_len:int=4,input_size:int=1):super().__init__();self.lstm=nn.LSTM(input_size=int(input_size),hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout if num_layers>1 else .0);self.head=nn.Sequential(nn.Linear(hidden_size,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,pred_len))
	def forward(self,x:torch.Tensor)->torch.Tensor:
		if x.dim()==2:x=x.unsqueeze(-1)
		elif x.dim()!=3:raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D.")
		return self.head(self.lstm(x)[0][:,-1,:])
def load_lstm_forecaster_artifacts(model_path,meta_path=None,scaler_path=None)->tuple[dict,object]:
	_,meta_path,scaler_path=resolve_lstm_artifact_paths(model_path,meta_path,scaler_path)
	if not meta_path.exists():raise FileNotFoundError(f"Missing LSTM meta artifact: '{meta_path}'. Expected '<stem>_meta.json' next to the model.")
	if not scaler_path.exists():raise FileNotFoundError(f"Missing LSTM scaler artifact: '{scaler_path}'. Expected '<stem>_scaler.pkl' next to the model.")
	meta=json.loads(meta_path.read_text(encoding='utf-8'));missing_fields=[field for field in LSTM_REQUIRED_META_FIELDS if field not in meta]
	if missing_fields:raise ValueError(f"LSTM meta artifact '{meta_path}' is missing required fields: {missing_fields}")
	if str(meta.get('signal_name','')).strip().lower()=='price':raise ValueError(f"LSTM meta artifact '{meta_path}' still uses legacy signal_name='price'. Rebuild the artifact with the wholesale/import price protocol cutover.")
	if int(meta.get('price_protocol_version',-1))!=int(PRICE_PROTOCOL_VERSION):raise ValueError(f"LSTM meta artifact '{meta_path}' has unsupported price_protocol_version={meta.get('price_protocol_version')!r}; expected {PRICE_PROTOCOL_VERSION}.")
	with scaler_path.open('rb')as handle:scaler=pickle.load(handle)
	return meta,scaler
class _SignalForecasterRuntime:
	def __init__(self,**kwargs):defaults={'agent_index':None,'agent_profile':None,'component':None,'physical_scale_by_column':None,'optimized_metric':None,'postprocess_mode':POSTPROCESS_MODE_NONE,'baseline_mode':BASELINE_MODE_NONE,'blend_weight':None};defaults.update(kwargs);self.__dict__.update(defaults)
class LSTMForecaster:
	def __init__(self,model_path:str|None=None,hidden_size:int=128,num_layers:int=2,dropout:float=.23,pred_len:int=4,seq_len:int=1344,device:str|torch.device='cpu',scaler=None,input_size:int=1,time_feature_mode:str=TIME_FEATURE_MODE_NONE,model_mode:str='shared',physical_normalization_mode:str=PHYSICAL_NORMALIZATION_NONE,physical_scale_by_column=None,postprocess_mode:str=POSTPROCESS_MODE_NONE,baseline_mode:str=BASELINE_MODE_NONE,blend_weight:float|None=None,optimized_metric:str|None=None,signal_runtimes:dict[str,list[_SignalForecasterRuntime]]|None=None):
		self.device=torch.device(device)
		if signal_runtimes is None:signal_runtimes={WHOLESALE_PRICE_SIGNAL:[self._build_runtime(signal_name=WHOLESALE_PRICE_SIGNAL,model_path=model_path,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,pred_len=pred_len,seq_len=seq_len,scaler=scaler,input_size=input_size,time_feature_mode=time_feature_mode,model_mode=model_mode,physical_normalization_mode=physical_normalization_mode,physical_scale_by_column=physical_scale_by_column,postprocess_mode=postprocess_mode,baseline_mode=baseline_mode,blend_weight=blend_weight,optimized_metric=optimized_metric,device=self.device)]}
		self.signal_runtimes={str(signal_name):list(runtimes)for(signal_name,runtimes)in signal_runtimes.items()}
	@staticmethod
	def _build_runtime(*,signal_name:str,model_path:str|None,hidden_size:int,num_layers:int,dropout:float,pred_len:int,seq_len:int,scaler,device:torch.device,input_size:int=1,time_feature_mode:str=TIME_FEATURE_MODE_NONE,model_mode:str='shared',physical_normalization_mode:str=PHYSICAL_NORMALIZATION_NONE,physical_scale_by_column=None,agent_index:int|None=None,agent_profile:str|None=None,postprocess_mode:str=POSTPROCESS_MODE_NONE,baseline_mode:str=BASELINE_MODE_NONE,blend_weight:float|None=None,optimized_metric:str|None=None,component:str|None=None)->_SignalForecasterRuntime:
		model=LSTMForecastModel(hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,pred_len=pred_len,input_size=input_size).to(device)
		if model_path is not None:model.load_state_dict(torch.load(model_path,map_location=device))
		model.eval();return _SignalForecasterRuntime(signal_name=str(signal_name),seq_len=int(seq_len),pred_len=int(pred_len),hidden_size=int(hidden_size),num_layers=int(num_layers),dropout=float(dropout),model=model,scaler=scaler,input_size=int(input_size),time_feature_mode=normalize_time_feature_mode(time_feature_mode),model_mode=str(model_mode),physical_normalization_mode=_normalize_physical_mode(physical_normalization_mode),physical_scale_by_column=_coerce_physical_scale_array(physical_scale_by_column),agent_index=None if agent_index is None else int(agent_index),agent_profile=None if agent_profile is None else str(agent_profile),postprocess_mode=str(postprocess_mode or POSTPROCESS_MODE_NONE),baseline_mode=str(baseline_mode or BASELINE_MODE_NONE),blend_weight=None if blend_weight is None else float(blend_weight),optimized_metric=None if optimized_metric is None else str(optimized_metric),component=None if component is None else str(component))
	@classmethod
	def from_artifacts(cls,model_path:str,meta_path:str|None=None,scaler_path:str|None=None,device:str|torch.device='cpu',signal_name:str=WHOLESALE_PRICE_SIGNAL):meta,scaler=load_lstm_forecaster_artifacts(model_path=model_path,meta_path=meta_path,scaler_path=scaler_path);return cls(model_path=model_path,hidden_size=int(meta['hidden_size']),num_layers=int(meta['num_layers']),dropout=float(meta['dropout']),pred_len=int(meta['pred_len']),seq_len=int(meta['seq_len']),device=device,scaler=scaler,input_size=int(meta.get('input_size',1)),time_feature_mode=meta.get('time_feature_mode',TIME_FEATURE_MODE_NONE),model_mode=meta.get('model_mode','shared'),physical_normalization_mode=meta.get('normalization_mode',PHYSICAL_NORMALIZATION_NONE),postprocess_mode=meta.get('postprocess_mode',POSTPROCESS_MODE_NONE),baseline_mode=meta.get('baseline_mode',BASELINE_MODE_NONE),blend_weight=meta.get('blend_weight'),optimized_metric=meta.get('optimized_metric'),signal_runtimes=None).rename_default_signal(meta.get('signal_name',signal_name))
	@staticmethod
	def _normalize_artifact_bundle(bundle)->list[tuple[str,str|None,str|None]]:
		if isinstance(bundle,tuple)and len(bundle)==3 and not any(isinstance(item,(list,tuple))for item in bundle):return[bundle]
		if isinstance(bundle,list):return[tuple(item)for item in bundle]if bundle else[]
		if isinstance(bundle,tuple)and bundle and all(isinstance(item,(list,tuple))for item in bundle):return[tuple(item)for item in bundle]
		raise TypeError(f"Unsupported signal artifact bundle: {bundle!r}")
	@classmethod
	def from_signal_artifacts(cls,signal_artifacts:dict[str,object],*,device:str|torch.device='cpu'):device=torch.device(device);signal_runtimes={str(signal_name):[cls._build_runtime(signal_name=meta.get('signal_name',signal_name),model_path=model_path,hidden_size=int(meta['hidden_size']),num_layers=int(meta['num_layers']),dropout=float(meta['dropout']),pred_len=int(meta['pred_len']),seq_len=int(meta['seq_len']),scaler=scaler,input_size=int(meta.get('input_size',1)),time_feature_mode=meta.get('time_feature_mode',TIME_FEATURE_MODE_NONE),model_mode=meta.get('model_mode','shared'),physical_normalization_mode=meta.get('normalization_mode',PHYSICAL_NORMALIZATION_NONE),agent_index=meta.get('agent_index'),agent_profile=meta.get('agent_profile'),postprocess_mode=meta.get('postprocess_mode',POSTPROCESS_MODE_NONE),baseline_mode=meta.get('baseline_mode',BASELINE_MODE_NONE),blend_weight=meta.get('blend_weight'),optimized_metric=meta.get('optimized_metric'),component=meta.get('component'),device=device)for(model_path,meta_path,scaler_path)in cls._normalize_artifact_bundle(bundle)for(meta,scaler)in[load_lstm_forecaster_artifacts(model_path=model_path,meta_path=meta_path,scaler_path=scaler_path)]]for(signal_name,bundle)in signal_artifacts.items()};return cls(device=device,signal_runtimes=signal_runtimes)
	def rename_default_signal(self,signal_name:str):
		if WHOLESALE_PRICE_SIGNAL in self.signal_runtimes and signal_name!=WHOLESALE_PRICE_SIGNAL:
			self.signal_runtimes[str(signal_name)]=self.signal_runtimes.pop(WHOLESALE_PRICE_SIGNAL)
			for runtime in self.signal_runtimes[str(signal_name)]:runtime.signal_name=str(signal_name)
		return self
	def reset(self)->None:
		for runtimes in self.signal_runtimes.values():
			for runtime in runtimes:
				if runtime.physical_normalization_mode!=PHYSICAL_NORMALIZATION_NONE:runtime.physical_scale_by_column=None
	@staticmethod
	def _select_runtime_scale(scale_by_column:np.ndarray|None,runtime:_SignalForecasterRuntime)->np.ndarray|None:
		if scale_by_column is None:return
		if runtime.model_mode=='per_agent'and runtime.agent_index is not None:
			if runtime.agent_index>=scale_by_column.size:raise IndexError(f"Runtime agent_index={runtime.agent_index} is out of range for scale size={scale_by_column.size}.")
			return np.asarray([scale_by_column[runtime.agent_index]],dtype=np.float32)
		return scale_by_column.astype(np.float32,copy=True)
	def set_episode(self,episode_signals:dict[str,np.ndarray]|np.ndarray,episode_meta:dict[str,object]|None=None)->None:
		(self._episode_component_signals):dict[str,np.ndarray]={}
		if isinstance(episode_signals,dict):self._episode_component_signals={key:np.asarray(value,dtype=np.float32)for(key,value)in episode_signals.items()if key.startswith('load_')}
		meta=dict(episode_meta or{});load_scale,pv_peak_kw=_coerce_physical_scale_array(meta.get('load_scale')),_coerce_physical_scale_array(meta.get('pv_peak_kw'))
		if pv_peak_kw is not None and np.any(pv_peak_kw<=.0):raise ValueError("episode_meta['pv_peak_kw'] must stay positive for PV forecast normalization.")
		for(signal_name,runtimes)in self.signal_runtimes.items():
			for runtime in runtimes:
				if runtime.physical_normalization_mode==PHYSICAL_NORMALIZATION_LOAD_SCALE:runtime.physical_scale_by_column=self._select_runtime_scale(load_scale,runtime)
				elif runtime.physical_normalization_mode==PHYSICAL_NORMALIZATION_PV_PEAK:runtime.physical_scale_by_column=self._select_runtime_scale(pv_peak_kw,runtime)
	@staticmethod
	def _resolve_column_scale(runtime:_SignalForecasterRuntime,column_idx:int|None)->np.float32:
		scale_by_column=runtime.physical_scale_by_column
		if scale_by_column is None:return np.float32(1.)
		scale=np.asarray(scale_by_column,dtype=np.float32).reshape(-1)
		if scale.size==0:return np.float32(1.)
		column_idx=0 if column_idx is None else column_idx
		if column_idx>=scale.size and scale.size==1:return np.float32(scale[0])
		if column_idx>=scale.size:raise IndexError(f"physical_scale_by_column is too short for column {column_idx}: size={scale.size}")
		return np.float32(scale[column_idx])
	@staticmethod
	def _normalize_model_input(values:np.ndarray,runtime:_SignalForecasterRuntime,*,column_idx:int|None)->tuple[np.ndarray,np.float32]:
		scale=LSTMForecaster._resolve_column_scale(runtime,column_idx);normalized=np.asarray(values,dtype=np.float32)
		if runtime.physical_normalization_mode!=PHYSICAL_NORMALIZATION_NONE:normalized=(normalized/np.float32(max(float(scale),float(PHYSICAL_SCALE_EPS)))).astype(np.float32)
		return normalized,scale
	@staticmethod
	def _restore_prediction_scale(values:np.ndarray,runtime:_SignalForecasterRuntime,*,scale:np.float32)->np.ndarray:
		restored=np.asarray(values,dtype=np.float32)
		if runtime.physical_normalization_mode!=PHYSICAL_NORMALIZATION_NONE:restored=(restored*np.float32(scale)).astype(np.float32)
		if str(runtime.postprocess_mode)==POSTPROCESS_MODE_PHYSICAL_CLIP:upper=np.float32(max(float(scale),.0))if runtime.physical_normalization_mode==PHYSICAL_NORMALIZATION_PV_PEAK else None;restored=np.clip(restored,np.float32(.0),upper).astype(np.float32)
		return restored
	@staticmethod
	def _coerce_episode_timestamp_index(history_timestamps:Sequence[str|pd.Timestamp]|None,total_length:int)->pd.DatetimeIndex:
		total_length=int(total_length)
		if total_length<=0:return pd.DatetimeIndex([])
		index=coerce_timestamp_index(history_timestamps)
		if len(index)>=total_length:return index[:total_length]
		step_delta=infer_timestamp_step(index)
		if len(index)==0:return pd.date_range(start=DEFAULT_TIMESTAMP_START,periods=total_length,freq=step_delta)
		return index.append(pd.date_range(start=index[-1]+step_delta,periods=total_length-len(index),freq=step_delta))
	@staticmethod
	def _build_padded_history_windows(values:np.ndarray,seq_len:int)->np.ndarray:
		history=np.asarray(values,dtype=np.float32).reshape(-1);seq_len=int(seq_len)
		if history.size==0:return np.zeros((0,seq_len),dtype=np.float32)
		if seq_len<=0:raise ValueError(f"seq_len must be positive, got {seq_len}.")
		return np.asarray(np.lib.stride_tricks.sliding_window_view(np.concatenate([np.zeros((max(seq_len-1,0),),dtype=np.float32),history],axis=0),seq_len)[:history.size],dtype=np.float32)
	@staticmethod
	def _append_prediction_chunk_to_windows(windows:np.ndarray,prediction_chunk:np.ndarray)->np.ndarray:
		base=np.asarray(windows,dtype=np.float32);chunk=np.asarray(prediction_chunk,dtype=np.float32)
		if chunk.ndim==1:chunk=chunk[:,None]
		if chunk.size==0:return base
		if chunk.shape[1]>=base.shape[1]:return chunk[:,-base.shape[1]:].astype(np.float32,copy=False)
		return np.concatenate([base[:,chunk.shape[1]:],chunk],axis=1).astype(np.float32,copy=False)
	@staticmethod
	def _build_time_window_view(timestamp_index:pd.DatetimeIndex,*,seq_len:int,horizon:int,time_feature_mode:str)->np.ndarray|None:
		normalized_mode=normalize_time_feature_mode(time_feature_mode)
		if normalized_mode==TIME_FEATURE_MODE_NONE:return
		seq_len=int(seq_len);horizon=int(horizon);total_steps=len(timestamp_index)
		if total_steps==0:return np.zeros((0,seq_len,0),dtype=np.float32)
		step_delta=infer_timestamp_step(timestamp_index);full_index=pd.date_range(end=timestamp_index[0]-step_delta,periods=max(seq_len-1,0),freq=step_delta).append(timestamp_index).append(pd.date_range(start=timestamp_index[-1]+step_delta,periods=max(horizon-1,0),freq=step_delta));return np.transpose(np.lib.stride_tricks.sliding_window_view(encode_forecast_time_features(full_index,normalized_mode),seq_len,axis=0),(0,2,1)).astype(np.float32,copy=False)
	def _predict_model_chunk_batch(self,runtime:_SignalForecasterRuntime,history_windows:np.ndarray,*,time_windows:np.ndarray|None,column_idx:int|None,batch_size:int)->np.ndarray:
		histories=np.asarray(history_windows,dtype=np.float32)
		if histories.size==0:return np.zeros((0,int(runtime.pred_len)),dtype=np.float32)
		batch_size=max(int(batch_size),1);predictions:list[np.ndarray]=[];normalized_mode=normalize_time_feature_mode(runtime.time_feature_mode if int(runtime.input_size)>1 else'none')
		with torch.inference_mode():
			for start in range(0,len(histories),batch_size):
				end=start+batch_size;history_batch=histories[start:end];normalized_history,scale=self._normalize_model_input(history_batch,runtime,column_idx=column_idx);load_channel=runtime.scaler.transform(normalized_history.reshape(-1,1)).reshape(normalized_history.shape).astype(np.float32)if runtime.scaler is not None else normalized_history.astype(np.float32)
				if int(runtime.input_size)<=1 or normalized_mode==TIME_FEATURE_MODE_NONE:features=load_channel
				else:
					if time_windows is None:raise ValueError('time_windows are required when runtime.input_size > 1.')
					features=np.concatenate([load_channel[:,:,None],np.asarray(time_windows[start:end],dtype=np.float32)],axis=2).astype(np.float32)
				tensor=torch.tensor(features,dtype=torch.float32,device=self.device);prediction=runtime.model(tensor).detach().cpu().numpy().astype(np.float32)
				if runtime.scaler is not None:prediction=runtime.scaler.inverse_transform(prediction.reshape(-1,1)).reshape(prediction.shape).astype(np.float32)
				predictions.append(self._restore_prediction_scale(prediction,runtime,scale=scale))
		if not predictions:return np.zeros((0,int(runtime.pred_len)),dtype=np.float32)
		return np.concatenate(predictions,axis=0).astype(np.float32,copy=False)
	def _predict_univariate_episode_matrix(self,runtime:_SignalForecasterRuntime,history:np.ndarray,horizon:int,*,column_idx:int|None=None,history_timestamps:Sequence[str|pd.Timestamp]|None=None,batch_size:int=8192)->np.ndarray:
		values=np.asarray(history,dtype=np.float32).reshape(-1);total_steps,horizon=int(values.size),int(horizon)
		if horizon<=0:return np.zeros((total_steps,0),dtype=np.float32)
		if total_steps==0:return np.zeros((0,horizon),dtype=np.float32)
		current_value=values[:,None].astype(np.float32,copy=False)
		if horizon==1:return current_value.copy()
		timestamp_index=self._coerce_episode_timestamp_index(history_timestamps,total_steps);rolling_windows=self._build_padded_history_windows(values,runtime.seq_len);time_window_view=self._build_time_window_view(timestamp_index,seq_len=runtime.seq_len,horizon=horizon,time_feature_mode=runtime.time_feature_mode if int(runtime.input_size)>1 else TIME_FEATURE_MODE_NONE);remaining=horizon-1
		if str(runtime.postprocess_mode)==POSTPROCESS_MODE_BASELINE_BLEND:
			if str(runtime.baseline_mode)!=BASELINE_MODE_LAST_VALUE:raise ValueError(f"Unsupported baseline_mode '{runtime.baseline_mode}'.")
			future=np.empty((total_steps,remaining),dtype=np.float32);weight=np.float32(1. if runtime.blend_weight is None else float(runtime.blend_weight))
			for step_idx in range(remaining):
				current_time_windows=None if time_window_view is None else time_window_view[step_idx:step_idx+total_steps];raw_batch=self._predict_model_chunk_batch(runtime,rolling_windows,time_windows=current_time_windows,column_idx=column_idx,batch_size=batch_size);baseline_values=rolling_windows[:,-1].astype(np.float32,copy=False);next_values=(baseline_values+weight*(raw_batch[:,0]-baseline_values)).astype(np.float32);future[:,step_idx]=next_values
				if step_idx+1<remaining:rolling_windows=self._append_prediction_chunk_to_windows(rolling_windows,next_values[:,None])
			return np.concatenate([current_value,future],axis=1).astype(np.float32,copy=False)
		future_chunks:list[np.ndarray]=[];produced=0
		while produced<remaining:
			current_time_windows=None if time_window_view is None else time_window_view[produced:produced+total_steps];raw_batch=self._predict_model_chunk_batch(runtime,rolling_windows,time_windows=current_time_windows,column_idx=column_idx,batch_size=batch_size);take=min(int(runtime.pred_len),remaining-produced);prediction_chunk=np.asarray(raw_batch[:,:take],dtype=np.float32);future_chunks.append(prediction_chunk);produced+=take
			if produced<remaining:rolling_windows=self._append_prediction_chunk_to_windows(rolling_windows,prediction_chunk)
		future=np.concatenate(future_chunks,axis=1).astype(np.float32,copy=False);return np.concatenate([current_value,future],axis=1).astype(np.float32,copy=False)
	def predict(self,history:np.ndarray,horizon:int,*,signal_name:str=WHOLESALE_PRICE_SIGNAL,history_timestamps:Sequence[str|pd.Timestamp]|None=None)->np.ndarray:
		history=np.asarray(history,dtype=np.float32);prediction_matrix=self.predict_episode_matrix(history,horizon,signal_name=signal_name,history_timestamps=history_timestamps)
		if prediction_matrix.shape[0]==0:
			if history.ndim in{1,2}:return np.zeros((max(int(horizon),0),),dtype=np.float32)if history.ndim==1 else np.zeros((history.shape[1],max(int(horizon),0)),dtype=np.float32)
			raise ValueError(f"LSTMForecaster expects 1D or 2D history, got shape {history.shape}")
		return np.asarray(prediction_matrix[-1],dtype=np.float32)
	def predict_episode_matrix(self,history:np.ndarray,horizon:int,*,signal_name:str=WHOLESALE_PRICE_SIGNAL,history_timestamps:Sequence[str|pd.Timestamp]|None=None,batch_size:int=8192)->np.ndarray:
		if signal_name not in self.signal_runtimes:
			if len(self.signal_runtimes)==1:signal_name=next(iter(self.signal_runtimes))
			else:raise KeyError(f"LSTMForecaster has no runtime model for '{signal_name}'. Available: {sorted(self.signal_runtimes)}")
		runtimes=self.signal_runtimes[signal_name];history=np.asarray(history,dtype=np.float32)
		if any(r.component is not None for r in runtimes):return self._predict_component_split_episode_matrix(runtimes,history,horizon,history_timestamps=history_timestamps,batch_size=batch_size)
		if history.ndim==1:return self._predict_univariate_episode_matrix(runtimes[0],history,horizon,column_idx=0,history_timestamps=history_timestamps,batch_size=batch_size)
		if history.ndim!=2:raise ValueError(f"LSTMForecaster expects 1D or 2D history, got shape {history.shape}")
		per_column=len(runtimes)==history.shape[1];predictions=[self._predict_univariate_episode_matrix(runtimes[column_idx]if per_column else runtimes[0],history[:,column_idx],horizon,column_idx=0 if per_column else column_idx,history_timestamps=history_timestamps,batch_size=batch_size)for column_idx in range(history.shape[1])];return np.stack(predictions,axis=1).astype(np.float32,copy=False)
	def _predict_component_split_episode_matrix(self,runtimes:list[_SignalForecasterRuntime],history:np.ndarray,horizon:int,*,history_timestamps:Sequence[str|pd.Timestamp]|None=None,batch_size:int=8192)->np.ndarray:
		n_agents=history.shape[1]if history.ndim==2 else 1;total_steps=history.shape[0]if history.ndim>=1 else 0;comp_signals=getattr(self,'_episode_component_signals',{});agent_predictions:list[np.ndarray]=[]
		for agent_idx in range(n_agents):
			agent_runtimes=[runtime for runtime in runtimes if runtime.agent_index==agent_idx];agent_total=np.zeros((total_steps,int(horizon)),dtype=np.float32)
			for runtime in agent_runtimes:comp_key=f"load_{runtime.component}";comp_history=np.asarray(comp_signals[comp_key],dtype=np.float32)[:total_steps,agent_idx]if comp_key in comp_signals else history[:,agent_idx]if history.ndim==2 else history;agent_total+=self._predict_univariate_episode_matrix(runtime,comp_history,horizon,column_idx=0,history_timestamps=history_timestamps,batch_size=batch_size)
			agent_predictions.append(agent_total.astype(np.float32,copy=False))
		return np.stack(agent_predictions,axis=1).astype(np.float32,copy=False)
