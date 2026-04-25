from __future__ import annotations
import copy,json
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import RobustScaler,StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from tqdm.auto import tqdm
from data.loaders.prosumer import ProsumerDataset
from scripts.utils.torch_runtime import TorchRuntimeState,configure_torch_runtime,resolve_device
from predictors.lstm_forecaster import BASELINE_MODE_LAST_VALUE,BASELINE_MODE_NONE,DEFAULT_SUPPORTED_FORECAST_SIGNALS,LSTMForecaster,LSTMForecastModel,LSTM_ARTIFACT_FORMAT,LSTM_LOAD_HYBRID_ARTIFACT_FORMAT,PHYSICAL_NORMALIZATION_LOAD_SCALE,PHYSICAL_NORMALIZATION_NONE,PHYSICAL_NORMALIZATION_PV_PEAK,POSTPROCESS_MODE_BASELINE_BLEND,POSTPROCESS_MODE_NONE,POSTPROCESS_MODE_PHYSICAL_CLIP,get_default_lstm_artifact_dir,get_default_lstm_artifact_paths,save_lstm_forecaster_artifacts
from predictors.time_features import TIME_FEATURE_MODE_HOUR_WEEK_YEAR,TIME_FEATURE_MODE_NONE,coerce_timestamp_index,encode_forecast_time_features,infer_timestamp_step,normalize_time_feature_mode,time_feature_dim
from scripts.utils.price_protocol import WHOLESALE_PRICE_SIGNAL,normalize_internal_signal_name
DEFAULT_WEEK_STEPS,PHYSICAL_SCALE_EPS,HEATPUMP_BLOCKED_BIAS_GUARD_KW=672,np.float32(1e-06),.15
SIGNAL_TRAINING_OVERRIDE_FIELDS={'history_window':'history_window','hidden_size':'lstm_hidden_size','num_layers':'lstm_num_layers','dropout':'lstm_dropout','batch_size':'lstm_batch_size','epochs':'lstm_epochs','lr':'lstm_lr','train_ratio':'lstm_train_ratio','val_ratio':'lstm_val_ratio'}
HEATPUMP_BLOCKED_MONTH_GROUPS=('Jan-Mar',(1,2,3)),('Apr-Jun',(4,5,6)),('Jul-Sep',(7,8,9)),('Oct-Dec',(10,11,12))
def _validate_choice(name:str,value:str|None,default:str,allowed:set[str])->str:
	normalized=str(default if value in(None,'')else value).strip().lower()
	if normalized not in allowed:raise ValueError(f"Unsupported {name} '{value}'. Only {', '.join(repr(item)for item in sorted(allowed))} are implemented.")
	return normalized
def _normalize_load_blend_candidates(candidates:Sequence[float]|None)->tuple[float,...]:
	if candidates is None:return tuple(float(index)/1e1 for index in range(11))
	normalized=tuple(float(value)for value in candidates)
	if not normalized:raise ValueError('forecast.load_blend_candidates must contain at least one candidate weight.')
	if any(value<.0 or value>1. for value in normalized):raise ValueError('forecast.load_blend_candidates values must stay within [0.0, 1.0].')
	return normalized
def _normalize_date_range(start_date,end_date)->dict[str,str|None]:return{'start_date':None if start_date in(None,'')else str(start_date),'end_date':None if end_date in(None,'')else str(end_date)}
def resolve_signal_physical_normalization_mode(signal_name:str)->str:normalized=_normalize_signal_name(signal_name);return PHYSICAL_NORMALIZATION_LOAD_SCALE if normalized=='load'else PHYSICAL_NORMALIZATION_PV_PEAK if normalized=='pv'else PHYSICAL_NORMALIZATION_NONE
def _resolve_signal_behavior(cfg,signal_name:str)->dict[str,object]:
	normalized=_normalize_signal_name(signal_name)
	if normalized=='load':postprocess_mode=_validate_choice('forecast.load_hybrid_mode',getattr(cfg.forecast,'load_hybrid_mode',POSTPROCESS_MODE_BASELINE_BLEND),POSTPROCESS_MODE_BASELINE_BLEND,{POSTPROCESS_MODE_BASELINE_BLEND});return{'model_mode':_validate_choice('forecast.load_model_mode',getattr(cfg.forecast,'load_model_mode','per_agent'),'per_agent',{'per_agent'}),'time_feature_mode':normalize_time_feature_mode(getattr(cfg.forecast,'load_time_feature_mode',TIME_FEATURE_MODE_NONE)),'postprocess_mode':postprocess_mode,'baseline_mode':_validate_choice('forecast.load_baseline_mode',getattr(cfg.forecast,'load_baseline_mode',BASELINE_MODE_LAST_VALUE),BASELINE_MODE_LAST_VALUE,{BASELINE_MODE_LAST_VALUE}),'blend_candidates':_normalize_load_blend_candidates(getattr(cfg.forecast,'load_blend_candidates',None)),'scaler_type':str(getattr(cfg.forecast,'load_scaler_type','standard')).strip().lower(),'component_split':bool(getattr(cfg.forecast,'load_component_split',False)),'artifact_format':LSTM_LOAD_HYBRID_ARTIFACT_FORMAT if postprocess_mode==POSTPROCESS_MODE_BASELINE_BLEND else LSTM_ARTIFACT_FORMAT}
	if normalized=='pv':return{'model_mode':'shared','time_feature_mode':normalize_time_feature_mode(getattr(cfg.forecast,'pv_time_feature_mode',TIME_FEATURE_MODE_HOUR_WEEK_YEAR)),'postprocess_mode':_validate_choice('forecast.pv_postprocess_mode',getattr(cfg.forecast,'pv_postprocess_mode',POSTPROCESS_MODE_PHYSICAL_CLIP),POSTPROCESS_MODE_PHYSICAL_CLIP,{POSTPROCESS_MODE_NONE,POSTPROCESS_MODE_PHYSICAL_CLIP}),'baseline_mode':BASELINE_MODE_NONE,'blend_candidates':(1.,),'scaler_type':'standard','component_split':False,'artifact_format':LSTM_ARTIFACT_FORMAT}
	return{'model_mode':'shared','time_feature_mode':TIME_FEATURE_MODE_NONE,'postprocess_mode':POSTPROCESS_MODE_NONE,'baseline_mode':BASELINE_MODE_NONE,'blend_candidates':(1.,),'scaler_type':'standard','component_split':False,'artifact_format':LSTM_ARTIFACT_FORMAT}
def build_lstm_source_signature(cfg,signal_name:str)->dict[str,object]:
	normalized_signal=_normalize_signal_name(signal_name);train_exclusion={'start_date':None,'end_date':None}
	if int(cfg.data.train_year)==int(cfg.data.test_year)and not cfg.data.train_start_date and not cfg.data.train_end_date and(cfg.data.test_start_date or cfg.data.test_end_date):train_exclusion=_normalize_date_range(cfg.data.test_start_date,cfg.data.test_end_date)
	return{'signal_name':normalized_signal,'agent_profiles':[str(profile)for profile in cfg.data.agent_profiles],'train_year':int(cfg.data.train_year),'test_year':int(cfg.data.test_year),'train_date_range':_normalize_date_range(cfg.data.train_start_date,cfg.data.train_end_date),'test_date_range':{'start_date':None,'end_date':None},'train_excluded_date_range':train_exclusion,'load_components':[str(component)for component in cfg.data.load_components],'pv_reference':str(cfg.data.pv_reference),'pv_capacity_kw':[float(value)for value in cfg.data.pv_capacity_kw or[]],'num_agents':int(cfg.env.num_agents),'future_horizon':int(cfg.env.future_horizon),'history_window':int(cfg.forecast.history_window)}
def _coerce_physical_scale_by_column(physical_scale_by_column:Sequence[float]|np.ndarray|float|None,*,expected_size:int)->np.ndarray|None:
	if physical_scale_by_column is None:return
	scale=np.asarray(physical_scale_by_column,dtype=np.float32).reshape(-1)
	if scale.size==0:return
	if scale.size==1 and expected_size>1:scale=np.repeat(scale,expected_size)
	elif scale.size!=expected_size:raise ValueError(f"physical_scale_by_column size mismatch: expected {expected_size}, got {scale.size}")
	return scale.astype(np.float32,copy=True)
def _apply_physical_normalization(values:np.ndarray,physical_scale_by_column:Sequence[float]|np.ndarray|float|None)->np.ndarray:
	normalized=np.asarray(values,dtype=np.float32)
	if normalized.ndim==1:expected_size=1
	elif normalized.ndim==2:expected_size=normalized.shape[1]
	else:raise ValueError(f"Expected 1D or 2D values for physical normalization, got shape {normalized.shape}")
	scale=_coerce_physical_scale_by_column(physical_scale_by_column,expected_size=expected_size)
	if scale is None:return normalized.astype(np.float32,copy=True)
	divisor=np.maximum(scale,np.float32(PHYSICAL_SCALE_EPS)).astype(np.float32)
	if normalized.ndim==1:return(normalized/divisor[0]).astype(np.float32)
	return(normalized/divisor[None,:]).astype(np.float32)
def _slice_scale_by_source_columns(values:Sequence[float]|np.ndarray|float|None,source:'SignalCsvSource')->np.ndarray|None:
	if values is None:return
	scale=np.asarray(values,dtype=np.float32).reshape(-1)
	if scale.size==0:return
	indices=tuple(source.column_indices or tuple(range(len(source.value_columns))))
	if not indices:return
	if scale.size==len(indices)and indices==tuple(range(len(indices))):return scale.astype(np.float32,copy=True)
	if any(index<0 or index>=scale.size for index in indices):raise IndexError(f"Signal source column_indices={indices} are out of bounds for scale size={scale.size}.")
	return scale[list(indices)].astype(np.float32,copy=True)
def resolve_signal_physical_scale_from_source(source:SignalCsvSource,signal_name:str,*,split:str)->np.ndarray|None:
	mode=resolve_signal_physical_normalization_mode(signal_name)
	if mode==PHYSICAL_NORMALIZATION_NONE:return
	dataset_kwargs=dict((source.dataset_kwargs or{}).get(split)or{})
	if not dataset_kwargs:return
	if mode==PHYSICAL_NORMALIZATION_LOAD_SCALE:return _coerce_physical_scale_by_column(_slice_scale_by_source_columns(dataset_kwargs.get('load_scale'),source),expected_size=len(source.value_columns))
	if mode==PHYSICAL_NORMALIZATION_PV_PEAK:
		dataset=ProsumerDataset(**dataset_kwargs);pv_peak_kw=np.asarray(dataset._meta_template.get('pv_peak_kw',[]),dtype=np.float32).reshape(-1)
		if pv_peak_kw.size==0:return
		if np.any(pv_peak_kw<=.0):raise ValueError('ProsumerDataset produced non-positive pv_peak_kw for PV forecast normalization.')
		return _coerce_physical_scale_by_column(_slice_scale_by_source_columns(pv_peak_kw,source),expected_size=len(source.value_columns))
class SignalCsvSource:
	def __init__(self,**kwargs):self.__dict__.update(kwargs)
class SignalForecastEvaluation:
	def __init__(self,**kwargs):self.__dict__.update(kwargs)
def _normalize_signal_name(signal_name:str)->str:return normalize_internal_signal_name(signal_name)
_WHOLESALE_PRICE_DERIVED_OBSERVATION_FEATURES={'wholesale_price_relative','wholesale_price_spread','wholesale_price_rank'}
def _forecast_signal_for_observation_feature(feature_name:str)->str:
	normalized=_normalize_signal_name(feature_name)
	return WHOLESALE_PRICE_SIGNAL if normalized in _WHOLESALE_PRICE_DERIVED_OBSERVATION_FEATURES else normalized
def configured_forecast_signals(cfg)->list[str]:unique=list(dict.fromkeys(normalized for signal_name in cfg.forecast.target_signals if(normalized:=_normalize_signal_name(signal_name))));return unique or list(DEFAULT_SUPPORTED_FORECAST_SIGNALS)
def required_forecast_signals(cfg)->list[str]:
	configured,active=set(configured_forecast_signals(cfg)),[]
	for signal_name in cfg.obs.sequence_features:
		if(normalized:=_forecast_signal_for_observation_feature(signal_name))in configured and normalized not in active:active.append(normalized)
	return active or configured_forecast_signals(cfg)
def clone_config_with_signal_overrides(cfg,overrides:dict[str,object]|None=None):
	cloned=copy.deepcopy(cfg)
	if not overrides:return cloned
	unknown_keys=sorted(key for key in overrides if key not in SIGNAL_TRAINING_OVERRIDE_FIELDS and key not in{'future_horizon','device'})
	if unknown_keys:raise KeyError(f"Unknown signal training override keys: {unknown_keys}. Allowed keys: {sorted(SIGNAL_TRAINING_OVERRIDE_FIELDS)} + ['future_horizon', 'device'].")
	for(key,value)in overrides.items():
		if key=='future_horizon':cloned.env.future_horizon=int(value);continue
		if key=='device':cloned.runtime.device=resolve_device(value);continue
		setattr(cloned.forecast,SIGNAL_TRAINING_OVERRIDE_FIELDS[key],value)
	return cloned
def resolve_signal_training_settings(cfg,signal_name:str,overrides:dict[str,object]|None=None)->tuple[object,dict[str,object]]:signal_name=_normalize_signal_name(signal_name);local_cfg=clone_config_with_signal_overrides(cfg,overrides=overrides);behavior=_resolve_signal_behavior(local_cfg,signal_name);settings={'signal_name':signal_name,'history_window':int(local_cfg.forecast.history_window),'future_horizon':int(local_cfg.env.future_horizon),'hidden_size':int(local_cfg.forecast.lstm_hidden_size),'num_layers':int(local_cfg.forecast.lstm_num_layers),'dropout':float(local_cfg.forecast.lstm_dropout),'batch_size':int(local_cfg.forecast.lstm_batch_size),'epochs':int(local_cfg.forecast.lstm_epochs),'lr':float(local_cfg.forecast.lstm_lr),'train_ratio':float(local_cfg.forecast.lstm_train_ratio),'val_ratio':float(local_cfg.forecast.lstm_val_ratio),'device':str(local_cfg.runtime.device),'input_size':1+int(time_feature_dim(behavior['time_feature_mode'])),**behavior};return local_cfg,settings
def _resolve_prosumer_dataset_kwargs(cfg,data_dir:Path,split:str)->dict[str,object]:
	year=int(cfg.data.train_year if split=='train'else cfg.data.test_year);start_date=cfg.data.train_start_date if split=='train'else cfg.data.test_start_date;end_date=cfg.data.train_end_date if split=='train'else cfg.data.test_end_date;n_agents=int(cfg.env.num_agents);agent_profiles=[str(profile)for profile in cfg.data.agent_profiles]
	if len(agent_profiles)!=n_agents:raise ValueError(f"Forecast prosumer config mismatch: cfg.env.num_agents={n_agents} but cfg.data.agent_profiles has {len(agent_profiles)} entry(ies). Keep forecast notebook/config values in sync, for example: cfg.env.num_agents = len(cfg.data.agent_profiles).")
	exclude_start_date=None;exclude_end_date=None
	if split=='train'and int(cfg.data.train_year)==int(cfg.data.test_year)and not cfg.data.train_start_date and not cfg.data.train_end_date and(cfg.data.test_start_date or cfg.data.test_end_date):exclude_start_date=cfg.data.test_start_date;exclude_end_date=cfg.data.test_end_date
	return{'data_dir':data_dir,'episode_length':1,'n_agents':n_agents,'agent_profiles':agent_profiles,'year':year,'start_date':start_date,'end_date':end_date,'exclude_start_date':exclude_start_date,'exclude_end_date':exclude_end_date,'load_components':list(cfg.data.load_components),'pv_reference':str(cfg.data.pv_reference),'pv_capacity_kw':list(cfg.data.pv_capacity_kw),'load_scale':list(cfg.data.load_scale),'pv_scale':list(cfg.data.pv_scale),'node_ids':list(range(n_agents))}
def _prosumer_value_columns(agent_profiles:Sequence[str],signal_name:str)->tuple[str,...]:signal_name=_normalize_signal_name(signal_name);return(WHOLESALE_PRICE_SIGNAL,)if signal_name==WHOLESALE_PRICE_SIGNAL else tuple(f"{signal_name}_{profile}"for profile in agent_profiles)
def _resolve_prosumer_signal_source(cfg,data_dir:Path,signal_name:str)->SignalCsvSource|None:
	known_components={f"load_{comp}"for comp in cfg.data.load_components}
	if signal_name not in{WHOLESALE_PRICE_SIGNAL,'load','pv'}and signal_name not in known_components:return
	agent_profiles=[str(profile)for profile in cfg.data.agent_profiles];base_signal='load'if signal_name in known_components else signal_name;value_columns=_prosumer_value_columns(agent_profiles,base_signal);return SignalCsvSource(source_kind='prosumer',signal_name=signal_name,value_columns=value_columns,column_indices=tuple(range(len(value_columns))),dataset_kwargs={'train':_resolve_prosumer_dataset_kwargs(cfg,data_dir,'train'),'test':_resolve_prosumer_dataset_kwargs(cfg,data_dir,'test')})
def _load_prosumer_signal_frame_from_source(source:SignalCsvSource,signal_name:str,*,split:str)->tuple[pd.DataFrame,tuple[str,...]]:
	dataset_kwargs=dict((source.dataset_kwargs or{}).get(split)or{})
	if not dataset_kwargs:raise ValueError(f"Prosumer signal source is missing dataset kwargs for split='{split}'.")
	dataset=ProsumerDataset(**dataset_kwargs);timestamps=pd.Series(dataset._timestamps).reset_index(drop=True);dataset_signal_key=source.signal_name if source.signal_name in dataset._signals else signal_name;signal_values=np.asarray(dataset._signals[dataset_signal_key],dtype=np.float32)
	if signal_values.ndim==1:value_columns,frame=(WHOLESALE_PRICE_SIGNAL,),pd.DataFrame({'timestamp':timestamps,WHOLESALE_PRICE_SIGNAL:signal_values})
	else:column_indices=tuple(source.column_indices or tuple(range(signal_values.shape[1])));signal_values=signal_values[:,list(column_indices)];value_columns=tuple(source.value_columns);frame=pd.DataFrame(signal_values,columns=list(value_columns));frame.insert(0,'timestamp',timestamps)
	frame['segment_id']=0;return frame,value_columns
def _load_signal_matrix_from_source(source:SignalCsvSource,signal_name:str,*,split:str)->tuple[pd.DataFrame,np.ndarray,tuple[str,...]]:frame,value_columns=_load_prosumer_signal_frame_from_source(source,signal_name,split=split);values=frame.loc[:,list(value_columns)].to_numpy(dtype=np.float32);return frame,values.reshape(-1)if values.ndim==2 and values.shape[1]==1 else values,value_columns
def select_signal_source_columns(source:SignalCsvSource,column_indices:Sequence[int])->SignalCsvSource:
	selected_indices=tuple(int(index)for index in column_indices)
	if not selected_indices:raise ValueError('select_signal_source_columns requires at least one column index.')
	if any(index<0 or index>=len(source.value_columns)for index in selected_indices):raise IndexError(f"Requested column_indices={selected_indices} for source columns={source.value_columns}.")
	return SignalCsvSource(source_kind=getattr(source,'source_kind',None),signal_name=source.signal_name,value_columns=tuple(source.value_columns[index]for index in selected_indices),column_indices=selected_indices,dataset_kwargs=copy.deepcopy(source.dataset_kwargs))
def _load_signal_segment_frames_from_source(source:SignalCsvSource,signal_name:str,*,split:str,drop_warmup:bool=False)->tuple[pd.DataFrame,list[pd.DataFrame],tuple[str,...]]:
	frame,value_columns=_load_prosumer_signal_frame_from_source(source,signal_name,split=split)
	if drop_warmup and'is_warmup'in frame.columns:frame=frame.loc[~frame['is_warmup'].astype(bool)].copy()
	if frame.empty:raise ValueError(f"Signal source split='{split}' does not contain any usable rows for signal '{signal_name}'.")
	working=frame.copy()
	if'segment_id'not in working.columns:working['segment_id']=0
	working['segment_id']=pd.to_numeric(working['segment_id'],errors='coerce').fillna(-1).astype(int);segment_frames:list[pd.DataFrame]=[]
	for segment_id in sorted(working['segment_id'].unique()):
		segment_frame=working.loc[working['segment_id']==segment_id].reset_index(drop=True)
		if segment_frame.empty:continue
		segment_frames.append(segment_frame)
	if not segment_frames:raise ValueError(f"Signal source split='{split}' does not contain any segment frames for signal '{signal_name}'.")
	return working.reset_index(drop=True),segment_frames,value_columns
def _slice_copy(item,start:int,end:int):return item.iloc[start:end].copy()if isinstance(item,pd.DataFrame)else np.asarray(item[start:end],dtype=np.float32).copy()
def _temporal_split_items(items:Sequence[pd.DataFrame|np.ndarray],*,train_ratio:float=.7,val_ratio:float=.15,seq_len:int,pred_len:int)->dict[str,object]:
	min_window=int(seq_len)+int(pred_len)
	if train_ratio<=.0 or val_ratio<=.0 or train_ratio+val_ratio>=1.:raise ValueError('Expected 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1.')
	train_items:list[pd.DataFrame|np.ndarray]=[];val_items:list[pd.DataFrame|np.ndarray]=[];segment_summaries:list[dict[str,int]]=[]
	for(segment_idx,item)in enumerate(items):
		total_steps=len(item);train_end,val_end=int(total_steps*train_ratio),min(total_steps,int(total_steps*(train_ratio+val_ratio)));val_context_start=max(0,train_end-int(seq_len))
		if train_end>=min_window:train_items.append(_slice_copy(item,0,train_end))
		if val_end-train_end>0 and val_end-val_context_start>=min_window:val_items.append(_slice_copy(item,val_context_start,val_end))
		segment_summaries.append({'segment_idx':int(segment_idx),'total_steps':total_steps,'train_end':int(train_end),'val_end':int(val_end)})
	if not train_items:raise ValueError('No train segments are long enough for the requested history/prediction windows.')
	if not val_items:
		fallback=max(train_items,key=len);fallback_tail=_slice_copy(fallback,max(0,len(fallback)-(2*int(seq_len)+int(pred_len))),len(fallback))
		if len(fallback_tail)<min_window:raise ValueError('No validation segments are long enough for the requested history/prediction windows.')
		val_items=[fallback_tail]
	return{'train_segments':train_items,'val_segments':val_items,'segment_summaries':segment_summaries}
def _build_series_window_pair(series:np.ndarray,*,total_window:int,seq_len:int,scaler:object|None)->tuple[np.ndarray,np.ndarray]:
	normalized=np.asarray(series,dtype=np.float32)
	if scaler is not None:normalized=scaler.transform(normalized.reshape(-1,1)).reshape(-1).astype(np.float32)
	windows=np.lib.stride_tricks.sliding_window_view(normalized,total_window).astype(np.float32);return windows[:,:seq_len].astype(np.float32),windows[:,seq_len:].astype(np.float32)
def _build_time_feature_history(segment_frame:pd.DataFrame,*,seq_len:int,pred_len:int,time_feature_mode:str)->np.ndarray:
	total_window=int(seq_len)+int(pred_len);timestamps=coerce_timestamp_index(segment_frame['timestamp'].tolist());time_features=encode_forecast_time_features(timestamps,time_feature_mode).astype(np.float32)
	if time_features.shape[1]==0:return np.empty((len(segment_frame)-total_window+1,int(seq_len),0),dtype=np.float32)
	windows=np.lib.stride_tricks.sliding_window_view(time_features,window_shape=total_window,axis=0);return np.moveaxis(windows,-1,1).astype(np.float32)[:,:seq_len,:]
def build_supervised_windows_from_time_feature_segments(segment_frames:list[pd.DataFrame],*,value_columns:Sequence[str],seq_len:int,pred_len:int,scaler:object|None,physical_scale_by_column:Sequence[float]|np.ndarray|float|None=None,time_feature_mode:str=TIME_FEATURE_MODE_NONE)->tuple[np.ndarray,np.ndarray]:
	total_window=int(seq_len)+int(pred_len);x_parts:list[np.ndarray]=[];y_parts:list[np.ndarray]=[]
	for segment_frame in segment_frames:
		values=_reshape_signal_values(segment_frame.loc[:,list(value_columns)].to_numpy(dtype=np.float32))
		if values.shape[0]<total_window:continue
		normalized_values=_apply_physical_normalization(values,physical_scale_by_column);time_history=_build_time_feature_history(segment_frame,seq_len=seq_len,pred_len=pred_len,time_feature_mode=time_feature_mode)
		for column_idx in range(normalized_values.shape[1]):value_history,targets=_build_series_window_pair(normalized_values[:,column_idx],total_window=total_window,seq_len=seq_len,scaler=scaler);value_history=value_history.reshape(-1,int(seq_len),1).astype(np.float32);x_parts.append(np.concatenate([value_history,time_history],axis=2)if time_history.shape[2]else value_history);y_parts.append(targets)
	if not x_parts:raise ValueError('No valid supervised windows could be built from the provided multi-column time-feature segments.')
	return np.concatenate(x_parts,axis=0),np.concatenate(y_parts,axis=0)
def resolve_signal_csv_source(data_dir:str|Path,signal_name:str,cfg)->SignalCsvSource|None:return _resolve_prosumer_signal_source(cfg,Path(data_dir),_normalize_signal_name(signal_name))
def _reshape_signal_values(values:np.ndarray)->np.ndarray:values=np.asarray(values,dtype=np.float32);return values.reshape(-1,1)if values.ndim==1 else values
def fit_signal_scaler(values:np.ndarray|list[np.ndarray],*,physical_scale_by_column:Sequence[float]|np.ndarray|float|None=None,scaler_type:str='standard')->StandardScaler|RobustScaler:
	if isinstance(values,list):flattened=np.concatenate([_apply_physical_normalization(np.asarray(chunk,dtype=np.float32),physical_scale_by_column).reshape(-1)for chunk in values],axis=0)
	else:flattened=_apply_physical_normalization(np.asarray(values,dtype=np.float32),physical_scale_by_column).reshape(-1)
	if str(scaler_type).lower()=='robust':scaler=RobustScaler()
	else:scaler=StandardScaler()
	scaler.fit(flattened.reshape(-1,1));return scaler
def summarize_signal_values(values:np.ndarray|list[np.ndarray])->dict[str,float]:flattened=np.concatenate([np.asarray(chunk,dtype=np.float32).reshape(-1)for chunk in values],axis=0)if isinstance(values,list)else np.asarray(values,dtype=np.float32).reshape(-1);return{'min':float(np.min(flattened)),'max':float(np.max(flattened)),'mean':float(np.mean(flattened)),'std':float(np.std(flattened))}
def _build_supervised_windows_from_arrays(chunks:Sequence[np.ndarray],*,seq_len:int,pred_len:int,scaler:object|None,physical_scale_by_column:Sequence[float]|np.ndarray|float|None=None,source_label:str='segment')->tuple[np.ndarray,np.ndarray]:
	total_window=int(seq_len)+int(pred_len);x_parts:list[np.ndarray]=[];y_parts:list[np.ndarray]=[];skipped_chunks=0
	for chunk in chunks:
		values=_apply_physical_normalization(_reshape_signal_values(chunk),physical_scale_by_column)
		if values.shape[0]<total_window:skipped_chunks+=1;continue
		for column_idx in range(values.shape[1]):x_chunk,y_chunk=_build_series_window_pair(values[:,column_idx],total_window=total_window,seq_len=seq_len,scaler=scaler);x_parts.append(x_chunk);y_parts.append(y_chunk)
	if x_parts:
		if skipped_chunks and len(chunks)>1:print(f"[forecast] skipped {skipped_chunks} short {source_label}(s) while building supervised windows.")
		return np.concatenate(x_parts,axis=0),np.concatenate(y_parts,axis=0)
	if len(chunks)==1:values=_reshape_signal_values(np.asarray(chunks[0],dtype=np.float32));raise ValueError(f"Signal {source_label} is too short for seq_len={seq_len} and pred_len={pred_len}: shape={values.shape}")
	raise ValueError(f"No valid supervised windows could be built from the provided {source_label}s. All {len(chunks)} {source_label}s were shorter than seq_len + pred_len.")
def build_supervised_windows_from_matrix(values:np.ndarray,*,seq_len:int,pred_len:int,scaler:object|None,physical_scale_by_column:Sequence[float]|np.ndarray|float|None=None)->tuple[np.ndarray,np.ndarray]:return _build_supervised_windows_from_arrays([values],seq_len=seq_len,pred_len=pred_len,scaler=scaler,physical_scale_by_column=physical_scale_by_column,source_label='matrix')
def _make_tensor_loader(x:np.ndarray,y:np.ndarray,*,batch_size:int,shuffle:bool,device:str|torch.device|TorchRuntimeState='cpu',pin_memory:bool|None=None)->DataLoader:dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(y));resolved_device=resolve_device(device);return DataLoader(dataset,batch_size=min(batch_size,len(dataset)),shuffle=shuffle,pin_memory=resolved_device.type=='cuda'if pin_memory is None else bool(pin_memory))
def train_lstm_model(model,train_loader:DataLoader,val_loader:DataLoader,*,epochs:int,lr:float,device:str|torch.device|TorchRuntimeState,show_progress:bool=False,progress_label:str|None=None)->dict[str,object]:
	runtime_state=configure_torch_runtime(device);resolved_device=runtime_state.device;model=model.to(resolved_device);optimizer,criterion=torch.optim.Adam(model.parameters(),lr=lr),torch.nn.MSELoss();use_amp,grad_scaler=resolved_device.type=='cuda',torch.amp.GradScaler('cuda',enabled=resolved_device.type=='cuda');non_blocking=runtime_state.non_blocking_transfers and use_amp;best_val_loss,best_state_dict=float('inf'),copy.deepcopy(model.state_dict());history={'train_loss':[],'val_loss':[]};progress=tqdm(range(int(epochs)),desc=progress_label or'forecast epochs',leave=False,disable=not show_progress)
	try:
		for _ in progress:
			model.train();train_losses=[]
			for(batch_x,batch_y)in train_loader:
				batch_x,batch_y=batch_x.to(resolved_device,non_blocking=non_blocking),batch_y.to(resolved_device,non_blocking=non_blocking);optimizer.zero_grad(set_to_none=True)
				with torch.autocast(device_type='cuda',dtype=torch.bfloat16)if use_amp else nullcontext():prediction=model(batch_x);loss=criterion(prediction,batch_y)
				if use_amp:grad_scaler.scale(loss).backward();grad_scaler.step(optimizer);grad_scaler.update()
				else:loss.backward();optimizer.step()
				train_losses.append(float(loss.detach().cpu().item()))
			model.eval();val_losses=[]
			with torch.no_grad():
				for(batch_x,batch_y)in val_loader:
					batch_x,batch_y=batch_x.to(resolved_device,non_blocking=non_blocking),batch_y.to(resolved_device,non_blocking=non_blocking)
					with torch.autocast(device_type='cuda',dtype=torch.bfloat16)if use_amp else nullcontext():prediction=model(batch_x);val_loss=criterion(prediction,batch_y)
					val_losses.append(float(val_loss.cpu().item()))
			mean_train=float(np.mean(train_losses))if train_losses else .0;mean_val=float(np.mean(val_losses))if val_losses else mean_train;history['train_loss'].append(mean_train);history['val_loss'].append(mean_val)
			if show_progress:progress.set_postfix({'train':f"{mean_train:.4f}",'val':f"{mean_val:.4f}"})
			if mean_val<best_val_loss:best_val_loss,best_state_dict=mean_val,copy.deepcopy(model.state_dict())
	finally:progress.close()
	model.load_state_dict(best_state_dict);return{'model':model,'history':history,'best_val_loss':float(best_val_loss),'best_state_dict':best_state_dict,'runtime':runtime_state}
def _restore_supervised_window_values(values:np.ndarray,*,scaler:object|None,physical_scale:float|None)->np.ndarray:
	restored=np.asarray(values,dtype=np.float32);original_shape=restored.shape
	if scaler is not None:restored=scaler.inverse_transform(restored.reshape(-1,1)).reshape(original_shape).astype(np.float32)
	if physical_scale is not None:restored=(restored*np.float32(physical_scale)).astype(np.float32)
	return restored.astype(np.float32)
def _predict_supervised_windows(model,windows:np.ndarray,*,runtime_state:TorchRuntimeState,batch_size:int)->np.ndarray:
	predictions:list[np.ndarray]=[];model.eval()
	with torch.no_grad():
		for start in range(0,len(windows),max(int(batch_size),1)):batch=torch.tensor(windows[start:start+max(int(batch_size),1)],dtype=torch.float32,device=runtime_state.device);predictions.append(model(batch).detach().cpu().numpy().astype(np.float32))
	if not predictions:raise ValueError('Validation windows are required to search load blend weights.')
	return np.concatenate(predictions,axis=0).astype(np.float32)
def _build_validation_window_timestamps(segment_frames:Sequence[pd.DataFrame],*,seq_len:int,pred_len:int)->pd.DatetimeIndex:total_window=int(seq_len)+int(pred_len);timestamps=[pd.DatetimeIndex(pd.to_datetime(frame['timestamp'].iloc[int(seq_len):int(seq_len)+int(len(frame)-total_window+1)].tolist(),utc=True))for frame in(segment_frame.reset_index(drop=True)for segment_frame in segment_frames)if len(frame)>=total_window];return pd.DatetimeIndex([])if not timestamps else timestamps[0].append(timestamps[1:])
def _select_week_evaluation_slice(frame:pd.DataFrame,values:np.ndarray,*,seq_len:int)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
	if'week_id'in frame.columns and'is_warmup'in frame.columns and not(non_warmup:=frame.loc[~frame['is_warmup'].astype(bool)]).empty:
		first_target_row=non_warmup.iloc[0];segment_mask=np.ones(len(frame),dtype=bool)if'segment_id'not in frame.columns else frame['segment_id'].astype(int).to_numpy()==int(first_target_row['segment_id']);warmup_mask=segment_mask&frame['is_warmup'].astype(bool).to_numpy();target_mask=segment_mask&~frame['is_warmup'].astype(bool).to_numpy()&(frame['week_id'].astype(int).to_numpy()==int(first_target_row['week_id']));history_values,target_values=values[warmup_mask],values[target_mask]
		if history_values.size==0:history_values=values[segment_mask][:min(seq_len,int(segment_mask.sum()))];history_timestamps=frame.loc[segment_mask].iloc[:len(history_values)]['timestamp'].to_numpy()
		else:history_timestamps=frame.loc[warmup_mask,'timestamp'].to_numpy()
		return history_values,target_values,history_timestamps,frame.loc[target_mask,'timestamp'].to_numpy()
	history=values[:seq_len];target=values[seq_len:seq_len+DEFAULT_WEEK_STEPS]
	if np.asarray(target).size==0:target=values[seq_len:]
	history_timestamps=frame.loc[:len(history)-1,'timestamp'].to_numpy()if'timestamp'in frame.columns else np.arange(len(history));timestamps=frame.loc[seq_len:seq_len+len(target)-1,'timestamp'].to_numpy()if'timestamp'in frame.columns else np.arange(len(target));return history,np.asarray(target),np.asarray(history_timestamps),np.asarray(timestamps)
def compute_forecast_metrics(target:np.ndarray,prediction:np.ndarray)->dict[str,float|None]:
	target_array=np.asarray(target,dtype=np.float32);prediction_array=np.asarray(prediction,dtype=np.float32)
	if target_array.shape!=prediction_array.shape:raise ValueError(f"Forecast metric computation requires target/prediction to share the same shape, got {target_array.shape} vs {prediction_array.shape}.")
	error=prediction_array.reshape(-1)-target_array.reshape(-1);rmse=float(np.sqrt(np.mean(np.square(error))));mae=float(np.mean(np.abs(error)));non_zero_mask=np.abs(target_array.reshape(-1))>1e-08;mape=None if not np.any(non_zero_mask)else float(np.mean(np.abs(error[non_zero_mask]/target_array.reshape(-1)[non_zero_mask]))*1e2);return{'rmse':rmse,'mae':mae,'mape':mape}
def _select_load_blend_weight_from_validation(*,model,x_val:np.ndarray,y_val:np.ndarray,scaler:object|None,physical_scale:float|None,runtime_state:TorchRuntimeState,batch_size:int,candidate_weights:Sequence[float],component:str|None=None,window_timestamps:pd.DatetimeIndex|None=None)->dict[str,object]:
	raw_predictions=_predict_supervised_windows(model,x_val,runtime_state=runtime_state,batch_size=batch_size);raw_step1=_restore_supervised_window_values(raw_predictions[:,0],scaler=scaler,physical_scale=physical_scale);baseline_step1=_restore_supervised_window_values(x_val[:,-1,0],scaler=scaler,physical_scale=physical_scale);target_step1=_restore_supervised_window_values(y_val[:,0],scaler=scaler,physical_scale=physical_scale);best_weight,best_mae,optimized_metric,bias_guard_satisfied=float(candidate_weights[0]),float('inf'),'mae_step1',None
	if str(component or'').strip().lower()=='heatpump'and window_timestamps is not None and len(window_timestamps)==len(target_step1):selection=_select_heatpump_blocked_blend_weight(target_step1=target_step1,baseline_step1=baseline_step1,raw_step1=raw_step1,window_timestamps=pd.DatetimeIndex(window_timestamps),candidate_weights=tuple(float(weight)for weight in candidate_weights));best_weight,optimized_metric,bias_guard_satisfied=float(selection['selected_weight']),'blocked_bias_guard_step1',bool(selection['bias_guard_satisfied'])
	else:
		for candidate_weight in candidate_weights:
			weight=np.float32(candidate_weight);blended_step1=baseline_step1+weight*(raw_step1-baseline_step1);mae=float(np.mean(np.abs(blended_step1-target_step1)))
			if mae<best_mae:best_mae,best_weight=mae,float(candidate_weight)
	blended_step1=baseline_step1+np.float32(best_weight)*(raw_step1-baseline_step1);return{'postprocess_mode':POSTPROCESS_MODE_BASELINE_BLEND,'baseline_mode':BASELINE_MODE_LAST_VALUE,'blend_weight':best_weight,'optimized_metric':optimized_metric,'bias_guard_satisfied':bias_guard_satisfied}
def _select_heatpump_blocked_blend_weight(*,target_step1:np.ndarray,baseline_step1:np.ndarray,raw_step1:np.ndarray,window_timestamps:pd.DatetimeIndex,candidate_weights:Sequence[float],bias_threshold_kw:float=HEATPUMP_BLOCKED_BIAS_GUARD_KW)->dict[str,object]:
	if len(window_timestamps)!=len(target_step1):raise ValueError(f"window_timestamps must align with target_step1 for heatpump blocked blend selection, got len(window_timestamps)={len(window_timestamps)} vs len(target_step1)={len(target_step1)}.")
	base_frame=pd.DataFrame({'timestamp':pd.DatetimeIndex(window_timestamps),'target':np.asarray(target_step1,dtype=np.float32),'baseline':np.asarray(baseline_step1,dtype=np.float32),'raw':np.asarray(raw_step1,dtype=np.float32)});base_frame['month']=base_frame['timestamp'].dt.month;candidate_rows:list[dict[str,object]]=[]
	for candidate_weight in candidate_weights:
		block_rmses:list[float]=[];block_maes:list[float]=[];block_abs_biases:list[float]=[];max_abs_biases:list[float]=[];passes_bias_guard=True
		for(_,months)in HEATPUMP_BLOCKED_MONTH_GROUPS:
			block=base_frame.loc[base_frame['month'].isin(months)]
			if block.empty:continue
			blended=block['baseline'].to_numpy(dtype=np.float32)+np.float32(candidate_weight)*(block['raw'].to_numpy(dtype=np.float32)-block['baseline'].to_numpy(dtype=np.float32));target=block['target'].to_numpy(dtype=np.float32);metrics=compute_forecast_metrics(target,blended);abs_bias=abs(float(np.mean(blended-target)));passes_bias_guard=passes_bias_guard and abs_bias<=float(bias_threshold_kw);block_rmses.append(float(metrics['rmse']));block_maes.append(float(metrics['mae']));block_abs_biases.append(abs_bias);max_abs_biases.append(abs_bias)
		if not block_rmses:continue
		candidate_rows.append({'candidate_weight':float(candidate_weight),'avg_block_rmse':float(np.mean(block_rmses)),'avg_block_mae':float(np.mean(block_maes)),'avg_abs_block_bias':float(np.mean(block_abs_biases)),'max_abs_block_bias':float(np.max(max_abs_biases)),'passes_bias_guard':bool(passes_bias_guard)})
	candidate_frame=pd.DataFrame(candidate_rows)
	if candidate_frame.empty:raise ValueError('No blocked-validation candidates were produced for heatpump blend selection.')
	viable=candidate_frame.loc[candidate_frame['passes_bias_guard']].copy();ranked,satisfied=(candidate_frame.sort_values(['max_abs_block_bias','avg_abs_block_bias','avg_block_mae','candidate_weight']).reset_index(drop=True),False)if viable.empty else(viable.sort_values(['avg_block_mae','avg_abs_block_bias','candidate_weight']).reset_index(drop=True),True);selected=ranked.iloc[0];return{'selected_weight':float(selected['candidate_weight']),'bias_guard_satisfied':bool(satisfied)}
def evaluate_signal_one_week(source:SignalCsvSource,model,*,mode:str,**kwargs):
	signal_name,scaler=kwargs['signal_name'],kwargs['scaler'];test_frame,test_values,_=_load_signal_matrix_from_source(source,signal_name,split='test');history_seed,target_values,history_timestamps,timestamps=_select_week_evaluation_slice(test_frame,test_values,seq_len=kwargs['seq_len']);forecaster=LSTMForecaster(model_path=None,hidden_size=kwargs['hidden_size'],num_layers=kwargs['num_layers'],dropout=kwargs['dropout'],pred_len=kwargs['pred_len'],seq_len=kwargs['seq_len'],device=kwargs['device'],scaler=scaler,input_size=kwargs.get('input_size',1),time_feature_mode=kwargs.get('time_feature_mode',TIME_FEATURE_MODE_NONE),model_mode=kwargs.get('model_mode','shared'),physical_normalization_mode=kwargs.get('physical_normalization_mode',PHYSICAL_NORMALIZATION_NONE),physical_scale_by_column=kwargs.get('physical_scale_by_column'),postprocess_mode=kwargs.get('postprocess_mode',POSTPROCESS_MODE_NONE),baseline_mode=kwargs.get('baseline_mode',BASELINE_MODE_NONE),blend_weight=kwargs.get('blend_weight'),optimized_metric=kwargs.get('optimized_metric')).rename_default_signal(signal_name);runtime=forecaster.signal_runtimes[signal_name][0];runtime.model.load_state_dict(copy.deepcopy(model.state_dict()));runtime.model.eval();runtime.agent_index,runtime.agent_profile=None if kwargs.get('agent_index')is None else int(kwargs['agent_index']),None if kwargs.get('agent_profile')is None else str(kwargs['agent_profile']);target_matrix=_reshape_signal_values(target_values);target=target_values.reshape(-1).astype(np.float32)if target_matrix.shape[1]==1 else target_matrix.T.astype(np.float32)
	if mode=='open_loop':rollout=forecaster.predict(history_seed,horizon=target_values.shape[0]+1,signal_name=signal_name,history_timestamps=history_timestamps);prediction=np.asarray(rollout,dtype=np.float32)[1:target_values.shape[0]+1]if target_matrix.shape[1]==1 else np.asarray(rollout,dtype=np.float32)[:,1:target_values.shape[0]+1]
	elif mode=='online_aligned':
		history_matrix,one_step_predictions=_reshape_signal_values(history_seed),[]
		for step_idx in range(target_matrix.shape[0]):current_history=history_matrix if step_idx==0 else np.concatenate([history_matrix,target_matrix[:step_idx]],axis=0);current_timestamps=np.asarray(history_timestamps)if step_idx==0 else np.concatenate([np.asarray(history_timestamps),np.asarray(timestamps[:step_idx])],axis=0);rollout=forecaster.predict(current_history.reshape(-1)if current_history.ndim==2 and current_history.shape[1]==1 else current_history,horizon=max(int(kwargs['pred_len'])+1,2),signal_name=signal_name,history_timestamps=current_timestamps);rollout_array=np.asarray(rollout,dtype=np.float32);one_step_predictions.append(np.float32(rollout_array[1])if target_matrix.shape[1]==1 else rollout_array[:,1].astype(np.float32))
		prediction=np.asarray(one_step_predictions,dtype=np.float32)if target_matrix.shape[1]==1 else np.stack(one_step_predictions,axis=0).T.astype(np.float32)
	else:raise ValueError("Unknown evaluation mode, expected 'online_aligned' or 'open_loop'.")
	prediction=np.asarray(prediction,dtype=np.float32);return SignalForecastEvaluation(signal_name=signal_name,evaluation_mode=mode,timestamps=np.asarray(timestamps),target=target,prediction=prediction,metrics=compute_forecast_metrics(target,prediction),source_columns=source.value_columns)
def _extract_segment_value_arrays(segment_frames:list[pd.DataFrame],value_columns:Sequence[str])->list[np.ndarray]:return[values.reshape(-1)if values.ndim==2 and values.shape[1]==1 else values for segment_frame in segment_frames for values in[segment_frame.loc[:,list(value_columns)].to_numpy(dtype=np.float32)]]
def _aggregate_per_agent_evaluations(signal_name:str,evaluation_mode:str,evaluations:Sequence[SignalForecastEvaluation])->SignalForecastEvaluation:
	if not evaluations:raise ValueError('Per-agent evaluation aggregation requires at least one evaluation.')
	base_timestamps=np.asarray(evaluations[0].timestamps);target_rows:list[np.ndarray]=[];prediction_rows:list[np.ndarray]=[];source_columns:list[str]=[]
	for evaluation in evaluations:
		evaluation_timestamps=np.asarray(evaluation.timestamps)
		if evaluation_timestamps.shape!=base_timestamps.shape or not np.array_equal(evaluation_timestamps,base_timestamps):raise ValueError('Per-agent evaluation timestamps must align before aggregation.')
		target_array=np.asarray(evaluation.target,dtype=np.float32);prediction_array=np.asarray(evaluation.prediction,dtype=np.float32)
		if target_array.ndim==1:target_row,prediction_row=target_array,prediction_array
		elif target_array.ndim==2 and target_array.shape[0]==1:target_row,prediction_row=target_array.reshape(-1),prediction_array.reshape(-1)
		else:raise ValueError('Per-agent aggregation expects 1D evaluations or 2D evaluations with one source column.')
		target_rows.append(target_row);prediction_rows.append(prediction_row);source_columns.extend(str(column)for column in evaluation.source_columns)
	target=np.stack(target_rows,axis=0).astype(np.float32);prediction=np.stack(prediction_rows,axis=0).astype(np.float32);return SignalForecastEvaluation(signal_name=signal_name,evaluation_mode=evaluation_mode,timestamps=base_timestamps,target=target,prediction=prediction,metrics=compute_forecast_metrics(target,prediction),source_columns=tuple(source_columns))
def _aggregate_training_history(agent_results:Sequence[dict[str,object]])->dict[str,list[float]]:
	if not agent_results:return{'train_loss':[],'val_loss':[]}
	train_curves=[np.asarray(result['training']['history']['train_loss'],dtype=np.float32)for result in agent_results];val_curves=[np.asarray(result['training']['history']['val_loss'],dtype=np.float32)for result in agent_results];return{'train_loss':np.mean(np.stack(train_curves,axis=0),axis=0).astype(np.float32).tolist(),'val_loss':np.mean(np.stack(val_curves,axis=0),axis=0).astype(np.float32).tolist()}
def _summarize_per_agent_train_stats(agent_results:Sequence[dict[str,object]])->dict[str,object]:stats_by_agent={str(result.get('agent_profile',result['columns'][0])):dict(result['train_stats'])for result in agent_results};return{'agent_stats':stats_by_agent,'min':float(min(stat['min']for stat in stats_by_agent.values())),'max':float(max(stat['max']for stat in stats_by_agent.values())),'mean':float(np.mean([stat['mean']for stat in stats_by_agent.values()])),'std':float(np.mean([stat['std']for stat in stats_by_agent.values()]))}
def _print_signal_evaluation_summary(signal_name:str,online_evaluation:SignalForecastEvaluation,open_loop_evaluation:SignalForecastEvaluation)->None:print(f"[forecast] {signal_name} metrics: online_rmse={online_evaluation.metrics['rmse']:.6f}, online_mae={online_evaluation.metrics['mae']:.6f}, open_loop_rmse={open_loop_evaluation.metrics['rmse']:.6f}, open_loop_mae={open_loop_evaluation.metrics['mae']:.6f}")
def _build_signal_training_specs(local_cfg,signal_name:str,*,source:SignalCsvSource,settings:dict[str,object],data_dir:Path)->list[dict[str,object]]:
	if signal_name!='load'or str(settings['model_mode'])!='per_agent':return[{'source':source,'agent_index':None,'agent_profile':None,'component':None}]
	profiles=[str(profile)for profile in local_cfg.data.agent_profiles][:len(source.value_columns)];components=list(local_cfg.data.load_components)if bool(settings.get('component_split',False))else[None];specs:list[dict[str,object]]=[]
	for(agent_index,agent_profile)in enumerate(profiles):
		for component in components:
			comp_signal=None if component is None else f"load_{component}";selected_source=source if comp_signal is None else resolve_signal_csv_source(data_dir,comp_signal,cfg=local_cfg)
			if selected_source is None:raise FileNotFoundError(f"No source for component signal '{comp_signal}'. Ensure ProsumerDataset exposes load_{component}.")
			specs.append({'source':select_signal_source_columns(selected_source,[agent_index]),'agent_index':agent_index,'agent_profile':agent_profile,'component':component})
	return specs
def _prepare_signal_training_data(local_cfg,signal_name:str,*,source:SignalCsvSource,settings:dict[str,object],train_physical_scale:np.ndarray|None,runtime_state:TorchRuntimeState)->dict[str,object]:
	seq_len,pred_len,batch_size=int(local_cfg.forecast.history_window),int(local_cfg.env.future_horizon),int(local_cfg.forecast.lstm_batch_size);_,train_segment_frames,value_columns=_load_signal_segment_frames_from_source(source,signal_name,split='train',drop_warmup=True);train_ratio,val_ratio=float(local_cfg.forecast.lstm_train_ratio),float(local_cfg.forecast.lstm_val_ratio)
	if int(settings['input_size'])>1:split_source=train_segment_frames;split=_temporal_split_items(split_source,train_ratio=train_ratio,val_ratio=val_ratio,seq_len=seq_len,pred_len=pred_len);train_values=_extract_segment_value_arrays(split['train_segments'],value_columns);scaler=fit_signal_scaler(train_values,physical_scale_by_column=train_physical_scale,scaler_type=str(settings.get('scaler_type','standard')));builder=build_supervised_windows_from_time_feature_segments;x_train,y_train=builder(split['train_segments'],value_columns=value_columns,seq_len=seq_len,pred_len=pred_len,scaler=scaler,physical_scale_by_column=train_physical_scale,time_feature_mode=str(settings['time_feature_mode']));x_val,y_val=builder(split['val_segments'],value_columns=value_columns,seq_len=seq_len,pred_len=pred_len,scaler=scaler,physical_scale_by_column=train_physical_scale,time_feature_mode=str(settings['time_feature_mode']));val_window_timestamps=_build_validation_window_timestamps(split['val_segments'],seq_len=seq_len,pred_len=pred_len)
	else:
		segment_values=_extract_segment_value_arrays(train_segment_frames,value_columns)
		if not segment_values:raise ValueError(f"Signal source split='train' does not contain any segments for signal '{signal_name}'.")
		split=_temporal_split_items([_reshape_signal_values(segment)for segment in segment_values],train_ratio=train_ratio,val_ratio=val_ratio,seq_len=seq_len,pred_len=pred_len);train_values=split['train_segments'];scaler=fit_signal_scaler(train_values,physical_scale_by_column=train_physical_scale,scaler_type=str(settings.get('scaler_type','standard')));builder=_build_supervised_windows_from_arrays;x_train,y_train=builder(split['train_segments'],seq_len=seq_len,pred_len=pred_len,scaler=scaler,physical_scale_by_column=train_physical_scale);x_val,y_val=builder(split['val_segments'],seq_len=seq_len,pred_len=pred_len,scaler=scaler,physical_scale_by_column=train_physical_scale);val_window_timestamps=None
	return{'split':split,'value_columns':value_columns,'train_stats':summarize_signal_values(train_values),'scaler':scaler,'x_val':x_val,'y_val':y_val,'val_window_timestamps':val_window_timestamps,'train_loader':_make_tensor_loader(x_train,y_train,batch_size=batch_size,shuffle=True,device=runtime_state.device,pin_memory=runtime_state.pin_memory),'val_loader':_make_tensor_loader(x_val,y_val,batch_size=batch_size,shuffle=False,device=runtime_state.device,pin_memory=runtime_state.pin_memory)}
def _train_single_signal_lstm(local_cfg,signal_name:str,*,source:SignalCsvSource,settings:dict[str,object],runtime_state:TorchRuntimeState,overrides:dict[str,object]|None=None,show_progress:bool=False,compute_evaluations:bool=True,agent_index:int|None=None,agent_profile:str|None=None,component:str|None=None)->dict[str,object]:
	seq_len,pred_len=int(local_cfg.forecast.history_window),int(local_cfg.env.future_horizon);physical_normalization_mode=resolve_signal_physical_normalization_mode(signal_name);train_physical_scale,test_physical_scale=resolve_signal_physical_scale_from_source(source,signal_name,split='train'),resolve_signal_physical_scale_from_source(source,signal_name,split='test');prepared=_prepare_signal_training_data(local_cfg,signal_name,source=source,settings=settings,train_physical_scale=train_physical_scale,runtime_state=runtime_state);split,value_columns,train_stats,scaler=prepared['split'],prepared['value_columns'],prepared['train_stats'],prepared['scaler'];x_val,y_val,val_window_timestamps=prepared['x_val'],prepared['y_val'],prepared['val_window_timestamps'];train_loader,val_loader=prepared['train_loader'],prepared['val_loader'];progress_name=f"{signal_name if agent_profile is None else f'{signal_name}[{agent_profile}]'}{''if component is None else f'/{component}'}";print(f"[forecast] {progress_name}: train_segments={len(split['train_segments'])}, val_segments={len(split['val_segments'])}, columns={list(value_columns)}, train_stats={train_stats}, settings={settings}");model=LSTMForecastModel(hidden_size=int(local_cfg.forecast.lstm_hidden_size),num_layers=int(local_cfg.forecast.lstm_num_layers),dropout=float(local_cfg.forecast.lstm_dropout),pred_len=pred_len,input_size=int(settings['input_size']));result=train_lstm_model(model,train_loader=train_loader,val_loader=val_loader,epochs=int(local_cfg.forecast.lstm_epochs),lr=float(local_cfg.forecast.lstm_lr),device=runtime_state,show_progress=show_progress,progress_label=f"{progress_name} epochs");hybrid_config={'postprocess_mode':str(settings['postprocess_mode']),'baseline_mode':str(settings['baseline_mode']),'blend_weight':None,'optimized_metric':None}
	if _normalize_signal_name(signal_name)=='load'and str(settings['postprocess_mode'])==POSTPROCESS_MODE_BASELINE_BLEND:
		if x_val is None or y_val is None:raise ValueError('Load hybrid validation requires supervised validation windows.')
		physical_scale=None if train_physical_scale is None else float(np.asarray(train_physical_scale,dtype=np.float32).reshape(-1)[0]);hybrid_config=_select_load_blend_weight_from_validation(model=result['model'],x_val=x_val,y_val=y_val,scaler=scaler,physical_scale=physical_scale,runtime_state=runtime_state,batch_size=int(local_cfg.forecast.lstm_batch_size),candidate_weights=settings['blend_candidates'],component=component,window_timestamps=val_window_timestamps)
	artifact_paths=_managed_lstm_artifact_paths(local_cfg,signal_name,agent_index=agent_index if str(settings['model_mode'])=='per_agent'else None,component=component);artifact_paths['artifact_dir'].mkdir(parents=True,exist_ok=True);saved_paths=save_lstm_forecaster_artifacts(model_path=artifact_paths['model_path'],state_dict=result['best_state_dict'],scaler=scaler,seq_len=seq_len,pred_len=pred_len,hidden_size=int(local_cfg.forecast.lstm_hidden_size),num_layers=int(local_cfg.forecast.lstm_num_layers),dropout=float(local_cfg.forecast.lstm_dropout),signal_name=signal_name,future_horizon=pred_len,artifact_format=str(settings['artifact_format']),normalization_mode=physical_normalization_mode,source_signature=build_lstm_source_signature(local_cfg,signal_name),input_size=int(settings['input_size']),time_feature_mode=str(settings['time_feature_mode']),model_mode=str(settings['model_mode']),agent_index=agent_index,agent_profile=agent_profile,postprocess_mode=str(hybrid_config['postprocess_mode']),baseline_mode=str(hybrid_config['baseline_mode']),blend_weight=hybrid_config['blend_weight'],optimized_metric=hybrid_config['optimized_metric'],component=component,scaler_type=str(settings.get('scaler_type','standard')));refreshed_validation=validate_lstm_artifact(local_cfg,signal_name,artifact_paths,overrides=overrides,agent_index=agent_index,agent_profile=agent_profile,component=component)
	if not refreshed_validation['compatible']:raise RuntimeError(f"Saved LSTM artifact failed validation after training: {_format_lstm_artifact_issue(refreshed_validation)}")
	print(f"[forecast] artifact refreshed at {saved_paths['model_path']}");online_evaluation,open_loop_evaluation=None,None
	if compute_evaluations:
		evaluation_kwargs={'signal_name':signal_name,'scaler':scaler,'seq_len':seq_len,'pred_len':pred_len,'hidden_size':int(local_cfg.forecast.lstm_hidden_size),'num_layers':int(local_cfg.forecast.lstm_num_layers),'dropout':float(local_cfg.forecast.lstm_dropout),'device':runtime_state.device,'input_size':int(settings['input_size']),'time_feature_mode':str(settings['time_feature_mode']),'model_mode':str(settings['model_mode']),'physical_normalization_mode':physical_normalization_mode,'physical_scale_by_column':test_physical_scale,'agent_index':agent_index,'agent_profile':agent_profile,'postprocess_mode':str(hybrid_config['postprocess_mode']),'baseline_mode':str(hybrid_config['baseline_mode']),'blend_weight':hybrid_config['blend_weight'],'optimized_metric':hybrid_config['optimized_metric']};online_evaluation,open_loop_evaluation=evaluate_signal_one_week(source,result['model'],mode='online_aligned',**evaluation_kwargs),evaluate_signal_one_week(source,result['model'],mode='open_loop',**evaluation_kwargs);_print_signal_evaluation_summary(progress_name,online_evaluation,open_loop_evaluation)
	return{'signal_name':signal_name,'source':source,'columns':value_columns,'segment_summary':split['segment_summaries'],'train_stats':train_stats,'artifact_paths':saved_paths,'training':result,'evaluation':online_evaluation,'online_evaluation':online_evaluation,'open_loop_evaluation':open_loop_evaluation,'settings':settings,'runtime':runtime_state,'agent_index':agent_index,'agent_profile':agent_profile,'component':component,'hybrid':hybrid_config}
def train_signal_lstm(cfg,signal_name:str,*,device:str|torch.device|TorchRuntimeState|None=None,overrides:dict[str,object]|None=None,show_progress:bool=False,compute_evaluations:bool=True)->dict[str,object]:
	local_cfg,settings=resolve_signal_training_settings(cfg,signal_name,overrides=overrides);data_dir=Path(local_cfg.data.data_dir or Path(__file__).resolve().parent.parent/'data');runtime_state=configure_torch_runtime(local_cfg,device=local_cfg.runtime.device if device is None else device,seed=local_cfg.runtime.seed);signal_name=_normalize_signal_name(signal_name);source=resolve_signal_csv_source(data_dir,signal_name,cfg=local_cfg)
	if source is None:raise FileNotFoundError(f"No train/test source found for forecast signal '{signal_name}' under '{data_dir}'.")
	if int(local_cfg.env.future_horizon)<=0:raise ValueError('LSTM forecast training requires cfg.env.future_horizon > 0.')
	specs=_build_signal_training_specs(local_cfg,signal_name,source=source,settings=settings,data_dir=data_dir);results=[_train_single_signal_lstm(local_cfg,signal_name,settings=settings,runtime_state=runtime_state,overrides=overrides,show_progress=show_progress,compute_evaluations=compute_evaluations,**spec)for spec in specs]
	if len(results)==1 and results[0].get('agent_index')is None and results[0].get('component')is None:return results[0]
	online_evaluation=None if not compute_evaluations else _aggregate_per_agent_evaluations(signal_name,'online_aligned',[result['online_evaluation']for result in results]);open_loop_evaluation=None if not compute_evaluations else _aggregate_per_agent_evaluations(signal_name,'open_loop',[result['open_loop_evaluation']for result in results]);return{'signal_name':signal_name,'source':source,'columns':tuple(source.value_columns),'segment_summary':{str(result['agent_profile']):result['segment_summary']for result in results},'train_stats':_summarize_per_agent_train_stats(results),'artifact_paths':[result['artifact_paths']for result in results],'training':{'history':_aggregate_training_history(results),'best_val_loss':float(np.mean([result['training']['best_val_loss']for result in results],dtype=np.float32))},'evaluation':online_evaluation,'online_evaluation':online_evaluation,'open_loop_evaluation':open_loop_evaluation,'settings':settings,'runtime':runtime_state,'agent_results':results,'hybrid':{str(result['agent_profile']):dict(result['hybrid'])for result in results}}
def _expected_signal_optimized_metric(signal_name:str,*,postprocess_mode:str,component:str|None=None)->str|None:normalized_signal=_normalize_signal_name(signal_name);return None if normalized_signal!='load'or str(postprocess_mode)!=POSTPROCESS_MODE_BASELINE_BLEND else'blocked_bias_guard_step1'if str(component or'').strip().lower()=='heatpump'else'mae_step1'
def expected_lstm_artifact_meta(cfg,signal_name:str,overrides:dict[str,object]|None=None,*,agent_index:int|None=None,agent_profile:str|None=None,component:str|None=None)->dict[str,object]:
	local_cfg,settings=resolve_signal_training_settings(cfg,signal_name,overrides=overrides);future_horizon=int(settings['future_horizon']);expected={'artifact_format':str(settings['artifact_format']),'signal_name':str(settings['signal_name']),'future_horizon':future_horizon,'pred_len':future_horizon,'seq_len':int(settings['history_window']),'hidden_size':int(settings['hidden_size']),'num_layers':int(settings['num_layers']),'dropout':float(settings['dropout']),'input_size':int(settings['input_size']),'time_feature_mode':str(settings['time_feature_mode']),'model_mode':str(settings['model_mode']),'normalization_mode':resolve_signal_physical_normalization_mode(signal_name),'source_signature':build_lstm_source_signature(local_cfg,signal_name),'agent_index':None if agent_index is None else int(agent_index),'agent_profile':None if agent_profile is None else str(agent_profile),'component':None if component is None else str(component)}
	if expected['artifact_format']==LSTM_LOAD_HYBRID_ARTIFACT_FORMAT or str(settings['postprocess_mode'])!=POSTPROCESS_MODE_NONE:expected.update({'postprocess_mode':str(settings['postprocess_mode']),'baseline_mode':str(settings['baseline_mode']),'blend_weight':None,'optimized_metric':_expected_signal_optimized_metric(signal_name,postprocess_mode=str(settings['postprocess_mode']),component=component)})
	return expected
def compare_lstm_artifact_meta(actual_meta:dict[str,object]|None,expected_meta:dict[str,object])->dict[str,object]:
	actual_meta=dict(actual_meta or{});comparable_fields=tuple(expected_meta.keys());actual,expected,mismatches={field:actual_meta.get(field)for field in comparable_fields},{field:expected_meta.get(field)for field in comparable_fields},{}
	for field in comparable_fields:
		expected_value,actual_value=expected.get(field),actual.get(field);matches=actual_value==expected_value if field not in{'dropout','blend_weight'}else actual_value is not None and bool(np.isclose(float(actual_value),float(expected_value)))if field=='dropout'else field in actual_meta and(actual_value is None or .0<=float(actual_value)<=1.)
		if not matches:mismatches[field]={'expected':expected_value,'actual':actual_value}
	return{'compatible':not mismatches,'expected':expected,'actual':actual,'mismatches':mismatches}
def validate_lstm_artifact(cfg,signal_name:str,paths:dict[str,str|Path],*,overrides:dict[str,object]|None=None,agent_index:int|None=None,agent_profile:str|None=None,component:str|None=None)->dict[str,object]:
	normalized_signal=_normalize_signal_name(signal_name);expected=expected_lstm_artifact_meta(cfg,normalized_signal,overrides=overrides,agent_index=agent_index,agent_profile=agent_profile,component=component);resolved_paths={name:Path(path)for(name,path)in paths.items()};required_files={key:resolved_paths[key]for key in('model_path','meta_path','scaler_path')if key in resolved_paths};existing_files={key:path.exists()for(key,path)in required_files.items()};missing_files=[key for(key,exists)in existing_files.items()if not exists];result:dict[str,object]={'signal_name':normalized_signal,'artifact_path':str(resolved_paths.get('model_path','')),'paths':{key:str(path)for(key,path)in required_files.items()},'expected':expected,'actual':{},'mismatches':{},'missing_files':missing_files,'compatible':False,'issue_type':None,'agent_index':None if agent_index is None else int(agent_index),'agent_profile':None if agent_profile is None else str(agent_profile)}
	if missing_files:result['issue_type']='incomplete'if any(existing_files.values())else'missing';return result
	try:actual_meta=json.loads(required_files['meta_path'].read_text(encoding='utf-8'))
	except(OSError,json.JSONDecodeError)as exc:result['issue_type'],result['error']='invalid_meta',f"Failed to read meta: {exc}";return result
	comparison=compare_lstm_artifact_meta(actual_meta,expected);result.update(comparison);result['issue_type']=None if comparison['compatible']else'mismatch';return result
def _managed_lstm_artifact_paths(cfg,signal_name:str,*,agent_index:int|None=None,component:str|None=None)->dict[str,Path]:artifact_root=cfg.forecast.lstm_artifact_root;return get_default_lstm_artifact_paths(root=get_default_lstm_artifact_dir()if artifact_root in(None,'')else Path(artifact_root),signal_name=_normalize_signal_name(signal_name),future_horizon=int(cfg.env.future_horizon),agent_index=agent_index,component=component)
def _format_lstm_artifact_issue(validation:dict[str,object])->str:
	signal_name,artifact_path,issue_type=validation['signal_name'],validation['artifact_path'],validation.get('issue_type')
	if issue_type in{'missing','incomplete'}:return f"signal='{signal_name}', artifact='{artifact_path}', {'artifact is incomplete'if issue_type=='incomplete'else'missing files'}: {', '.join(validation.get('missing_files',[]))}. Delete stale artifacts or enable cfg.forecast.auto_train_missing=True to rebuild them automatically."
	if issue_type=='invalid_meta':return f"signal='{signal_name}', artifact='{artifact_path}', {validation.get('error','meta is unreadable')}. Delete the stale artifact or enable cfg.forecast.auto_train_missing=True to rebuild it automatically."
	mismatch_text=', '.join(f"{field}(expected={detail['expected']}, actual={detail['actual']})"for(field,detail)in sorted(validation.get('mismatches',{}).items()));return f"signal='{signal_name}', artifact='{artifact_path}', config mismatch: {mismatch_text}. Delete the stale artifact or enable cfg.forecast.auto_train_missing=True to rebuild it automatically."
def _signal_artifact_specs(cfg,signal_name:str)->list[dict[str,object]]:
	normalized_signal=_normalize_signal_name(signal_name);behavior=_resolve_signal_behavior(cfg,normalized_signal)
	if normalized_signal=='load'and str(behavior['model_mode'])=='per_agent':profiles=[str(profile)for profile in cfg.data.agent_profiles][:int(cfg.env.num_agents)];components=list(cfg.data.load_components)if bool(behavior['component_split'])else[None];return[{'signal_name':normalized_signal,'agent_index':int(agent_index),'agent_profile':profile,'component':component,'paths':_managed_lstm_artifact_paths(cfg,normalized_signal,agent_index=agent_index,component=component)}for(agent_index,profile)in enumerate(profiles)for component in components]
	return[{'signal_name':normalized_signal,'agent_index':None,'agent_profile':None,'paths':_managed_lstm_artifact_paths(cfg,normalized_signal)}]
def _collect_lstm_artifact_inventory(cfg,*,overrides_by_signal:dict[str,dict[str,object]]|None=None)->dict[str,object]:
	resolved_overrides_by_signal=dict(getattr(cfg.forecast,'signal_training_overrides',{})if overrides_by_signal is None else overrides_by_signal);artifacts:dict[str,object]={};missing_artifacts:dict[str,list[dict[str,object]]]={};invalid_artifacts:dict[str,list[dict[str,object]]]={}
	for signal_name in configured_forecast_signals(cfg):
		compatible_artifacts,signal_missing,signal_invalid=[],[],[];signal_overrides=dict(resolved_overrides_by_signal.get(signal_name)or{})
		for spec in _signal_artifact_specs(cfg,signal_name):
			validation=validate_lstm_artifact(cfg,signal_name,spec['paths'],overrides=signal_overrides or None,agent_index=spec['agent_index'],agent_profile=spec['agent_profile'],component=spec.get('component'))
			if validation['compatible']:compatible_artifacts.append((str(spec['paths']['model_path']),str(spec['paths']['meta_path']),str(spec['paths']['scaler_path'])))
			elif validation.get('issue_type')=='missing':signal_missing.append(validation)
			else:signal_invalid.append(validation)
		if compatible_artifacts and not signal_missing and not signal_invalid:artifacts[signal_name]=compatible_artifacts if len(compatible_artifacts)>1 else compatible_artifacts[0]
		if signal_missing:missing_artifacts[signal_name]=signal_missing
		if signal_invalid:invalid_artifacts[signal_name]=signal_invalid
	return{'artifacts':artifacts,'missing_artifacts':missing_artifacts,'missing_signals':sorted(missing_artifacts),'invalid_artifacts':invalid_artifacts,'mismatched_signals':sorted(invalid_artifacts)}
def _raise_lstm_artifact_requirements_error(cfg,inventory:dict[str,object])->None:
	problem_lines=[f"- {_format_lstm_artifact_issue(validation)}"for validations in inventory.get('invalid_artifacts',{}).values()for validation in validations]+[f"- {_format_lstm_artifact_issue(validation)}"for validations in inventory.get('missing_artifacts',{}).values()for validation in validations]
	if not problem_lines:return
	mismatch_count=sum(len(validations)for validations in inventory.get('invalid_artifacts',{}).values());missing_count=sum(len(validations)for validations in inventory.get('missing_artifacts',{}).values());raise(RuntimeError if mismatch_count and missing_count else ValueError if mismatch_count else FileNotFoundError)('Managed LSTM forecast artifacts are missing or incompatible:\n'+'\n'.join(problem_lines))
def collect_available_lstm_artifacts(cfg,*,overrides_by_signal:dict[str,dict[str,object]]|None=None)->dict[str,object]:return dict(_collect_lstm_artifact_inventory(cfg,overrides_by_signal=getattr(cfg.forecast,'signal_training_overrides',{})if overrides_by_signal is None else overrides_by_signal)['artifacts'])
def ensure_lstm_artifacts(cfg,*,device:str|torch.device|None=None,overrides_by_signal:dict[str,dict[str,object]]|None=None)->dict[str,object]:
	artifact_root=Path(cfg.forecast.lstm_artifact_root)if cfg.forecast.lstm_artifact_root not in(None,'')else get_default_lstm_artifact_dir();artifact_root.mkdir(parents=True,exist_ok=True);resolved_overrides_by_signal=getattr(cfg.forecast,'signal_training_overrides',{})if overrides_by_signal is None else overrides_by_signal;inventory_before=_collect_lstm_artifact_inventory(cfg,overrides_by_signal=resolved_overrides_by_signal);auto_train_missing=bool(cfg.forecast.auto_train_missing)
	if not auto_train_missing:
		if inventory_before['missing_signals']or inventory_before['invalid_artifacts']:_raise_lstm_artifact_requirements_error(cfg,inventory_before)
		return{'mode':'managed_multi_signal','artifacts':dict(inventory_before['artifacts']),'trained_signals':[],'retrained_signals':[],'mismatched_signals':list(inventory_before['mismatched_signals']),'invalid_artifacts':dict(inventory_before['invalid_artifacts']),'missing_signals':list(inventory_before['missing_signals'])}
	trained_signals,retrained_signals=[],[];data_dir=Path(cfg.data.data_dir or Path(__file__).resolve().parent.parent/'data')
	for signal_name in configured_forecast_signals(cfg):
		if signal_name in inventory_before['artifacts']:continue
		invalid_validations=inventory_before['invalid_artifacts'].get(signal_name,[])
		for invalid_validation in invalid_validations:print(f"[forecast] artifact mismatch detected: {_format_lstm_artifact_issue(invalid_validation)}")
		if resolve_signal_csv_source(data_dir,signal_name,cfg=cfg)is None:print(f"[forecast] skip '{signal_name}': no matching train/test source found.");continue
		print(f"[forecast] retraining signal={signal_name}");train_signal_lstm(cfg,signal_name,device=device,overrides=dict(resolved_overrides_by_signal.get(signal_name)or{})or None);(retrained_signals if invalid_validations else trained_signals).append(signal_name)
	inventory_after=_collect_lstm_artifact_inventory(cfg,overrides_by_signal=resolved_overrides_by_signal)
	return{'mode':'managed_multi_signal','artifacts':dict(inventory_after['artifacts']),'trained_signals':trained_signals,'retrained_signals':retrained_signals,'mismatched_signals':list(inventory_before['mismatched_signals']),'invalid_artifacts':dict(inventory_before['invalid_artifacts']),'missing_signals':list(inventory_after['missing_signals'])}
