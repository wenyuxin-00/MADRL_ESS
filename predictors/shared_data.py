from __future__ import annotations
import hashlib,json,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd,torch
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
from data.loaders.registry import _resolve_split_dates,_same_year_has_explicit_train_range,build_dataset
from predictors.lstm_forecaster import LSTMForecaster,load_lstm_forecaster_artifacts
from predictors.registry import build_forecaster
from predictors.time_features import DEFAULT_LOCAL_TIMEZONE,coerce_timestamp_index
from predictors.training import ensure_lstm_artifacts
from scripts.utils.price_protocol import PRICE_PROTOCOL_VERSION,WHOLESALE_PRICE_SEQ_FIELD,WHOLESALE_PRICE_SIGNAL,assert_no_legacy_price_schema
from scripts.utils.project_paths import get_shared_data_root
_FLOAT_PRECISION,_SCHEMA_VERSION=6,3
_ARTIFACT_META_KEYS='signal_name','future_horizon','seq_len','hidden_size','num_layers','dropout','input_size','time_feature_mode','model_mode','normalization_mode','postprocess_mode','baseline_mode','blend_weight','component','agent_index','agent_profile'
def _json_default(value:Any):
	if isinstance(value,(Path,torch.device)):return str(value)
	if hasattr(value,'item'):
		try:return value.item()
		except Exception:pass
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
def _normalize_for_signature(value:Any)->Any:
	if isinstance(value,dict):return{str(k):_normalize_for_signature(v)for(k,v)in sorted(dict(value).items(),key=lambda item:str(item[0]))}
	if isinstance(value,(list,tuple)):return[_normalize_for_signature(v)for v in value]
	if isinstance(value,(np.floating,float)):return f"{float(round(float(value),_FLOAT_PRECISION)):.{_FLOAT_PRECISION}f}"
	if isinstance(value,(np.integer,int)):return int(value)
	return str(value)if isinstance(value,Path)else value
def _signature_hash(payload:dict[str,object])->str:return hashlib.sha256(json.dumps(_normalize_for_signature(payload),sort_keys=True,ensure_ascii=True).encode('utf-8')).hexdigest()[:16]
def _file_sha256(path:str|Path)->str:
	digest=hashlib.sha256()
	with Path(path).open('rb')as handle:
		for chunk in iter(lambda:handle.read(1048576),b''):digest.update(chunk)
	return digest.hexdigest()
def _artifact_meta(meta:dict[str,object])->dict[str,object]:
	normalized:dict[str,object]={}
	for key in _ARTIFACT_META_KEYS:
		if key=='future_horizon':
			raw_value=meta.get('future_horizon',meta.get('pred_len',0))
			normalized[key]=int(0 if raw_value is None else raw_value)
			continue
		if key in{'seq_len','hidden_size','num_layers','input_size'}:
			raw_value=meta.get(key,0)
			normalized[key]=int(0 if raw_value is None else raw_value)
			continue
		if key=='agent_index':
			raw_value=meta.get(key)
			normalized[key]=None if raw_value is None else int(raw_value)
			continue
		if key=='dropout':
			raw_value=meta.get(key,.0)
			normalized[key]=float(.0 if raw_value is None else raw_value)
			continue
		normalized[key]=meta.get(key)
	return _normalize_for_signature(normalized)
def _artifact_fingerprint(forecast_ready:dict[str,object])->dict[str,object]:
	artifacts=dict(forecast_ready.get('artifacts')or{})
	if not artifacts:raise FileNotFoundError('No managed LSTM artifacts are available for shared MADRL data generation.')
	return{str(signal_name):[{'model_sha256':_file_sha256(model_path),'meta_sha256':_file_sha256(meta_path),'scaler_sha256':_file_sha256(scaler_path),'meta':_artifact_meta(load_lstm_forecaster_artifacts(model_path,meta_path,scaler_path)[0])}for(model_path,meta_path,scaler_path)in LSTMForecaster._normalize_artifact_bundle(bundle)]for(signal_name,bundle)in sorted(artifacts.items())}
def _test_window_in_signature(cfg)->bool:return int(cfg.data.train_year)==int(cfg.data.test_year)and not _same_year_has_explicit_train_range(cfg)
def _test_window_strategy(cfg)->str:return'cfg_window'if _test_window_in_signature(cfg)else'full_year_runtime_slice'
def _shared_data_signature_payload(cfg,*,artifact_fingerprint:dict[str,object])->dict[str,object]:return{'schema_version':_SCHEMA_VERSION,'num_agents':int(cfg.env.num_agents),'episode_limit':int(cfg.env.episode_limit),'future_horizon':int(cfg.env.future_horizon),'agent_profiles':list(cfg.data.agent_profiles),'agent_bus_ids':list(cfg.grid.agent_bus_ids),'train_year':int(cfg.data.train_year),'test_year':int(cfg.data.test_year),'train_start_date':cfg.data.train_start_date,'train_end_date':cfg.data.train_end_date,'same_year_has_explicit_train_range':bool(_same_year_has_explicit_train_range(cfg)),'test_window_in_signature':bool(_test_window_in_signature(cfg)),'test_window_strategy':_test_window_strategy(cfg),'test_start_date':cfg.data.test_start_date if _test_window_in_signature(cfg)else None,'test_end_date':cfg.data.test_end_date if _test_window_in_signature(cfg)else None,'load_components':list(cfg.data.load_components),'pv_reference':str(cfg.data.pv_reference),'pv_capacity_kw':list(cfg.data.pv_capacity_kw),'load_scale':list(cfg.data.load_scale),'pv_scale':list(cfg.data.pv_scale),'price_protocol_version':int(PRICE_PROTOCOL_VERSION),'artifacts':artifact_fingerprint}
def _resolve_shared_split_controls(cfg,split:str)->dict[str,object]:kwargs={'override_start_date':None,'override_end_date':None,'override_exclude_start_date':None,'override_exclude_end_date':None}if str(split)=='test'and _test_window_strategy(cfg)=='full_year_runtime_slice'else{};selected_year,start_date,end_date,exclude_start_date,exclude_end_date=_resolve_split_dates(cfg,split,**kwargs);return{'split':str(split),'year':int(selected_year),'start_date':start_date,'end_date':end_date,'exclude_start_date':exclude_start_date,'exclude_end_date':exclude_end_date,'window_strategy':_test_window_strategy(cfg)if str(split)=='test'else'cfg_window'}
def _episode_manifest_entry(dataset,episode_idx:int,episode:dict[str,object])->dict[str,object]:history_start_idx,active_start_idx,active_end_idx=list(getattr(dataset,'_episode_slices',[]))[int(episode_idx)];timestamps=[str(value)for value in list(dict(episode.get('meta',{})).get('timestamps')or[])];first,last=(None,None)if not timestamps else(str(timestamps[0]),str(timestamps[-1]));return{'episode_idx':int(episode_idx),'first_timestamp':first,'last_timestamp':last,'first_local_date':None if first is None else str(pd.Timestamp(first).date()),'last_local_date':None if last is None else str(pd.Timestamp(last).date()),'history_start_idx':int(history_start_idx),'active_start_idx':int(active_start_idx),'active_end_idx':int(active_end_idx)}
def _build_calendar_time_matrix(timestamps:list[str|pd.Timestamp]|np.ndarray|tuple[str|pd.Timestamp,...],n_agents:int)->np.ndarray:
	index=coerce_timestamp_index(timestamps)
	if getattr(index,'tz',None)is not None:index=index.tz_convert(DEFAULT_LOCAL_TIMEZONE)
	hour=index.hour.to_numpy(dtype=np.float32)+index.minute.to_numpy(dtype=np.float32)/np.float32(6e1);year=index.dayofyear.to_numpy(dtype=np.float32)-np.float32(1.)+hour/np.float32(24.);per_step=np.stack([np.sin(np.float32(2.*np.pi)*hour/np.float32(24.)),np.cos(np.float32(2.*np.pi)*hour/np.float32(24.)),np.sin(np.float32(2.*np.pi)*year/np.float32(365.25)),np.cos(np.float32(2.*np.pi)*year/np.float32(365.25))],axis=1).astype(np.float32,copy=False);return np.broadcast_to(per_step[:,None,:],(len(index),int(n_agents),4)).astype(np.float32,copy=False)
def _compute_future_mean_price(wholesale_price:np.ndarray,future_horizon:int)->tuple[np.ndarray,np.ndarray]:
	values=np.asarray(wholesale_price,dtype=np.float32).reshape(-1)
	if values.size==0:return np.zeros((0,),dtype=np.float32),np.zeros((0,),dtype=np.float32)
	horizon=int(max(future_horizon,1));idx=np.arange(values.size,dtype=np.int64);csum=np.concatenate([[.0],np.cumsum(values.astype(np.float64),dtype=np.float64)]);start,end=idx+1,np.minimum(idx+1+horizon,values.size);next_idx,start_next=np.minimum(idx+1,values.size-1),np.minimum(idx+2,values.size);end_next=np.minimum(next_idx+1+horizon,values.size);mu_t=np.where(start<values.size,(csum[end]-csum[start])/np.maximum(end-start,1),values[np.minimum(idx,values.size-1)]);mu_next=np.where(start_next<values.size,(csum[end_next]-csum[start_next])/np.maximum(end_next-start_next,1),values[next_idx]);return mu_t.astype(np.float32),mu_next.astype(np.float32)
def _merge_history_with_episode_signals(signals:dict[str,np.ndarray],history_signals:dict[str,np.ndarray])->tuple[dict[str,np.ndarray],int]:history_arrays={name:np.asarray(history_signals.get(name),dtype=np.float32)for name in signals};merged={name:np.concatenate([history_arrays[name],np.asarray(signal,dtype=np.float32)],axis=0).astype(np.float32,copy=False)if history_arrays[name].size else np.asarray(signal,dtype=np.float32).copy()for(name,signal)in signals.items()};prefix=max((int(values.shape[0])for values in history_arrays.values()),default=0);return merged,prefix
def _open_split_arrays(split_dir:Path,*,n_episodes:int,episode_length:int,n_agents:int,sequence_length:int)->tuple[dict[str,str],dict[str,np.ndarray]]:files={'calendar_time':'calendar_time.npy',WHOLESALE_PRICE_SEQ_FIELD:f"{WHOLESALE_PRICE_SEQ_FIELD}.npy",'load_seq':'load_seq.npy','pv_seq':'pv_seq.npy','mu_t':'mu_t.npy','mu_next':'mu_next.npy'};shapes={'calendar_time':(n_episodes,episode_length,n_agents,4),WHOLESALE_PRICE_SEQ_FIELD:(n_episodes,episode_length,sequence_length),'load_seq':(n_episodes,episode_length,n_agents,sequence_length),'pv_seq':(n_episodes,episode_length,n_agents,sequence_length),'mu_t':(n_episodes,episode_length),'mu_next':(n_episodes,episode_length)};arrays={name:open_memmap(split_dir/file_name,mode='w+',dtype=np.float32,shape=shapes[name])for(name,file_name)in files.items()};return files,arrays
def _close_split_arrays(arrays:dict[str,np.ndarray])->None:
	for array in arrays.values():
		array.flush();mmap=getattr(array,'_mmap',None)
		if mmap is not None:mmap.close()
def _write_split_shared_data(cfg,*,split:str,split_dir:Path)->dict[str,object]:
	split_controls=_resolve_shared_split_controls(cfg,str(split));dataset=build_dataset(cfg,mode=str(split),override_start_date=split_controls['start_date'],override_end_date=split_controls['end_date'],override_exclude_start_date=split_controls['exclude_start_date'],override_exclude_end_date=split_controls['exclude_end_date']);n_episodes,episode_length,n_agents,sequence_length=int(dataset.num_episodes()),int(cfg.env.episode_limit),int(cfg.env.num_agents),int(cfg.env.future_horizon)+1;split_dir.mkdir(parents=True,exist_ok=True);files,arrays=_open_split_arrays(split_dir,n_episodes=n_episodes,episode_length=episode_length,n_agents=n_agents,sequence_length=sequence_length);episode_manifest,predict_specs=[],{WHOLESALE_PRICE_SEQ_FIELD:WHOLESALE_PRICE_SIGNAL,'load_seq':'load','pv_seq':'pv'}
	try:
		forecaster=build_forecaster(cfg);vectorized=hasattr(forecaster,'predict_episode_matrix')
		for episode_idx in tqdm(range(n_episodes),desc=f"shared_data[{split}] episodes",leave=False):
			episode=dataset.get_episode(episode_idx);signals={key:np.asarray(value,dtype=np.float32)for(key,value)in dict(episode.get('signals',{})).items()};history={key:np.asarray(value,dtype=np.float32)for(key,value)in dict(episode.get('history_signals',{})).items()};meta=dict(episode.get('meta',{}));timestamps=list(meta.get('timestamps')or[]);merged,prefix=_merge_history_with_episode_signals(signals,history);all_timestamps=[str(v)for v in list(episode.get('history_timestamps')or[])]+timestamps;episode_manifest.append(_episode_manifest_entry(dataset,episode_idx,episode));forecaster.reset();forecaster.set_episode(merged,meta);arrays['mu_t'][episode_idx],arrays['mu_next'][episode_idx]=_compute_future_mean_price(signals[WHOLESALE_PRICE_SIGNAL],int(cfg.env.future_horizon));arrays['calendar_time'][episode_idx]=_build_calendar_time_matrix(timestamps,n_agents)
			if vectorized:
				for(out_name,signal_name)in predict_specs.items():arrays[out_name][episode_idx]=forecaster.predict_episode_matrix(merged[signal_name],sequence_length,signal_name=signal_name,history_timestamps=all_timestamps).astype(np.float32)[prefix:]
				continue
			for step_idx in range(episode_length):
				current_timestamps,end=all_timestamps[:prefix+step_idx+1],prefix+step_idx+1
				for(out_name,signal_name)in predict_specs.items():arrays[out_name][episode_idx,step_idx]=forecaster.predict(merged[signal_name][:end],sequence_length,signal_name=signal_name,history_timestamps=current_timestamps).astype(np.float32)
	finally:_close_split_arrays(arrays)
	manifest={'schema_version':_SCHEMA_VERSION,'price_protocol_version':int(PRICE_PROTOCOL_VERSION),'split':str(split),'num_episodes':n_episodes,'episode_length':episode_length,'num_agents':n_agents,'sequence_length':sequence_length,'split_controls':split_controls,'episodes':episode_manifest,'files':files};(split_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,default=_json_default),encoding='utf-8');return manifest
def _shared_data_root(root:str|Path|None=None)->Path:return get_shared_data_root(root)/'mainline'
def _root_manifest_is_complete(shared_dir:Path,signature_hash:str)->bool:
	try:manifest=json.loads((shared_dir/'manifest.json').read_text(encoding='utf-8'))
	except(OSError,json.JSONDecodeError,FileNotFoundError):return False
	return str(manifest.get('signature_hash',''))==str(signature_hash)and all((shared_dir/split/'manifest.json').exists()for split in('train','test'))
class SharedDataResult:
	def __init__(self,*,shared_data_dir:Path,signature_hash:str,manifest:dict[str,object],reused:bool):self.shared_data_dir,self.signature_hash,self.manifest,self.reused=Path(shared_data_dir),str(signature_hash),dict(manifest),bool(reused)
def build_shared_data_status_summary(result:SharedDataResult,*,test_start_date:str|None=None,test_end_date:str|None=None)->dict[str,object]:return{'shared_data_status':'reused_existing_shared_data'if result.reused else'generated_new_shared_data','shared_data_message':'Reused existing shared MADRL data package.'if result.reused else'Generated a new shared MADRL data package from available forecast artifacts.','shared_data_dir':str(result.shared_data_dir),'shared_data_signature':str(result.signature_hash),'shared_data_reused':bool(result.reused),'test_start_date':test_start_date,'test_end_date':test_end_date}
class PrecomputedObservationStore:
	def __init__(self,split_dir:str|Path):
		self.split_dir=Path(split_dir).resolve();self.manifest=json.loads((self.split_dir/'manifest.json').read_text(encoding='utf-8'));assert_no_legacy_price_schema(dict(self.manifest.get('files',{})).keys(),context='Shared-data manifest')
		if int(self.manifest.get('schema_version',-1))!=int(_SCHEMA_VERSION):raise ValueError(f"Unsupported shared-data schema version at '{self.split_dir}': expected={_SCHEMA_VERSION}, actual={self.manifest.get('schema_version')!r}.")
		if int(self.manifest.get('price_protocol_version',PRICE_PROTOCOL_VERSION))!=int(PRICE_PROTOCOL_VERSION):raise ValueError(f"Unsupported shared-data price protocol version at '{self.split_dir}': expected={PRICE_PROTOCOL_VERSION}, actual={self.manifest.get('price_protocol_version')!r}.")
		self._arrays={name:np.load(self.split_dir/file_name,mmap_mode='r')for(name,file_name)in dict(self.manifest.get('files',{})).items()}
	def episode(self,episode_idx:int)->dict[str,np.ndarray]:return{name:np.asarray(array[int(episode_idx)])for(name,array)in self._arrays.items()}
def load_madrl_shared_data_manifest(path:str|Path)->dict[str,object]:return json.loads((Path(path).resolve()/'manifest.json').read_text(encoding='utf-8'))
def select_shared_data_episode_indices(manifest:dict[str,object],*,start_date:str|None,end_date:str|None)->list[int]:
	if not(episodes:=list(manifest.get('episodes')or[])):raise ValueError('Shared-data split manifest does not contain episode metadata.')
	if start_date in(None,'')and end_date in(None,''):return[int(entry['episode_idx'])for entry in episodes]
	normalized_start,normalized_end=None if start_date in(None,'')else pd.Timestamp(str(start_date)).date(),None if end_date in(None,'')else pd.Timestamp(str(end_date)).date();selected:list[int]=[]
	for entry in episodes:
		first_local_date=None if entry.get('first_local_date')in(None,'')else pd.Timestamp(str(entry['first_local_date'])).date();last_local_date=None if entry.get('last_local_date')in(None,'')else pd.Timestamp(str(entry['last_local_date'])).date()
		if normalized_start is not None and(first_local_date is None or first_local_date<normalized_start):continue
		if normalized_end is not None and(last_local_date is None or last_local_date>normalized_end):continue
		selected.append(int(entry['episode_idx']))
	if not selected:raise ValueError(f"No shared-data episodes fall fully inside the requested date window: start_date={start_date!r}, end_date={end_date!r}.")
	return selected
def ensure_madrl_shared_data(cfg,*,root:str|Path|None=None)->SharedDataResult:
	forecast_ready=None if str(cfg.forecast.type).strip().lower()!='lstm'else ensure_lstm_artifacts(cfg,device=cfg.runtime.device);artifact_info={'forecast_ready':forecast_ready,'fingerprint':{'mode':'non_lstm'}if forecast_ready is None else _artifact_fingerprint(forecast_ready)};signature_payload=_shared_data_signature_payload(cfg,artifact_fingerprint=dict(artifact_info['fingerprint']));signature_hash=_signature_hash(signature_payload);root_dir=_shared_data_root(root);shared_dir=(root_dir/signature_hash).resolve();root_dir.mkdir(parents=True,exist_ok=True)
	if _root_manifest_is_complete(shared_dir,signature_hash):return SharedDataResult(shared_data_dir=shared_dir,signature_hash=signature_hash,manifest=load_madrl_shared_data_manifest(shared_dir),reused=True)
	temp_dir=(root_dir/f"{signature_hash}.tmp.{uuid.uuid4().hex}").resolve()
	try:
		temp_dir.mkdir(parents=True,exist_ok=False);train_manifest,test_manifest=_write_split_shared_data(cfg,split='train',split_dir=temp_dir/'train'),_write_split_shared_data(cfg,split='test',split_dir=temp_dir/'test');manifest={'schema_version':_SCHEMA_VERSION,'price_protocol_version':int(PRICE_PROTOCOL_VERSION),'signature_hash':signature_hash,'signature':_normalize_for_signature(signature_payload),'created_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),'shared_data_dir':str(shared_dir),'artifact_inventory':dict(artifact_info['fingerprint']),'data_controls':{'agent_profiles':list(cfg.data.agent_profiles),'agent_bus_ids':list(cfg.grid.agent_bus_ids),'train_year':int(cfg.data.train_year),'test_year':int(cfg.data.test_year),'train_start_date':cfg.data.train_start_date,'train_end_date':cfg.data.train_end_date,'test_start_date':cfg.data.test_start_date,'test_end_date':cfg.data.test_end_date,'same_year_has_explicit_train_range':bool(_same_year_has_explicit_train_range(cfg)),'test_window_strategy':_test_window_strategy(cfg),'load_scale':_normalize_for_signature(list(cfg.data.load_scale)),'pv_scale':_normalize_for_signature(list(cfg.data.pv_scale)),'pv_capacity_kw':_normalize_for_signature(list(cfg.data.pv_capacity_kw)),'load_components':list(cfg.data.load_components),'pv_reference':str(cfg.data.pv_reference),'future_horizon':int(cfg.env.future_horizon),'episode_limit':int(cfg.env.episode_limit)},'splits':{'train':train_manifest,'test':test_manifest}};(temp_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,default=_json_default),encoding='utf-8')
		if shared_dir.exists():shutil.rmtree(shared_dir,ignore_errors=True)
		temp_dir.replace(shared_dir);return SharedDataResult(shared_data_dir=shared_dir,signature_hash=signature_hash,manifest=manifest,reused=False)
	finally:
		if temp_dir.exists():shutil.rmtree(temp_dir,ignore_errors=True)
