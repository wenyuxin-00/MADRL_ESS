from __future__ import annotations
import numpy as np,pandas as pd
from envs.observation.normalization import ObservationNormalizer
_SAFETY_LOCAL_FIELDS=['soc_raw','load_raw','pv_raw','battery_capacity_kwh','p_max_kw']
_LOCAL_DIMS={'time':2,'calendar_time':4,'wholesale_price':1,'load':1,'pv':1,'soc':1}
_SEQUENCE_SCOPES={'wholesale_price':'shared','load':'per_agent','pv':'per_agent'}
_TS_FALLBACK=pd.Timestamp('2000-01-01 00:00:00+00:00')
def _adjacency(n_agents:int,adjacency_type:str)->np.ndarray:
	if adjacency_type=='identity':return np.eye(n_agents,dtype=np.float32)
	if adjacency_type!='fully_connected_no_self':raise ValueError(f"Unknown adjacency_type '{adjacency_type}', expected 'identity' or 'fully_connected_no_self'.")
	adjacency=np.ones((n_agents,n_agents),dtype=np.float32);np.fill_diagonal(adjacency,.0);return adjacency
def _time_features(cur_step:int,episode_length:int,n_agents:int)->np.ndarray:phase=2.*np.pi*float(cur_step)/float(episode_length);features=np.asarray([np.sin(phase),np.cos(phase)],dtype=np.float32);return np.repeat(features.reshape(1,-1),int(n_agents),axis=0)
def _calendar_features(timestamp_value:str,n_agents:int)->np.ndarray:timestamp=pd.Timestamp(timestamp_value);hour_of_day=float(timestamp.hour)+float(timestamp.minute)/6e1;day_of_year=float(timestamp.dayofyear-1)+hour_of_day/24.;phases=np.asarray([np.sin(2.*np.pi*hour_of_day/24.),np.cos(2.*np.pi*hour_of_day/24.),np.sin(2.*np.pi*day_of_year/365.25),np.cos(2.*np.pi*day_of_year/365.25)],dtype=np.float32);return np.repeat(phases.reshape(1,-1),int(n_agents),axis=0)
def _pad_sequence(values,start:int,length:int)->np.ndarray:
	array=np.asarray(values,dtype=np.float32);tail=array[int(start):int(start)+int(length)];pad_rows=int(length)-int(tail.shape[0])
	if pad_rows>0:pad_width=((0,pad_rows),)+tuple((0,0)for _ in range(max(array.ndim-1,0)));tail=np.pad(tail,pad_width,mode='constant')
	return tail.astype(np.float32,copy=False)if array.ndim==1 else tail.T.astype(np.float32,copy=False)
def _signal_history(env,signal_name:str)->np.ndarray:signal=env.get_signal(signal_name);prefix=np.asarray(env.history_signals.get(signal_name),dtype=np.float32);return signal[:env.cur_step+1].copy()if prefix.size==0 else np.concatenate([prefix,signal[:env.cur_step+1]],axis=0).astype(np.float32,copy=False)
class DefaultObservationBuilder:
	def __init__(self,local_features:list[str],sequence_features:list[str],future_horizon:int,adjacency_type:str='identity',normalizer:ObservationNormalizer|None=None,precomputed:bool=False):
		self.local_feature_names=[str(name)for name in local_features];self.sequence_feature_names=[str(name)for name in sequence_features];unknown_local=sorted(set(self.local_feature_names)-set(_LOCAL_DIMS));unknown_sequence=sorted(set(self.sequence_feature_names)-set(_SEQUENCE_SCOPES))
		if unknown_local:raise ValueError(f"Unknown local feature(s): {unknown_local}")
		if unknown_sequence:raise ValueError(f"Unknown sequence feature(s): {unknown_sequence}")
		self.future_horizon=int(future_horizon);self.sequence_length=self.future_horizon+1;self.adjacency_type=str(adjacency_type);self.normalizer=normalizer;self.precomputed=bool(precomputed);self.local_dim=sum(_LOCAL_DIMS[name]for name in self.local_feature_names)
	def get_schema(self,n_agents:int)->dict[str,tuple[int,...]]:return{'local':(int(n_agents),self.local_dim),'adjacency':(int(n_agents),int(n_agents)),'safety_local':(int(n_agents),len(_SAFETY_LOCAL_FIELDS)),**{f"{name}_seq":(self.sequence_length,)if _SEQUENCE_SCOPES[name]=='shared'else(int(n_agents),self.sequence_length)for name in self.sequence_feature_names}}
	def get_layout(self,n_agents:int)->dict[str,dict]:
		layout={'local':{'group':'local','scope':'per_agent','dim':int(self.local_dim),'fields':list(self.local_feature_names)},'adjacency':{'group':'graph','scope':'shared','dim':int(n_agents),'fields':['adjacency']},'safety_local':{'group':'projector','scope':'per_agent','dim':len(_SAFETY_LOCAL_FIELDS),'fields':list(_SAFETY_LOCAL_FIELDS)}};schema=self.get_schema(n_agents)
		for name in self.sequence_feature_names:layout[f"{name}_seq"]={'feature_name':name,'group':'sequence','scope':_SEQUENCE_SCOPES[name],'dim':1,'shape':schema[f"{name}_seq"]}
		return layout
	def zeros(self,n_agents:int)->dict[str,np.ndarray]:return{key:np.zeros(shape,dtype=np.float32)for(key,shape)in self.get_schema(n_agents).items()}
	def build_raw(self,env)->dict[str,np.ndarray]:return self._build(env,normalize=False)
	def build(self,env)->dict[str,np.ndarray]:return self._build(env,normalize=True)
	def _normalize(self,feature_name:str,values:np.ndarray,*,normalize:bool,sequence:bool)->np.ndarray:
		if not normalize or self.normalizer is None:return np.asarray(values,dtype=np.float32)
		transform=self.normalizer.transform_sequence if sequence else self.normalizer.transform_local;return np.asarray(transform(feature_name,values),dtype=np.float32)
	def _local_feature(self,env,name:str)->np.ndarray:
		if name=='time':return _time_features(env.cur_step,env.episode_length,env.n)
		if name=='calendar_time':
			if self.precomputed:
				if'calendar_time'not in env._episode_precomputed:raise KeyError("Precomputed observation data does not contain 'calendar_time'.")
				return np.asarray(env._episode_precomputed['calendar_time'][env.cur_step],dtype=np.float32)
			timestamps=list(dict(getattr(env,'episode_meta',{})).get('timestamps')or[]);timestamp=str(timestamps[int(env.cur_step)])if 0<=int(env.cur_step)<len(timestamps)else str(_TS_FALLBACK+pd.Timedelta(minutes=15*int(env.cur_step)));return _calendar_features(timestamp,env.n)
		if name=='soc':return np.asarray(env.soc,dtype=np.float32).reshape(-1,1)
		if name=='wholesale_price':return np.full((env.n,1),float(env.get_signal_step(name)),dtype=np.float32)
		return np.asarray(env.get_signal_step(name),dtype=np.float32).reshape(-1,1)
	def _sequence_feature(self,env,name:str)->np.ndarray:
		if self.precomputed:
			cache_key=f"{name}_seq"
			if cache_key not in env._episode_precomputed:raise KeyError(f"Precomputed observation data does not contain '{cache_key}'.")
			values=np.asarray(env._episode_precomputed[cache_key][env.cur_step],dtype=np.float32);expected_shape=(self.sequence_length,)if _SEQUENCE_SCOPES[name]=='shared'else(int(env.n),self.sequence_length)
			if tuple(values.shape)!=expected_shape:raise ValueError(f"Precomputed observation horizon contract mismatch at DefaultObservationBuilder._sequence_feature(...): old shared-data object '{getattr(env,'_precomputed_data_dir',None)}' returned '{cache_key}' with shape {tuple(values.shape)}, but current cfg.env.future_horizon={self.future_horizon} and env.n={int(env.n)} require shape {expected_shape}. Expected shared-data generated with the current config. Re-run notebooks/forecast/forecast_lstm.ipynb, then rerun the consuming notebook or entrypoint.")
			return values
		if getattr(env,'forecaster',None)is None:return _pad_sequence(env.get_signal(name),env.cur_step,self.sequence_length)
		history=_signal_history(env,name);timestamps=list(dict(getattr(env,'episode_meta',{})).get('timestamps')or[]);history_timestamps=[*env.history_timestamps,*[str(timestamp)for timestamp in timestamps[:max(0,int(env.cur_step)+1)]]];return np.asarray(env.forecaster.predict(history,self.sequence_length,signal_name=name,history_timestamps=history_timestamps),dtype=np.float32)
	def _build(self,env,*,normalize:bool)->dict[str,np.ndarray]:
		local_parts=[self._normalize(name,self._local_feature(env,name),normalize=normalize,sequence=False)for name in self.local_feature_names];obs={'local':np.concatenate(local_parts,axis=1).astype(np.float32)if local_parts else np.zeros((env.n,0),dtype=np.float32),'adjacency':_adjacency(env.n,self.adjacency_type),'safety_local':env._current_safety_local().astype(np.float32)}
		for name in self.sequence_feature_names:obs[f"{name}_seq"]=self._normalize(name,self._sequence_feature(env,name),normalize=normalize,sequence=True)
		return obs
