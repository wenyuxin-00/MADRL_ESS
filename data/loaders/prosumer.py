from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Sequence
import numpy as np,pandas as pd
from data.loaders.constants import DEFAULT_PROCESSED_SUBDIR,PROSUMER_PROFILE_ALLOWLIST,TZ_LOCAL
AVAILABLE_AGENT_PROFILES=tuple(PROSUMER_PROFILE_ALLOWLIST)
VALID_LOAD_COMPONENTS='household','heatpump'
VALID_PV_REFERENCES='east','south','west'
def _resolve_dataset_dir(data_dir:str|Path)->Path:
	root=Path(data_dir)
	for candidate in(root,root/DEFAULT_PROCESSED_SUBDIR,root/'prosumer'):
		if(candidate/'household.csv').exists():return candidate
	return root/DEFAULT_PROCESSED_SUBDIR
def _parse_optional_local_date(value:str|date|None)->date|None:return None if value in(None,'')else pd.Timestamp(value).date()
def _coerce_scale_vector(scale:Sequence[float]|float|None,*,name:str,n_agents:int)->np.ndarray:
	if scale is None:return np.ones((n_agents,),dtype=np.float32)
	if isinstance(scale,(int,float,np.floating)):
		if float(scale)<.0:raise ValueError(f"{name} must be non-negative, got {scale}.")
		return np.full((n_agents,),float(scale),dtype=np.float32)
	values=np.asarray(list(scale),dtype=np.float32).reshape(-1)
	if values.size==0:return np.ones((n_agents,),dtype=np.float32)
	if values.size!=n_agents:raise ValueError(f"{name} length should equal n_agents={n_agents}, got {values.size}")
	if np.any(values<.0):raise ValueError(f"{name} must be non-negative, got {values.tolist()}.")
	return values.astype(np.float32)
class ProsumerDataset:
	def __init__(self,data_dir:str|Path,episode_length:int,n_agents:int,agent_profiles:list[str],year:int,start_date:str|date|None=None,end_date:str|date|None=None,exclude_start_date:str|date|None=None,exclude_end_date:str|date|None=None,load_components:Sequence[str]=('household','heatpump'),pv_reference:str='south',pv_capacity_kw:Sequence[float]|None=None,load_scale:Sequence[float]|float|None=None,pv_scale:Sequence[float]|float|None=None,node_ids:Sequence[int]|None=None,history_warmup_steps:int=0)->None:self.data_dir=_resolve_dataset_dir(data_dir);self.episode_length=int(episode_length);self.history_warmup_steps=int(history_warmup_steps);self.n_agents=int(n_agents);self.agent_profiles=list(agent_profiles);self.year=int(year);self.start_date,self.end_date=_parse_optional_local_date(start_date),_parse_optional_local_date(end_date);self.exclude_start_date,self.exclude_end_date=_parse_optional_local_date(exclude_start_date),_parse_optional_local_date(exclude_end_date);self.load_components,self.pv_reference=tuple(str(component)for component in load_components),str(pv_reference).lower();self.pv_capacity_kw=None if pv_capacity_kw is None or len(pv_capacity_kw)==0 else list(pv_capacity_kw);self.load_scale,self.pv_scale=_coerce_scale_vector(load_scale,name='load_scale',n_agents=self.n_agents),_coerce_scale_vector(pv_scale,name='pv_scale',n_agents=self.n_agents);self.node_ids=list(node_ids)if node_ids is not None else list(range(self.n_agents));(self._signals):dict[str,np.ndarray]={};(self._timestamps):pd.Series|None=None;(self._meta_template):dict[str,object]={};(self._episode_slices):list[tuple[int,int,int]]=[];self._num_episodes=0;self._validate_init_args();self._load()
	def _validate_init_args(self)->None:
		if self.episode_length<=0:raise ValueError(f"episode_length must be positive, got {self.episode_length}")
		if self.history_warmup_steps<0:raise ValueError(f"history_warmup_steps must be non-negative, got {self.history_warmup_steps}")
		if len(self.agent_profiles)!=self.n_agents:raise ValueError(f"agent_profiles length should equal n_agents={self.n_agents}, got {len(self.agent_profiles)}")
		unknown_profiles=[profile for profile in self.agent_profiles if profile not in AVAILABLE_AGENT_PROFILES]
		if unknown_profiles:raise ValueError(f"Unknown agent profiles: {unknown_profiles}. Allowed: {list(AVAILABLE_AGENT_PROFILES)}")
		invalid_components=[component for component in self.load_components if component not in VALID_LOAD_COMPONENTS]
		if invalid_components:raise ValueError(f"Unsupported load components {invalid_components}. Allowed: {list(VALID_LOAD_COMPONENTS)}")
		if not self.load_components:raise ValueError('At least one load component is required.')
		if self.pv_reference not in VALID_PV_REFERENCES:raise ValueError(f"Unsupported pv_reference '{self.pv_reference}'. Allowed: {list(VALID_PV_REFERENCES)}")
		if self.pv_capacity_kw is not None and len(self.pv_capacity_kw)!=self.n_agents:raise ValueError(f"pv_capacity_kw length should equal n_agents={self.n_agents}, got {len(self.pv_capacity_kw)}")
		if len(self.node_ids)!=self.n_agents:raise ValueError(f"node_ids length should equal n_agents={self.n_agents}, got {len(self.node_ids)}")
		if self.start_date is not None and self.end_date is not None and self.start_date>self.end_date:raise ValueError(f"start_date should be <= end_date, got {self.start_date} > {self.end_date}")
		if self.exclude_start_date is not None and self.exclude_end_date is not None and self.exclude_start_date>self.exclude_end_date:raise ValueError(f"exclude_start_date should be <= exclude_end_date, got {self.exclude_start_date} > {self.exclude_end_date}")
	def _read_csv(self,name:str)->pd.DataFrame:
		path=self.data_dir/name
		if not path.exists():raise FileNotFoundError(f"Missing processed prosumer file: {path}")
		frame=pd.read_csv(path)
		if'timestamp'not in frame.columns:raise ValueError(f"Processed prosumer file '{path}' must contain a 'timestamp' column.")
		frame['timestamp']=pd.to_datetime(frame['timestamp'],utc=True).dt.tz_convert(TZ_LOCAL);return frame
	def _exclude_date_mask(self,local_dates:pd.Series)->np.ndarray:
		if self.exclude_start_date is not None:
			if self.exclude_end_date is None:return(local_dates>=self.exclude_start_date).to_numpy(dtype=bool)
			return((local_dates>=self.exclude_start_date)&(local_dates<=self.exclude_end_date)).to_numpy(dtype=bool)
		if self.exclude_end_date is not None:return(local_dates<=self.exclude_end_date).to_numpy(dtype=bool)
		return np.zeros((len(local_dates),),dtype=bool)
	def _build_split_masks(self,timestamps:pd.Series)->tuple[np.ndarray,np.ndarray]:
		local_dates=timestamps.dt.date;year_mask,exclude_mask,active_mask=(timestamps.dt.year==self.year).to_numpy(dtype=bool),self._exclude_date_mask(local_dates),(timestamps.dt.year==self.year).to_numpy(dtype=bool)
		if self.start_date is not None:active_mask&=(local_dates>=self.start_date).to_numpy(dtype=bool)
		if self.end_date is not None:active_mask&=(local_dates<=self.end_date).to_numpy(dtype=bool)
		active_mask&=~exclude_mask;accessible_mask=year_mask&~exclude_mask if self.history_warmup_steps>0 else active_mask.copy();return active_mask,accessible_mask
	def _extract_component(self,frame:pd.DataFrame,component_name:str)->pd.DataFrame:
		missing_profiles=[profile for profile in self.agent_profiles if profile not in frame.columns]
		if missing_profiles:raise ValueError(f"Load component '{component_name}' is missing columns for {missing_profiles}")
		return frame.loc[:,['timestamp',*self.agent_profiles]].copy()
	def _load_components(self)->tuple[pd.DataFrame,dict[str,pd.DataFrame],np.ndarray,np.ndarray]:
		component_frames:dict[str,pd.DataFrame]={};base_timestamp:pd.Series|None=None;active_mask_on_accessible:np.ndarray|None=None;accessible_source_positions:np.ndarray|None=None
		for component in self.load_components:
			frame=self._read_csv(f"{component}.csv");active_mask,accessible_mask=self._build_split_masks(frame['timestamp']);frame=frame.loc[accessible_mask].copy()
			if frame.empty:raise ValueError(f"Processed prosumer data does not contain any accessible rows for year={self.year}, exclude_start_date={self.exclude_start_date}, exclude_end_date={self.exclude_end_date}.")
			frame=self._extract_component(frame,component)
			if base_timestamp is None:base_timestamp,active_mask_on_accessible,accessible_source_positions=frame['timestamp'].reset_index(drop=True),np.asarray(active_mask[accessible_mask],dtype=bool),np.flatnonzero(accessible_mask).astype(np.int64)
			elif not base_timestamp.equals(frame['timestamp'].reset_index(drop=True)):raise ValueError(f"Load component '{component}' timestamp axis does not match the base load axis.")
			component_frames[component]=frame.reset_index(drop=True)
		if base_timestamp is None or active_mask_on_accessible is None or accessible_source_positions is None:raise ValueError('No prosumer load components were resolved for dataset construction.')
		merged,total_load=pd.DataFrame({'timestamp':base_timestamp}),np.zeros((len(base_timestamp),self.n_agents),dtype=np.float32)
		for component in self.load_components:total_load+=component_frames[component].loc[:,self.agent_profiles].to_numpy(dtype=np.float32)
		merged.loc[:,self.agent_profiles]=total_load*self.load_scale[None,:];return merged,component_frames,active_mask_on_accessible.astype(bool,copy=False),accessible_source_positions.astype(np.int64,copy=False)
	def _load_pv(self,base_timestamps:pd.Series)->tuple[np.ndarray,np.ndarray]:
		frame=self._read_csv('pv_reference.csv');_,accessible_mask=self._build_split_masks(frame['timestamp']);frame=frame.loc[accessible_mask].copy().set_index('timestamp').sort_index().reindex(base_timestamps);column=f"ref_{self.pv_reference}"
		if column not in frame.columns:raise ValueError(f"PV reference file is missing column '{column}'.")
		reference=pd.to_numeric(frame[column],errors='coerce').to_numpy(dtype=np.float32)
		if np.isnan(reference).any():raise ValueError(f"PV reference '{column}' contains NaN values after timestamp alignment.")
		ref_peak_kw=float(np.max(reference))if reference.size>0 else .0
		if self.pv_capacity_kw is None:pv,pv_peak_kw=np.repeat(reference[:,None],self.n_agents,axis=1).astype(np.float32),np.full((self.n_agents,),ref_peak_kw,dtype=np.float32)
		else:
			if ref_peak_kw<=.0:raise ValueError('PV reference peak is zero, cannot scale by pv_capacity_kw.')
			scales=np.asarray(self.pv_capacity_kw,dtype=np.float32)/np.float32(ref_peak_kw);pv,pv_peak_kw=(reference[:,None]*scales[None,:]).astype(np.float32),np.asarray(self.pv_capacity_kw,dtype=np.float32).copy()
		pv,pv_peak_kw=(pv*self.pv_scale[None,:]).astype(np.float32),(pv_peak_kw*self.pv_scale).astype(np.float32);return pv,pv_peak_kw
	def _load_price(self,base_timestamps:pd.Series)->np.ndarray:
		frame=self._read_csv('price.csv');_,accessible_mask=self._build_split_masks(frame['timestamp']);frame=frame.loc[accessible_mask].copy().set_index('timestamp').sort_index().reindex(base_timestamps);wholesale_price=pd.to_numeric(frame['price'],errors='coerce').to_numpy(dtype=np.float32)
		if np.isnan(wholesale_price).any():raise ValueError('Processed prosumer wholesale price series contains NaN values after timestamp alignment.')
		return wholesale_price
	def _load(self)->None:
		load_frame,component_frames,active_mask,accessible_source_positions=self._load_components();base_timestamps=load_frame['timestamp'].reset_index(drop=True);wholesale_price,(pv,pv_peak_kw)=self._load_price(base_timestamps),self._load_pv(base_timestamps);load=load_frame.loc[:,self.agent_profiles].to_numpy(dtype=np.float32)
		if load.shape!=pv.shape:raise ValueError(f"Load and PV shapes must match, got {load.shape} vs {pv.shape}")
		self._timestamps,self._signals=base_timestamps,{'wholesale_price':wholesale_price,'load':load,'pv':pv}
		for comp_name in self.load_components:self._signals[f"load_{comp_name}"]=component_frames[comp_name][self.agent_profiles].to_numpy(np.float32)*self.load_scale[None,:]
		self._meta_template={'node_ids':list(self.node_ids),'agent_profiles':list(self.agent_profiles),'year':int(self.year),'date_range':{'start_date':self.start_date.isoformat()if self.start_date is not None else None,'end_date':self.end_date.isoformat()if self.end_date is not None else None},'excluded_date_range':{'start_date':self.exclude_start_date.isoformat()if self.exclude_start_date is not None else None,'end_date':self.exclude_end_date.isoformat()if self.exclude_end_date is not None else None},'load_components':list(self.load_components),'pv_reference':self.pv_reference,'pv_peak_kw':pv_peak_kw.copy(),'load_scale':self.load_scale.copy(),'pv_scale':self.pv_scale.copy(),'wholesale_price_unit':'EUR/kWh','power_unit':'kW','energy_unit':'kWh','data_dir':str(self.data_dir),'component_columns':{component:list(component_frames[component].columns)for component in component_frames}};self._episode_slices=[];total_steps=len(base_timestamps)
		if total_steps<=0:raise ValueError(f"Processed prosumer data is too short for episode_length={self.episode_length}: total_steps={total_steps}")
		active_positions=np.flatnonzero(active_mask)
		if active_positions.size==0:raise ValueError(f"Processed prosumer data does not contain any active rows for year={self.year}, start_date={self.start_date}, end_date={self.end_date}, exclude_start_date={self.exclude_start_date}, exclude_end_date={self.exclude_end_date}.")
		split_points=np.flatnonzero(np.diff(accessible_source_positions)!=1)+1;bounds=np.concatenate([[0],split_points,[len(accessible_source_positions)]])
		for(seg_start,seg_end)in zip(bounds[:-1],bounds[1:],strict=False):
			seg_active=active_positions[(active_positions>=seg_start)&(active_positions<seg_end)]
			if seg_active.size==0:continue
			run_split=np.flatnonzero(np.diff(seg_active)!=1)+1
			for run_positions in np.split(seg_active,run_split):
				first_active,last_active=int(run_positions[0]),int(run_positions[-1]);first_usable=max(first_active,seg_start+self.history_warmup_steps);usable_steps=last_active-first_usable+1
				if usable_steps>=self.episode_length:
					num_episodes=usable_steps//self.episode_length
					for episode_offset in range(num_episodes):active_start=first_usable+episode_offset*self.episode_length;active_end=active_start+self.episode_length;self._episode_slices.append((active_start-self.history_warmup_steps,active_start,active_end))
		self._num_episodes=len(self._episode_slices)
		if self._num_episodes<=0:history_msg=f"history_warmup_steps={self.history_warmup_steps}, "if self.history_warmup_steps>0 else'';raise ValueError(f"Processed prosumer data does not contain any usable episodes after split filtering: {history_msg}episode_length={self.episode_length}, total_steps={total_steps}.")
	def num_episodes(self)->int:return self._num_episodes
	def get_episode(self,episode_idx:int)->dict:
		if episode_idx<0 or episode_idx>=self._num_episodes:raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self._num_episodes-1}]")
		history_start,start,end=self._episode_slices[episode_idx];timestamps=self._timestamps.iloc[start:end].reset_index(drop=True);history_timestamps=self._timestamps.iloc[history_start:start].reset_index(drop=True);episode_signals={'wholesale_price':self._signals['wholesale_price'][start:end].copy(),'load':self._signals['load'][start:end,:].copy(),'pv':self._signals['pv'][start:end,:].copy()};history_signals={'wholesale_price':self._signals['wholesale_price'][history_start:start].copy(),'load':self._signals['load'][history_start:start,:].copy(),'pv':self._signals['pv'][history_start:start,:].copy()}
		for comp_name in self.load_components:
			comp_key=f"load_{comp_name}"
			if comp_key in self._signals:episode_signals[comp_key]=self._signals[comp_key][start:end,:].copy();history_signals[comp_key]=self._signals[comp_key][history_start:start,:].copy()
		return{'signals':episode_signals,'history_signals':history_signals,'history_timestamps':history_timestamps.astype(str).tolist(),'history_length':int(start-history_start),'meta':{'episode_idx':int(episode_idx),'segment_id':int(self.year),'segment_episode_idx':int(episode_idx),'timestamps':timestamps.astype(str).tolist(),'signal_names':['wholesale_price','load','pv'],**self._meta_template}}
