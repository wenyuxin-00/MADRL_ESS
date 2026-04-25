from __future__ import annotations
import re
from typing import Sequence
import numpy as np,pandas as pd
TIME_FEATURE_MODE_NONE,TIME_FEATURE_MODE_HOUR_WEEK_YEAR='none','hour_week_year'
DEFAULT_FORECAST_STEP,DEFAULT_LOCAL_TIMEZONE=pd.Timedelta(minutes=15),'Europe/Berlin'
_TZ_SUFFIX_PATTERN=re.compile('(Z|[+-]\\d{2}:?\\d{2})$')
def normalize_time_feature_mode(mode:str|None)->str:
	normalized=str(mode or TIME_FEATURE_MODE_NONE).strip().lower()
	if normalized not in{TIME_FEATURE_MODE_NONE,TIME_FEATURE_MODE_HOUR_WEEK_YEAR}:raise ValueError(f"Unsupported forecast time_feature_mode '{mode}'.")
	return normalized
def time_feature_dim(mode:str|None)->int:return 0 if normalize_time_feature_mode(mode)==TIME_FEATURE_MODE_NONE else 6
def coerce_timestamp_index(timestamps:Sequence[str|pd.Timestamp]|pd.DatetimeIndex|None)->pd.DatetimeIndex:
	if timestamps is None:return pd.DatetimeIndex([])
	if isinstance(timestamps,pd.DatetimeIndex):return timestamps
	values=list(timestamps)
	if not values:return pd.DatetimeIndex([])
	return pd.DatetimeIndex(pd.to_datetime(values,utc=any(_TZ_SUFFIX_PATTERN.search(value.strip())if isinstance(value,str)else getattr(value,'tzinfo',None)is not None for value in values)))
def infer_timestamp_step(timestamps:Sequence[str|pd.Timestamp]|pd.DatetimeIndex|None)->pd.Timedelta:
	index=coerce_timestamp_index(timestamps)
	if len(index)<2:return DEFAULT_FORECAST_STEP
	deltas_ns=np.diff(index.asi8);deltas_ns=deltas_ns[deltas_ns>0];return DEFAULT_FORECAST_STEP if deltas_ns.size==0 else pd.to_timedelta(int(np.median(deltas_ns)),unit='ns')
def encode_forecast_time_features(timestamps:Sequence[str|pd.Timestamp]|pd.DatetimeIndex,mode:str|None)->np.ndarray:
	normalized_mode=normalize_time_feature_mode(mode);index=coerce_timestamp_index(timestamps)
	if getattr(index,'tz',None)is not None:index=index.tz_convert(DEFAULT_LOCAL_TIMEZONE)
	if normalized_mode==TIME_FEATURE_MODE_NONE:return np.zeros((len(index),0),dtype=np.float32)
	if len(index)==0:return np.zeros((0,time_feature_dim(normalized_mode)),dtype=np.float32)
	hour_of_day=index.hour.to_numpy(dtype=np.float32)+index.minute.to_numpy(dtype=np.float32)/np.float32(6e1)+index.second.to_numpy(dtype=np.float32)/np.float32(36e2);day_of_week,day_of_year=index.dayofweek.to_numpy(dtype=np.float32)+hour_of_day/np.float32(24.),index.dayofyear.to_numpy(dtype=np.float32)-np.float32(1.)+hour_of_day/np.float32(24.);hour_phase,week_phase,year_phase=np.float32(2.*np.pi)*hour_of_day/np.float32(24.),np.float32(2.*np.pi)*day_of_week/np.float32(7.),np.float32(2.*np.pi)*day_of_year/np.float32(365.25);return np.column_stack([np.sin(hour_phase),np.cos(hour_phase),np.sin(week_phase),np.cos(week_phase),np.sin(year_phase),np.cos(year_phase)]).astype(np.float32,copy=False)
