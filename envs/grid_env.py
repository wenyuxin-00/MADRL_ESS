from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
try:import gymnasium as gym;from gymnasium import spaces
except ModuleNotFoundError as exc:raise ModuleNotFoundError("GridEnv requires 'gymnasium'. Install gymnasium==0.29.1 in the active environment instead of falling back to legacy Gym.")from exc
from predictors.shared_data import PrecomputedObservationStore
from controllers.madrl.safety_projector import build_safety_local_numpy,validate_executed_actions_numpy
from envs.grid.deployments import resolve_fixed_battery_spec
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY,WHOLESALE_PRICE_SIGNAL,derive_import_price,get_import_price_markup
class GridEnv(gym.Env):
	metadata={'render_modes':[]}
	def __init__(self,cfg:Any,mode:str='train',dataset:Any|None=None,reward_fn:Any|None=None,forecaster:Any|None=None,obs_builder:Any|None=None,grid_core:Any|None=None,precomputed_data_dir:str|Path|None=None)->None:
		super().__init__();self.cfg=cfg;self.mode=mode;env_cfg=cfg.env;self.n=int(env_cfg.num_agents);self.episode_length=int(env_cfg.episode_limit);self.future_horizon=int(env_cfg.future_horizon);self.eff=float(env_cfg.efficiency);self.init_soc=float(env_cfg.init_soc);self.dt=float(env_cfg.dt);fixed_capacity_kwh,_,fixed_p_max_kw=resolve_fixed_battery_spec(env_cfg.battery_capacity,env_cfg.max_charge_rate,n_agents=self.n);self._fixed_capacity_kwh=np.asarray(fixed_capacity_kwh,dtype=np.float32);self._fixed_p_max_kw=np.asarray(fixed_p_max_kw,dtype=np.float32);runtime_seed=getattr(getattr(cfg,'runtime',None),'seed',None);self._default_seed=None if runtime_seed is None else int(runtime_seed);self._seeded_once=False;self.soc_min=float(env_cfg.soc_min);self.soc_max=float(env_cfg.soc_max);self.soc_target=float(env_cfg.soc_target)
		if not .0<=self.soc_min<=self.soc_max<=1.:raise ValueError(f"Invalid SoC range: soc_min={self.soc_min}, soc_max={self.soc_max}.")
		self.init_soc=float(np.clip(self.init_soc,self.soc_min,self.soc_max));self._grid_cfg=cfg.grid;from envs.rewards.NormalReward import NormalReward;self.reward_fn=reward_fn if reward_fn is not None else NormalReward(cfg);self.import_price_markup_eur_per_kwh=get_import_price_markup(cfg)
		if dataset is None:from data.loaders.registry import build_dataset;dataset=build_dataset(cfg,mode=mode)
		self._dataset=dataset
		if forecaster is None and precomputed_data_dir is None:from predictors.registry import PerfectForecaster;forecaster=PerfectForecaster()
		self.forecaster=forecaster
		if obs_builder is None:from envs.observation.default_builder import DefaultObservationBuilder;from envs.observation.normalization import build_observation_normalizer;obs_builder=DefaultObservationBuilder(local_features=cfg.obs.local_features,sequence_features=cfg.obs.sequence_features,future_horizon=self.future_horizon,adjacency_type=cfg.obs.adjacency_type,normalizer=build_observation_normalizer(cfg))
		elif getattr(obs_builder,'normalizer',None)is None and bool(getattr(cfg.obs,'normalization_enabled',False)):from envs.observation.normalization import build_observation_normalizer;obs_builder.normalizer=build_observation_normalizer(cfg)
		self.obs_builder=obs_builder;self.observation_schema=self.obs_builder.get_schema(self.n);self.observation_layout=self.obs_builder.get_layout(self.n)
		if grid_core is None:raise ValueError('GridEnv requires an attached GridCore instance.')
		self._grid_core=grid_core;self.num_available_episodes=self._dataset.num_episodes();self.cur_step=0;self.soc=np.full((self.n,),self.init_soc,dtype=np.float32);(self.signals):dict[str,np.ndarray]={};(self.history_signals):dict[str,np.ndarray]={};(self.history_timestamps):list[str]=[];(self.episode_meta):dict[str,Any]={};self._precomputed_data_dir=None if precomputed_data_dir is None else Path(precomputed_data_dir).resolve();self._precomputed_store=None if self._precomputed_data_dir is None else PrecomputedObservationStore(self._precomputed_data_dir);(self._episode_precomputed):dict[str,np.ndarray]={};(self._local_mpc_solver_cache):dict[tuple[object,...],Any]={};self._apply_episode_storage_config();self.action_space=[spaces.Box(low=-1.,high=1.,shape=(2,),dtype=np.float32)for _ in range(self.n)];self.observation_space=spaces.Dict({key:spaces.Box(low=-np.inf,high=np.inf,shape=shape,dtype=np.float32)for(key,shape)in self.observation_schema.items()})
	def _apply_episode_storage_config(self)->None:self.agent_c_bat=self._fixed_capacity_kwh.copy();self.agent_p_max=self._fixed_p_max_kw.copy();self.agent_e_min=(self.soc_min*self.agent_c_bat).astype(np.float32);self.agent_e_max=(self.soc_max*self.agent_c_bat).astype(np.float32)
	def _load_episode(self,episode_idx:int)->None:
		episode_data=self._dataset.get_episode(episode_idx);raw_signals=episode_data.get('signals',{});raw_history_signals=dict(episode_data.get('history_signals',{}))
		if WHOLESALE_PRICE_SIGNAL not in raw_signals or'load'not in raw_signals:raise KeyError(f"Environment requires signals['{WHOLESALE_PRICE_SIGNAL}'] and signals['load'].")
		self.signals={}
		for(name,signal_value)in raw_signals.items():
			signal=np.asarray(signal_value,dtype=np.float32)
			if signal.shape[0]!=self.episode_length:raise ValueError(f"signal '{name}' first dimension should match episode_length={self.episode_length}, got {signal.shape[0]}")
			self.signals[name]=signal
		self.history_signals={}
		for(name,signal)in self.signals.items():
			raw_history=raw_history_signals.get(name)
			if raw_history is None:self.history_signals[name]=np.zeros((0,*signal.shape[1:]),dtype=np.float32);continue
			history=np.asarray(raw_history,dtype=np.float32)
			if history.ndim!=signal.ndim or signal.ndim>1 and history.shape[1:]!=signal.shape[1:]:raise ValueError(f"history signal '{name}' shape should match active signal trailing dimensions, got {history.shape} vs {signal.shape}")
			self.history_signals[name]=history.astype(np.float32,copy=False)
		self.episode_meta=dict(episode_data.get('meta',{}));self.history_timestamps=[str(value)for value in list(episode_data.get('history_timestamps')or[])];history_length=int(episode_data.get('history_length',len(self.history_timestamps)))
		if history_length!=len(self.history_timestamps):raise ValueError(f"history_length={history_length} should match history_timestamps size={len(self.history_timestamps)}")
		if history_length!=int(self.history_signals[WHOLESALE_PRICE_SIGNAL].shape[0]):raise ValueError(f"history_length should match history_signals length, got {history_length} vs {self.history_signals[WHOLESALE_PRICE_SIGNAL].shape[0]}")
		if self.get_signal(WHOLESALE_PRICE_SIGNAL).ndim!=1:raise ValueError(f"signals['{WHOLESALE_PRICE_SIGNAL}'] must have shape (T,), got {self.get_signal(WHOLESALE_PRICE_SIGNAL).shape}")
		if self.get_signal('load').shape!=(self.episode_length,self.n):raise ValueError(f"signals['load'] must have shape {(self.episode_length,self.n)}, got {self.get_signal('load').shape}")
		pv_signal=self.signals.get('pv',np.zeros((self.episode_length,self.n),dtype=np.float32))
		if np.asarray(pv_signal,dtype=np.float32).shape!=(self.episode_length,self.n):raise ValueError(f"signals['pv'] must have shape {(self.episode_length,self.n)}, got {np.asarray(pv_signal,dtype=np.float32).shape}")
		self._apply_episode_storage_config()
	def get_signal(self,signal_name:str)->np.ndarray:
		if signal_name not in self.signals:available=sorted(self.signals);raise KeyError(f"Current episode does not contain signal '{signal_name}'. Available: {available}")
		return self.signals[signal_name]
	def get_signal_step(self,signal_name:str,step:int|None=None)->Any:
		step=self.cur_step if step is None else int(step);signal=self.get_signal(signal_name);value=signal[step]
		if signal.ndim==1:return float(value)
		return np.asarray(value,dtype=np.float32)
	def reset(self,episode_idx:int|None=None,*,seed:int|None=None,options:dict[str,Any]|None=None)->tuple[dict[str,np.ndarray],dict[str,Any]]:
		if seed is None and not self._seeded_once:seed=self._default_seed
		super().reset(seed=seed)
		if seed is not None:self._seeded_once=True
		if options is not None and episode_idx is None:episode_idx=options.get('episode_idx')
		if episode_idx is None:episode_idx=int(self.np_random.integers(0,self.num_available_episodes))
		elif episode_idx<0 or episode_idx>=self.num_available_episodes:raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self.num_available_episodes-1}]")
		self._load_episode(int(episode_idx));self._episode_precomputed={}if self._precomputed_store is None else self._precomputed_store.episode(int(episode_idx));self.cur_step=0;self.soc=np.full((self.n,),self.init_soc,dtype=np.float32)
		if self.forecaster is not None:self.forecaster.reset();self.forecaster.set_episode({name:np.asarray(signal,dtype=np.float32).copy()if np.asarray(self.history_signals.get(name),dtype=np.float32).size==0 else np.concatenate([np.asarray(self.history_signals.get(name),dtype=np.float32),signal],axis=0).astype(np.float32,copy=False)for(name,signal)in self.signals.items()},self.episode_meta)
		pv_signal=self.signals.get('pv',np.zeros((self.episode_length,self.n),dtype=np.float32));self._grid_core.reset(np.asarray(self.get_signal('load')[0],dtype=np.float32),np.asarray(pv_signal[0],dtype=np.float32));return self.obs_builder.build(self),{'episode_idx':int(episode_idx),'battery_capacity_kwh':self.agent_c_bat.astype(np.float32).copy(),'p_max':self.agent_p_max.astype(np.float32).copy(),IMPORT_PRICE_MARKUP_KEY:float(self.import_price_markup_eur_per_kwh),'episode_meta':dict(self.episode_meta)}
	def _split_action_components(self,actions:list[np.ndarray])->tuple[np.ndarray,np.ndarray]:
		action_array=np.asarray(actions,dtype=np.float32).reshape(self.n,-1)
		if action_array.shape[1]!=2:raise ValueError(f"GridEnv expects exactly two action dimensions per agent: [battery_action, pv_action]. Got action shape {action_array.shape}.")
		if not np.all(np.isfinite(action_array)):raise ValueError('GridEnv received non-finite action values.')
		if np.any(action_array<-1.00001)or np.any(action_array>1.00001):raise ValueError('GridEnv received action values outside [-1, 1]. Controllers must output already-feasible normalized actions.')
		battery_action=action_array[:,0].astype(np.float32);pv_action=action_array[:,1].astype(np.float32);return battery_action,pv_action
	def _current_safety_local(self)->np.ndarray:return build_safety_local_numpy(soc=np.asarray(self.soc,dtype=np.float32),load_raw=np.asarray(self.get_signal_step('load'),dtype=np.float32),pv_raw=np.asarray(self.get_signal_step('pv'),dtype=np.float32),battery_capacity_kwh=np.asarray(self.agent_c_bat,dtype=np.float32),p_max_kw=np.asarray(self.agent_p_max,dtype=np.float32))
	def _apply_storage_dynamics(self,battery_action:np.ndarray)->dict[str,np.ndarray]:
		e_bat_req=np.asarray(battery_action,dtype=np.float32)*self.agent_p_max;soc_t=self.soc.copy().astype(np.float32);e_t=soc_t*self.agent_c_bat;eff=max(self.eff,1e-06);e_bat=e_bat_req.astype(np.float32);delta_e=np.where(e_bat>=.0,e_bat*eff,e_bat/eff)*self.dt;e_next=(e_t+delta_e).astype(np.float32);invalid_energy=np.logical_or(e_next<self.agent_e_min-1e-05,e_next>self.agent_e_max+1e-05)
		if np.any(invalid_energy):bad_agent=int(np.nonzero(invalid_energy)[0][0]);raise ValueError(f"GridEnv received a battery action that would leave the local energy bounds for agent {bad_agent}: next energy {e_next[bad_agent]:.4f} kWh, allowed range [{self.agent_e_min[bad_agent]:.4f}, {self.agent_e_max[bad_agent]:.4f}] kWh.")
		e_next=np.clip(e_next,self.agent_e_min,self.agent_e_max).astype(np.float32);soc_next=(e_next/self.agent_c_bat).astype(np.float32);return{'e_bat_req':e_bat_req.astype(np.float32),'e_bat':e_bat,'soc_t':soc_t,'soc_next':soc_next,'e_t':e_t.astype(np.float32),'e_next':e_next}
	def _build_signal_state(self,t:int,pv_action:np.ndarray)->dict[str,Any]:wholesale_price_t=float(self.get_signal_step(WHOLESALE_PRICE_SIGNAL,t));import_price_t=float(derive_import_price(wholesale_price_t,markup_eur_per_kwh=self.import_price_markup_eur_per_kwh));load_t=np.asarray(self.get_signal_step('load',t),dtype=np.float32);pv_signal=self.signals.get('pv');pv_raw=np.zeros((self.n,),dtype=np.float32)if pv_signal is None else np.asarray(pv_signal[t],dtype=np.float32);pv_effective_req=(pv_raw*(.5*(np.asarray(pv_action,dtype=np.float32)+1.))).astype(np.float32);base_net_load_raw=(load_t-pv_raw).astype(np.float32);pv_effective=np.clip(pv_effective_req,.0,pv_raw).astype(np.float32);pv_curtail=(pv_raw-pv_effective).astype(np.float32);pv_utilization=np.ones_like(pv_raw,dtype=np.float32);valid_mask=pv_raw>1e-06;pv_utilization[valid_mask]=(pv_effective[valid_mask]/pv_raw[valid_mask]).astype(np.float32);base_net_load_effective=(load_t-pv_effective).astype(np.float32);return{'wholesale_price_t':wholesale_price_t,'import_price_t':import_price_t,'load_t':load_t,'pv_raw':pv_raw,'pv_effective':pv_effective,'pv_curtail':pv_curtail,'pv_utilization':pv_utilization.astype(np.float32),'base_net_load_raw':base_net_load_raw,'base_net_load_effective':base_net_load_effective}
	def _run_power_flow(self,storage_state:dict[str,np.ndarray],signal_state:dict[str,Any])->tuple[Any,str]:pf_result=self._grid_core.step(p_batt_kw=storage_state['e_bat'],base_load_kw=signal_state['base_net_load_effective']);pf_error=getattr(self._grid_core,'last_pf_error','');signal_state['net_load']=(np.asarray(signal_state['base_net_load_effective'],dtype=np.float32)+storage_state['e_bat']).astype(np.float32);signal_state['grid_import_kw']=np.maximum(signal_state['net_load'],.0).astype(np.float32);signal_state['grid_export_kw']=np.maximum(-signal_state['net_load'],.0).astype(np.float32);return pf_result,pf_error
	def step(self,actions:list[np.ndarray])->tuple[dict[str,np.ndarray],list[float],list[bool],list[bool],dict[str,Any]]:
		t=self.cur_step;battery_action,pv_action=self._split_action_components(actions);validate_executed_actions_numpy(self._current_safety_local(),np.column_stack([battery_action,pv_action]),efficiency=self.eff,dt_hours=self.dt,soc_min=self.soc_min,soc_max=self.soc_max);storage_state=self._apply_storage_dynamics(battery_action);signal_state=self._build_signal_state(t,pv_action);pf_result,pf_error=self._run_power_flow(storage_state,signal_state);p_max=self.agent_p_max.astype(np.float32);e_min=self.agent_e_min.astype(np.float32);e_max=self.agent_e_max.astype(np.float32);battery_capacity_kwh=self.agent_c_bat.astype(np.float32);reward_state={'import_price_t':signal_state['import_price_t'],'storage_price_t':signal_state['import_price_t'],'battery_power_t':storage_state['e_bat'],'actual_grid_power_t':signal_state['net_load'],'dt':self.dt,'v_violation':pf_result.v_violation,'psi_v_raw':float(pf_result.psi_v_raw),'psi_line_raw':float(pf_result.psi_line_raw),'psi_trafo_raw':float(pf_result.psi_trafo_raw)};reward_per_agent,components=self.reward_fn.compute(reward_state);reward=np.asarray(reward_per_agent,dtype=np.float32);self.soc=storage_state['soc_next'];self.cur_step+=1;done=self.cur_step>=self.episode_length;terminated_n=[False]*self.n;truncated_n=[done]*self.n;obs=self.obs_builder.zeros(self.n)if done else self.obs_builder.build(self)
		if self.mode=='train':
			info:dict[str,Any]={'episode_done':bool(done)}
			for(key,value)in components.items():info[str(key)]=np.asarray(value,dtype=np.float32)
			return obs,reward.tolist(),terminated_n,truncated_n,info
		pv_action_exec=(2.*signal_state['pv_utilization']-1.).astype(np.float32);line_limit=float(self._grid_cfg.line_max_loading_pct);info={'episode_done':done,'t':int(t),'wholesale_price':float(signal_state['wholesale_price_t']),'import_price':float(signal_state['import_price_t']),'load':signal_state['load_t'].astype(np.float32),'pv':signal_state['pv_raw'].astype(np.float32),'pv_effective':signal_state['pv_effective'].astype(np.float32),'pv_curtail':signal_state['pv_curtail'].astype(np.float32),'pv_utilization':signal_state['pv_utilization'].astype(np.float32),'pv_action_req':pv_action.astype(np.float32),'pv_action':pv_action_exec,'net_load':signal_state['net_load'].astype(np.float32),'grid_import_kw':signal_state['grid_import_kw'].astype(np.float32),'grid_export_kw':signal_state['grid_export_kw'].astype(np.float32),'e_bat_req':storage_state['e_bat_req'].astype(np.float32),'e_bat':storage_state['e_bat'].astype(np.float32),'p_max':p_max,'e_min':e_min,'e_max':e_max,'battery_capacity_kwh':battery_capacity_kwh,'soc_min':float(self.soc_min),'soc_max':float(self.soc_max),'soc_t':storage_state['soc_t'],'soc_next':storage_state['soc_next'],**components,'pf_converged':bool(pf_result.converged),'pf_error':pf_error,'vm_pu':pf_result.vm_pu,'line_loading_pct':pf_result.line_loading_pct,'trafo_loading_pct':pf_result.trafo_loading_pct,'trafo_p_signed_kw':pf_result.trafo_p_signed_kw,'v_violation':pf_result.v_violation,'n_v_violations':int(np.sum(pf_result.v_violation>.0)),'n_line_violations':int(np.any(np.asarray(pf_result.line_loading_pct,dtype=np.float32)>line_limit)),'n_trafo_violations':int(np.any(np.asarray(pf_result.trafo_loading_pct,dtype=np.float32)>line_limit)),'psi_v_raw':float(pf_result.psi_v_raw),'psi_line_raw':float(pf_result.psi_line_raw),'psi_trafo_raw':float(pf_result.psi_trafo_raw)};return obs,reward.tolist(),terminated_n,truncated_n,info
	def _cleanup_gurobi_cache(self)->None:
		for solver in list(self._local_mpc_solver_cache.values()):
			dispose=getattr(solver,'dispose',None)
			if callable(dispose):dispose()
		self._local_mpc_solver_cache.clear()
	def close(self)->None:self._cleanup_gurobi_cache()
