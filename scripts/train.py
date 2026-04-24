from __future__ import annotations
import os,time
from datetime import datetime
from typing import Any
import numpy as np,torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from controllers.madrl.base_agent import get_agent_cls
from controllers.madrl.safety_projector import action_info_to_numpy,compute_action_gap_metrics_torch,enforce_local_action_feasibility_torch,map_actor_output_to_soc_feasible_action_torch,merge_action_info_into_step_info,sample_feasible_random_battery_action_torch
from controllers.madrl.safety_projector import SAFE_POC_ALGO_NAME
from controllers.madrl_controller import _override_soc_penalty_metrics
from scripts.checkpoints import build_checkpoint_manifest,build_training_contract,build_training_contract_signature,write_checkpoint_manifest
from scripts.utils.project_paths import get_tensorboard_run_dir
from scripts.utils.torch_runtime import to_torch_nested
from scripts.utils.replay_buffer import ReplayBuffer
_PROJECTION_RESIDUAL_TOL=1e-06
def _iso_timestamp(value:datetime)->str:return value.astimezone().isoformat(timespec='seconds')
def _estimate_remaining_seconds(*,interaction_step:int,target_interactions:int,elapsed_seconds:float)->float|None:
	remaining_interactions=max(int(target_interactions)-int(interaction_step),0)
	if remaining_interactions==0:return .0
	if interaction_step<=0 or elapsed_seconds<=.0:return
	return float(remaining_interactions*(elapsed_seconds/float(interaction_step)))
def init_safety_tracking(runner:Any)->None:runner._safety_projection_stats={stage:{'calls':0,'time_s':.0}for stage in('rollout','target','actor')};runner._safety_diag={'samples':0,'projected_fraction':.0,'pre_trafo_import_violation_kw':.0,'pre_trafo_export_violation_kw':.0,'post_trafo_import_violation_kw':.0,'post_trafo_export_violation_kw':.0,'max_abs_delta':.0};runner._projector_local_infeasible_count=0
def _safe_projection_enabled(runner:Any)->bool:algo_cfg=getattr(runner.cfg,'algo',None);algo_name=getattr(algo_cfg,'name','');return str(algo_name)==SAFE_POC_ALGO_NAME and getattr(runner,'safety_projector',None)is not None
def _record_projection_event(runner:Any,*,stage:str,batch_size:int,elapsed_s:float,diagnostics:dict[str,Any]|None=None)->None:
	normalized_stage=str(stage)if str(stage)in runner._safety_projection_stats else'rollout';runner._safety_projection_stats[normalized_stage]['calls']+=1;runner._safety_projection_stats[normalized_stage]['time_s']+=float(elapsed_s)
	if diagnostics is not None:
		weighted_batch=max(int(diagnostics.get('batch_size',batch_size)),1);runner._safety_diag['samples']+=weighted_batch
		for field_name in('projected_fraction','pre_trafo_import_violation_kw','pre_trafo_export_violation_kw','post_trafo_import_violation_kw','post_trafo_export_violation_kw'):runner._safety_diag[field_name]+=float(diagnostics.get(field_name,.0))*weighted_batch
		runner._safety_diag['max_abs_delta']=max(runner._safety_diag['max_abs_delta'],float(diagnostics.get('max_abs_delta',.0)))
def _record_projector_local_infeasible(runner:Any,residual_action_info:dict[str,torch.Tensor]|None)->None:
	if residual_action_info is None or(residual:=residual_action_info.get('soc_penalty_unweighted'))is None:return
	residual_tensor=torch.as_tensor(residual,dtype=torch.float32)
	if residual_tensor.ndim==1:residual_tensor=residual_tensor.unsqueeze(0)
	affected=torch.any(residual_tensor>_PROJECTION_RESIDUAL_TOL,dim=-1);runner._projector_local_infeasible_count+=int(torch.count_nonzero(affected).item())
def _feasibility_kwargs(cfg:Any)->dict[str,float]:return{'efficiency':float(cfg.env.efficiency),'dt_hours':float(cfg.env.dt),'soc_min':float(cfg.env.soc_min),'soc_max':float(cfg.env.soc_max)}
def _merge_action_info(*parts:dict[str,torch.Tensor]|None)->dict[str,torch.Tensor]|None:
	merged:dict[str,torch.Tensor]={}
	for part in parts:
		if part is not None:merged.update(part)
	return merged or None
def _apply_feasible_random_exploration(runner:Any,safety_local:torch.Tensor,mapped_action_t:torch.Tensor)->tuple[torch.Tensor,dict[str,torch.Tensor]]:
	epsilon=float(runner.cfg.train.resolved_feasible_random_exploration(runner.total_steps))
	mask=torch.rand(mapped_action_t.shape[:-1],device=mapped_action_t.device)<epsilon
	random_action_t,random_info=sample_feasible_random_battery_action_torch(safety_local,mapped_action_t,**_feasibility_kwargs(runner.cfg))
	explored_action_t=torch.where(mask.unsqueeze(-1),random_action_t,mapped_action_t)
	info={'feasible_random_exploration_epsilon':torch.full(mask.shape,epsilon,dtype=mapped_action_t.dtype,device=mapped_action_t.device),'feasible_random_exploration_source':mask.to(dtype=mapped_action_t.dtype)}
	info.update(random_info)
	return explored_action_t,info
def build_safety_summary(runner:Any)->dict[str,Any]:
	if not _safe_projection_enabled(runner):return{'enabled':False,'algorithm':str(runner.cfg.algo.name)}
	diagnostic_denominator=max(int(runner._safety_diag['samples']),1);projection_time_total=float(sum(stage_stats['time_s']for stage_stats in runner._safety_projection_stats.values()));return{'enabled':True,'algorithm':str(runner.cfg.algo.name),'projector_mode':str(getattr(runner.cfg.safety,'projector_mode','joint_linearized')),'projection_batches':int(sum(stage_stats['calls']for stage_stats in runner._safety_projection_stats.values())),'projected_fraction':float(runner._safety_diag['projected_fraction']/diagnostic_denominator),'max_abs_action_delta':float(runner._safety_diag['max_abs_delta']),'mean_pre_trafo_import_violation_kw':float(runner._safety_diag['pre_trafo_import_violation_kw']/diagnostic_denominator),'mean_pre_trafo_export_violation_kw':float(runner._safety_diag['pre_trafo_export_violation_kw']/diagnostic_denominator),'mean_post_trafo_import_violation_kw':float(runner._safety_diag['post_trafo_import_violation_kw']/diagnostic_denominator),'mean_post_trafo_export_violation_kw':float(runner._safety_diag['post_trafo_export_violation_kw']/diagnostic_denominator),'projection_time_s':projection_time_total,'rollout_projection_calls':int(runner._safety_projection_stats['rollout']['calls']),'target_projection_calls':int(runner._safety_projection_stats['target']['calls']),'actor_projection_calls':int(runner._safety_projection_stats['actor']['calls']),'rollout_projection_time_s':float(runner._safety_projection_stats['rollout']['time_s']),'target_projection_time_s':float(runner._safety_projection_stats['target']['time_s']),'actor_projection_time_s':float(runner._safety_projection_stats['actor']['time_s']),'projector_local_infeasible_count':int(runner._projector_local_infeasible_count)}
def select_action_batch_with_info(runner:Any,obs_np:dict)->tuple[np.ndarray,dict[str,np.ndarray]|None]:
	obs_t=to_torch_nested(obs_np,runner.cfg.runtime.device)
	with torch.inference_mode():
		raw_action_t=torch.stack([agent.act_from_torch_obs(obs_t,noise_std=runner.noise_std)for agent in runner.agent_n],dim=1);action_info=None
		if'safety_local'in obs_t:
			mapped_action_t,mapping_info=map_actor_output_to_soc_feasible_action_torch(obs_t['safety_local'],raw_action_t,**_feasibility_kwargs(runner.cfg))
			rollout_action_t,exploration_info=_apply_feasible_random_exploration(runner,obs_t['safety_local'],mapped_action_t)
			if _safe_projection_enabled(runner):projection_started=time.perf_counter();projected_action_t,diagnostics=runner.safety_projector.project_actions_from_safety_local(obs_t['safety_local'],rollout_action_t,return_diagnostics=True);_record_projection_event(runner,stage='rollout',batch_size=int(raw_action_t.shape[0]),elapsed_s=time.perf_counter()-projection_started,diagnostics=diagnostics);action_t,projector_residual_info=enforce_local_action_feasibility_torch(obs_t['safety_local'],projected_action_t,**_feasibility_kwargs(runner.cfg));_record_projector_local_infeasible(runner,projector_residual_info);gap_info=compute_action_gap_metrics_torch(obs_t['safety_local'],projected_action_t,action_t);action_info=_merge_action_info(_override_soc_penalty_metrics(gap_info,projector_residual_info),mapping_info,exploration_info)
			else:action_t,residual_info=enforce_local_action_feasibility_torch(obs_t['safety_local'],rollout_action_t,**_feasibility_kwargs(runner.cfg));gap_info=compute_action_gap_metrics_torch(obs_t['safety_local'],rollout_action_t,action_t);action_info=_merge_action_info(_override_soc_penalty_metrics(gap_info,residual_info),mapping_info,exploration_info)
		else:action_t=raw_action_t
	return action_t.to(dtype=torch.float32).cpu().numpy(),action_info_to_numpy(action_info)
def apply_controller_action_postprocessing(runner:Any,reward:np.ndarray,info_list:list[dict[str,Any]],action_info:dict[str,np.ndarray]|None)->tuple[np.ndarray,list[dict[str,Any]]]:
	reward_array=np.asarray(reward,dtype=np.float32)
	if reward_array.ndim==2:reward_array=reward_array[...,None]
	processed_info_list:list[dict[str,Any]]=[]
	for(env_idx,info)in enumerate(info_list):
		env_action_info=None if action_info is None else{key:np.asarray(value[env_idx],dtype=np.float32)for(key,value)in action_info.items()};merged_info=merge_action_info_into_step_info(info,env_action_info);reward_vector=np.asarray(reward_array[env_idx,:,0],dtype=np.float32)
		merged_info['reward']=reward_vector
		processed_info_list.append(merged_info)
	return reward_array.astype(np.float32),processed_info_list
def build_replay_next_obs(next_obs:dict[str,np.ndarray],info_list:list[dict[str,Any]])->dict[str,np.ndarray]:
	replay_next_obs={key:np.asarray(value,dtype=np.float32).copy()for(key,value)in next_obs.items()}
	for env_idx,info in enumerate(info_list):
		bootstrap_obs=info.get('bootstrap_obs')
		if bootstrap_obs is None:continue
		for key,value in dict(bootstrap_obs).items():
			if key not in replay_next_obs:raise KeyError(f"bootstrap_obs contains unexpected observation key '{key}'.")
			replay_next_obs[key][env_idx]=np.asarray(value,dtype=np.float32)
	return replay_next_obs
def train_env_episode_length(train_env:Any)->int:
	value=getattr(train_env,'episode_length',None)
	if value is None:raise AttributeError(f"{type(train_env).__name__} must expose episode_length for TrainRunner; rebuild the vector env metadata instead of falling back to cfg.env.episode_limit.")
	length=int(value)
	if length<=0:raise ValueError(f"{type(train_env).__name__}.episode_length must be positive, got {length}.")
	return length
def build_shared_update_ctx(runner:Any,batch:dict[str,Any])->dict[str,Any]:
	next_obs=batch['next_obs']
	with torch.no_grad():target_actor_actions_raw=torch.stack([agent._actor_target_call(next_obs)for agent in runner.agent_n],dim=1)
	shared_ctx={'target_actor_actions_raw':target_actor_actions_raw,'batch_size':int(batch['action'].shape[0]),'device':batch['action'].device}
	if _safe_projection_enabled(runner):
		shared_ctx['safety_projector']=runner.safety_projector;shared_ctx['projection_event_recorder']=lambda**kwargs:_record_projection_event(runner,**kwargs)
	return shared_ctx
def save_runner_model(runner:Any,model_dir:str,episode:int)->None:
	algo_dir=os.path.join(model_dir,runner.cfg.algo.name);os.makedirs(algo_dir,exist_ok=True)
	for agent in runner.agent_n:agent.save_model(algo_dir,episode)
	manifest=build_checkpoint_manifest(algorithm=runner.cfg.algo.name,saved_episode_tag=episode,episodes_completed=runner.episodes_completed,total_steps=runner.total_steps,num_envs=runner.cfg.train.num_envs,episode_limit=train_env_episode_length(runner.env),save_dir=algo_dir,training_contract=build_training_contract(runner.cfg));manifest['target_total_steps']=int(dict(getattr(runner,'perf_summary',{})or{}).get('target_total_steps',runner.total_steps));manifest['global_interaction_step']=int(dict(getattr(runner,'perf_summary',{})or{}).get('global_interaction_step',runner.total_steps));write_checkpoint_manifest(algo_dir,manifest)
def close_runner(runner:Any)->None:
	if runner._closed:return
	runner.env.close();runner.env_evaluate.close();runner.writer.close();runner._closed=True
def build_training_health_summary(runner:Any)->dict[str,Any]:
	agent_health=[dict(getattr(agent,'training_health',{})or{})for agent in getattr(runner,'agent_n',[])]
	return{'agents':agent_health,'nonfinite_failure_count':int(sum(int(item.get('nonfinite_failure_count',0))for item in agent_health)),'actor_gradient_collapse_agents':[int(item.get('agent_id',idx))for(idx,item)in enumerate(agent_health)if bool(item.get('actor_gradient_collapse_flag',False))]}
def build_reward_summary(runner:Any)->dict[str,Any]:
	episode_indices=list(range(1,len(runner.episode_rewards)+1));components={str(meta.key):{'label':str(getattr(meta,'label',meta.key)),'color':str(getattr(meta,'color','#111827')),'sign':int(getattr(meta,'sign',0)),'values':[float(value)for value in runner.episode_reward_components.get(str(meta.key),[])]}for meta in runner.reward_component_meta};aggregates:dict[str,list[float]]={};grid_safety_keys=[str(meta.key)for meta in runner.reward_component_meta if str(meta.key).startswith('madrl_r_safe_')and not str(meta.key).endswith('_total')and int(getattr(meta,'sign',0))==-1]
	if grid_safety_keys and all(key in components for key in grid_safety_keys):aggregates['grid_safety_penalty']=[float(sum(runner.episode_reward_components[key][episode_idx]for key in grid_safety_keys))for episode_idx in range(len(episode_indices))]
	if getattr(runner,'_episode_throughput_bonus_weight_mean',None):aggregates['madrl_throughput_bonus_weight_mean']=[float(value)for value in runner._episode_throughput_bonus_weight_mean]
	if getattr(runner,'_episode_throughput_kwh_total',None):aggregates['throughput_kwh_total']=[float(value)for value in runner._episode_throughput_kwh_total]
	training_contract=build_training_contract(runner.cfg)
	return{'episodes':episode_indices,'episode_total_reward':[float(value)for value in runner.episode_rewards],'components':components,'aggregates':aggregates,'madrl_throughput_bonus_weight_final':float(getattr(runner,'_last_throughput_bonus_weight',0.0)),'training_health':build_training_health_summary(runner),'training_contract':training_contract,'training_contract_signature':build_training_contract_signature(runner.cfg)}
class TrainRunner:
	def __init__(self,cfg:Any,train_env:Any,eval_env:Any,env_name:str='GridEnv',number:int=1,seed:int=0)->None:self.cfg=cfg;self.seed=int(seed);self.env_name=env_name;self.env=train_env;self.env_evaluate=eval_env;agent_cls=get_agent_cls(self.cfg.algo.name);self.agent_n=[agent_cls(cfg,agent_id)for agent_id in range(self.cfg.env.num_agents)];self.safety_projector=getattr(self.agent_n[0],'safety_projector',None)if self.agent_n else None;self.replay_buffer=ReplayBuffer(self.cfg);log_dir=get_tensorboard_run_dir(algorithm=self.cfg.algo.name,env_name=env_name,run_number=number,seed=seed);log_dir.mkdir(parents=True,exist_ok=True);self.writer=SummaryWriter(log_dir=str(log_dir));self.vec_env_name=type(self.env).__name__;(self.history):list[Any]=[];(self.episode_rewards):list[float]=[];self.total_steps=self.episodes_completed=0;self.noise_std=float(self.cfg.train.noise_std_init);(self.perf_summary):dict[str,Any]={};(self.run_metadata):dict[str,Any]={};(self.reward_component_meta):list[Any]=[];(self.episode_reward_components):dict[str,list[float]]={};(self._episode_throughput_bonus_weight_mean):list[float]=[];(self._episode_throughput_kwh_total):list[float]=[];self._last_throughput_bonus_weight=0.;init_safety_tracking(self);self._closed=False
	def select_action_batch(self,obs_np:dict)->np.ndarray:return self._select_action_batch_with_info(obs_np)[0]
	_apply_controller_action_postprocessing=apply_controller_action_postprocessing;_build_replay_next_obs=staticmethod(build_replay_next_obs);_build_shared_update_ctx=build_shared_update_ctx;_select_action_batch_with_info=select_action_batch_with_info;build_reward_summary=build_reward_summary;build_safety_summary=build_safety_summary;build_training_health_summary=build_training_health_summary;close=close_runner;save_model=save_runner_model
	def run(self)->int:
		train_episode_limit=train_env_episode_length(self.env);target_total_steps=self.cfg.train.resolved_max_train_steps(train_episode_limit);target_interactions=target_total_steps//self.cfg.train.num_envs;learning_starts=int(self.cfg.train.resolved_learning_starts_transitions(train_episode_limit));actor_learning_starts=int(self.cfg.train.resolved_actor_learning_starts_transitions(train_episode_limit));interaction_step=episodes_completed=update_calls=0;noise_decay=float(self.cfg.train.resolved_noise_std_decay());started_at=datetime.now().astimezone();run_start=time.perf_counter();action_time_total=env_step_time_total=update_time_total=.0;reward_metas=list(self.env_evaluate.reward_fn.component_meta);self.reward_component_meta=reward_metas;self.episode_reward_components={str(meta.key):[]for meta in reward_metas};progress_episode_interval=max(1,int(getattr(self.cfg.train,'progress_episode_interval',10)));active_episode_rewards=np.zeros(self.cfg.train.num_envs,dtype=np.float32);active_component_totals={str(meta.key):np.zeros(self.cfg.train.num_envs,dtype=np.float32)for meta in reward_metas};active_throughput_kwh=np.zeros(self.cfg.train.num_envs,dtype=np.float32);active_bonus_weight_sum=np.zeros(self.cfg.train.num_envs,dtype=np.float32);active_bonus_weight_steps=np.zeros(self.cfg.train.num_envs,dtype=np.int32);progress=tqdm(total=max(target_total_steps,1),desc='Training',unit='step',disable=not bool(getattr(self.cfg.train,'show_progress',True)));last_progress_emit_episode,next_progress_episode_mark=-1,progress_episode_interval
		def emit_progress_advance()->bool:
			delta=int(self.total_steps)-int(getattr(progress,'n',0))
			if delta<=0:return False
			progress.update(delta);return True
		def emit_progress_postfix(*,force:bool=False,refresh:bool=False)->bool:
			nonlocal last_progress_emit_episode,next_progress_episode_mark
			if episodes_completed<=0:return False
			if not force and episodes_completed<next_progress_episode_mark:return False
			if last_progress_emit_episode==episodes_completed:return False
			avg_reward=float(np.mean(self.episode_rewards[-50:]))if self.episode_rewards else .0;elapsed_seconds=max(time.perf_counter()-run_start,.0);remaining_seconds=_estimate_remaining_seconds(interaction_step=self.total_steps,target_interactions=target_total_steps,elapsed_seconds=elapsed_seconds)
			last_progress_emit_episode=episodes_completed
			while episodes_completed>=next_progress_episode_mark:next_progress_episode_mark+=progress_episode_interval
			progress.set_postfix({'avg_reward':f"{avg_reward:.2f}",'steps/s':f"{self.total_steps/max(time.perf_counter()-run_start,1e-06):.1f}",'act_ms':f"{1e3*action_time_total/max(interaction_step,1):.2f}",'env_ms':f"{1e3*env_step_time_total/max(interaction_step,1):.2f}",'upd_ms':f"{1e3*update_time_total/max(update_calls,1):.2f}",'eta':'--'if remaining_seconds is None else f"{remaining_seconds:.1f}s"},refresh=refresh);return True
		try:
			obs,_=self.env.reset()
			while interaction_step<target_interactions:
				action_start=time.perf_counter();action_batch,action_info=self._select_action_batch_with_info(obs);action_time_total+=time.perf_counter()-action_start;env_step_start=time.perf_counter();next_obs,reward,terminated,truncated,info_list=self.env.step([action_batch[:,agent_id].copy()for agent_id in range(self.cfg.env.num_agents)]);reward,info_list=self._apply_controller_action_postprocessing(reward,info_list,action_info);terminated=np.asarray(terminated,dtype=np.float32);truncated=np.asarray(truncated,dtype=np.float32);env_step_time_total+=time.perf_counter()-env_step_start
				for(env_idx,info)in enumerate(info_list):
					step_total=float(np.sum(reward[env_idx]));active_episode_rewards[env_idx]+=step_total
					for meta in reward_metas:component_key=str(meta.key);component_value=float(np.sum(np.asarray(info[meta.key],dtype=np.float32)));active_component_totals[component_key][env_idx]+=float(meta.sign)*component_value
					active_throughput_kwh[env_idx]+=float(np.sum(np.asarray(info.get('madrl_throughput_kwh',0.),dtype=np.float32)))
					bonus_weight=float(info.get('madrl_throughput_bonus_weight',0.));active_bonus_weight_sum[env_idx]+=bonus_weight;active_bonus_weight_steps[env_idx]+=1;self._last_throughput_bonus_weight=bonus_weight
				replay_next_obs=self._build_replay_next_obs(next_obs,info_list);self.replay_buffer.store_transitions_batched(obs,action_batch,reward,replay_next_obs,terminated,truncated);obs=next_obs;interaction_step+=1;self.total_steps+=self.cfg.train.num_envs
				completed_episodes_this_step=0
				for(env_idx,info)in enumerate(info_list):
					if not bool(info.get('episode_done',False)):continue
					episode_reward=float(active_episode_rewards[env_idx]);self.episode_rewards.append(episode_reward)
					for meta in reward_metas:component_key=str(meta.key);self.episode_reward_components[component_key].append(float(active_component_totals[component_key][env_idx]))
					bonus_weight_mean=float(active_bonus_weight_sum[env_idx]/max(int(active_bonus_weight_steps[env_idx]),1));self._episode_throughput_bonus_weight_mean.append(bonus_weight_mean);self._episode_throughput_kwh_total.append(float(active_throughput_kwh[env_idx]));self.writer.add_scalar('train_episode_total_reward',episode_reward,global_step=self.total_steps);active_episode_rewards[env_idx]=.0;active_throughput_kwh[env_idx]=.0;active_bonus_weight_sum[env_idx]=.0;active_bonus_weight_steps[env_idx]=0
					for meta in reward_metas:active_component_totals[str(meta.key)][env_idx]=.0
					episodes_completed+=1;self.episodes_completed=episodes_completed;completed_episodes_this_step+=1
				if completed_episodes_this_step>0:emit_progress_postfix();emit_progress_advance()
				if self.cfg.train.use_noise_decay:self.noise_std=max(self.noise_std-noise_decay,float(self.cfg.train.noise_std_min))
				if self.total_steps>=learning_starts and self.replay_buffer.current_size>=self.cfg.train.batch_size and interaction_step%self.cfg.train.update_interval==0:
					update_start=time.perf_counter()
					for _ in range(self.cfg.train.updates_per_step):
						batch_torch=self.replay_buffer.sample_torch(self.cfg.runtime.device,pin_memory=bool(getattr(self.cfg.runtime,'pin_memory',False)),non_blocking=bool(getattr(self.cfg.runtime,'non_blocking_transfers',False)));shared_update_ctx=self._build_shared_update_ctx(batch_torch)
						shared_update_ctx['allow_actor_update']=bool(self.total_steps>=actor_learning_starts)
						for agent in self.agent_n:agent.train_on_batch(batch_torch,self.agent_n,shared_ctx=shared_update_ctx)
						update_calls+=1
					update_time_total+=time.perf_counter()-update_start
		finally:
			emitted_postfix=emit_progress_postfix(force=True)
			advanced_progress=emit_progress_advance()
			if emitted_postfix and not advanced_progress and hasattr(progress,'refresh'):progress.refresh()
			progress.close()
		finished_at=datetime.now().astimezone();total_elapsed=max(time.perf_counter()-run_start,1e-06);self.run_metadata={'started_at':_iso_timestamp(started_at),'finished_at':_iso_timestamp(finished_at),'elapsed_seconds':float(round(total_elapsed,3)),'estimated_end_time':_iso_timestamp(finished_at)};self.perf_summary={'seed':self.seed,'runtime_mode':str(self.cfg.runtime.execution_mode),'device':str(self.cfg.runtime.device),'vec_env':self.vec_env_name,'target_total_steps':int(target_total_steps),'global_interaction_step':int(self.total_steps),'learning_starts_transitions':int(learning_starts),'actor_learning_starts_transitions':int(actor_learning_starts),'n_step_return':int(self.cfg.train.n_step_return),'feasible_random_exploration_start':float(self.cfg.train.feasible_random_exploration_start),'feasible_random_exploration_end':float(self.cfg.train.feasible_random_exploration_end),'feasible_random_exploration_decay_steps':int(self.cfg.train.feasible_random_exploration_decay_steps),'total_wall_time_s':total_elapsed,'action_time_s':action_time_total,'env_step_time_s':env_step_time_total,'update_time_s':update_time_total,'sample_time_s':.0,'history_time_s':.0,'agent_update_time_s':.0,'update_calls':update_calls,'steps_per_sec':self.total_steps/total_elapsed,'avg_action_ms_per_iter':1e3*action_time_total/max(interaction_step,1),'avg_env_ms_per_iter':1e3*env_step_time_total/max(interaction_step,1),'avg_update_ms_per_call':1e3*update_time_total/max(update_calls,1),'projection_time_s':float(sum(stage_stats['time_s']for stage_stats in self._safety_projection_stats.values())),'rollout_projection_time_s':float(self._safety_projection_stats['rollout']['time_s']),'target_projection_time_s':float(self._safety_projection_stats['target']['time_s']),'actor_projection_time_s':float(self._safety_projection_stats['actor']['time_s']),'rollout_projection_calls':int(self._safety_projection_stats['rollout']['calls']),'target_projection_calls':int(self._safety_projection_stats['target']['calls']),'actor_projection_calls':int(self._safety_projection_stats['actor']['calls']),'training_health':build_training_health_summary(self)};self.episodes_completed=episodes_completed;return episodes_completed
