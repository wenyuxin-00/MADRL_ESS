from __future__ import annotations
import argparse
from copy import deepcopy
import json,os,subprocess,sys,warnings
from pathlib import Path
from typing import Any
for env_var in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(env_var,'1')
warnings.filterwarnings('ignore',message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*',category=FutureWarning)
import torch
from configs.profiles import compose_experiment_config,summarize_experiment
from controllers import MADRLController
from controllers.madrl.base_agent import get_agent_cls
from controllers.madrl.safety_projector import is_safe_poc_algorithm
from data.loaders.registry import resolve_train_episode_limit
from scripts.builder import build_env
from scripts.builder import build_train_runner
from scripts.checkpoints import ACTOR_ACTION_MAPPING_CONTRACT,TRAINING_HEALTH_CONTRACT,build_training_contract,build_training_contract_signature,build_training_run_paths,find_latest_training_run,resolve_checkpoint_to_load,slugify_checkpoint_token
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings as apply_mainline_experiment_settings,ensure_forecast_ready as ensure_mainline_forecast_ready
from scripts.utils.project_paths import get_checkpoint_root,get_tensorboard_run_dir,project_root
from scripts.utils.torch_runtime import configure_torch_runtime,describe_device,resolve_device
def _apply_train_controls(cfg,train_controls:dict[str,Any])->None:
	field_names='train_episodes','max_train_steps','num_envs','vec_env_type','parallel_episode_sampling','batch_size','buffer_size','update_interval','updates_per_step','learning_starts_transitions','actor_learning_starts_transitions','n_step_return','feasible_random_exploration_start','feasible_random_exploration_end','feasible_random_exploration_decay_steps','actor_lr','critic_lr','noise_std_init','noise_std_min','noise_decay_steps','use_noise_decay','show_progress','progress_postfix_interval','progress_episode_interval','progress_write_interval_seconds'
	for field_name in field_names:
		if field_name in train_controls:setattr(cfg.train,field_name,train_controls[field_name])
	for field_name in('train_window_days','window_stride_days'):
		if field_name in train_controls:
			value=int(train_controls[field_name])
			if value<=0:raise ValueError(f"train_controls.{field_name} must be positive, got {value}.")
			setattr(cfg.env,field_name,value)
	if'policy_update_freq'in train_controls:cfg.algo.policy_update_freq=int(train_controls['policy_update_freq'])
	if'discount_gamma'in train_controls:cfg.algo.gamma=float(train_controls['discount_gamma'])
def _apply_model_controls(cfg,model_controls:dict[str,Any]|None)->None:
	controls=dict(model_controls or{})
	if'hidden_dim'not in controls:return
	hidden_dim=int(controls['hidden_dim'])
	if hidden_dim<=0:raise ValueError(f"model_controls.hidden_dim must be positive, got {hidden_dim}.")
	cfg.model.hidden_dim=hidden_dim
def _apply_runtime_controls(cfg,runtime_controls:dict[str,Any]|None)->None:
	controls=dict(runtime_controls or{});bool_fields='pin_memory','non_blocking_transfers','enable_amp','enable_compile','compile_fullgraph','compile_dynamic';str_fields='amp_dtype','compile_mode','matmul_precision';deprecated_fields='forecast_data_source','observation_cache_root','observation_cache_batch_size','refresh_observation_cache';deprecated_hits=[field_name for field_name in deprecated_fields if field_name in controls]
	if deprecated_hits:joined=', '.join(sorted(deprecated_hits));raise ValueError(f"runtime_controls no longer supports legacy cache fields: {joined}. Use shared_data_dir/shared_data_signature for precomputed data, or omit them to use the live forecaster path.")
	for field_name in bool_fields:
		if field_name in controls:setattr(cfg.runtime,field_name,bool(controls[field_name]))
	for field_name in str_fields:
		if field_name in controls:setattr(cfg.runtime,field_name,str(controls[field_name]))
	if'shared_data_dir'in controls:cfg.runtime.shared_data_dir=None if controls['shared_data_dir']in(None,'')else str(Path(controls['shared_data_dir']).resolve())
	if'shared_data_signature'in controls:cfg.runtime.shared_data_signature=None if controls['shared_data_signature']in(None,'')else str(controls['shared_data_signature'])
def _apply_reward_controls(cfg,reward_controls:dict[str,Any]|None)->None:
	controls=dict(reward_controls or{});removed_reward_keys={'lambda_throughput':"Remove 'lambda_throughput'; it is no longer supported.",'w_action_pen':"w_action_pen was removed; use action_boundary_penalty_weight under the restored MADRL baseline reward contract.",'w_soc_pen':"w_soc_pen was removed; split it into action_boundary_penalty_weight and soc_boundary_regularization_weight.",'action_feasibility_regularization_weight':"action_feasibility_regularization_weight belongs to the reverted SoC-aware mapping experiment; use action_boundary_penalty_weight and re-run notebooks/madrl/train_base.ipynb.",'local_action_penalty_mode':"local_action_penalty_mode was removed in fixMADRL section 5.3; rerun notebooks/madrl/train_base.ipynb with projected actions.",'local_action_penalty_weight':"local_action_penalty_weight was removed in fixMADRL section 5.3; use action_boundary_penalty_weight under the restored baseline reward contract.",'terminal_soc_value_weight':"terminal_soc_value_weight was removed in fixMADRL Step 2; terminal SoC shaping is no longer part of NormalReward."};removed_hits=sorted(set(controls).intersection(removed_reward_keys))
	if removed_hits:details=' '.join(removed_reward_keys[key]for key in removed_hits);raise ValueError(f"Legacy reward_controls key(s) are no longer supported: {removed_hits}. {details}")
	known_reward_keys={'action_boundary_penalty_weight','soc_boundary_regularization_weight','throughput_bonus_eur_per_kwh_max','soc_boundary_epsilon','soc_boundary_margin','export_subsidy_eur_per_kwh','import_price_markup_eur_per_kwh','storage_objective_mode','storage_price_mode','storage_profit_weight','w_voltage_pen','w_line_pen','w_trafo_pen'};unknown_keys=set(controls)-known_reward_keys
	if unknown_keys:raise ValueError(f"Unknown reward_controls key(s): {sorted(unknown_keys)}. Supported keys: {sorted(known_reward_keys)}.")
	for field_name in('action_boundary_penalty_weight','soc_boundary_regularization_weight','throughput_bonus_eur_per_kwh_max','soc_boundary_epsilon','soc_boundary_margin','export_subsidy_eur_per_kwh','import_price_markup_eur_per_kwh','storage_profit_weight','w_voltage_pen','w_line_pen','w_trafo_pen'):
		if field_name in controls:setattr(cfg.reward,field_name,float(controls[field_name]))
	for field_name in('storage_objective_mode','storage_price_mode'):
		if field_name in controls:setattr(cfg.reward,field_name,str(controls[field_name]))
def _apply_safety_controls(cfg,safety_controls:dict[str,Any]|None)->None:
	controls=dict(safety_controls or{});numeric_fields='projection_iters','voltage_margin_pu','line_margin_pct','trafo_margin_pct','linearization_delta_kw'
	for field_name in numeric_fields:
		if field_name in controls:value=controls[field_name];setattr(cfg.safety,field_name,int(value)if field_name=='projection_iters'else float(value))
	if'record_diagnostics'in controls:cfg.safety.record_diagnostics=bool(controls['record_diagnostics'])
	if'projector_mode'in controls:cfg.safety.projector_mode=str(controls['projector_mode'])
	cfg.safety.enabled=bool(is_safe_poc_algorithm(cfg)and controls.get('enabled',True))
def _serialize_safety_cfg(cfg)->dict[str,Any]:return{'enabled':bool(getattr(cfg.safety,'enabled',False)),'projector_mode':str(getattr(cfg.safety,'projector_mode','joint_linearized')),'projection_iters':int(getattr(cfg.safety,'projection_iters',6)),'voltage_margin_pu':float(getattr(cfg.safety,'voltage_margin_pu',.005)),'line_margin_pct':float(getattr(cfg.safety,'line_margin_pct',5.)),'trafo_margin_pct':float(getattr(cfg.safety,'trafo_margin_pct',5.)),'linearization_delta_kw':float(getattr(cfg.safety,'linearization_delta_kw',.25)),'record_diagnostics':bool(getattr(cfg.safety,'record_diagnostics',True))}
def _default_algorithm(experiment_controls:dict[str,Any],train_controls:dict[str,Any])->str:
	algorithm=experiment_controls.get('algorithm')
	if algorithm:return str(algorithm)
	return'MATD3'if str(train_controls.get('profile','base'))=='gpu_fast'else'MADDPG'
def _json_default(value:Any):
	if isinstance(value,(Path,torch.device)):return str(value)
	if hasattr(value,'item'):
		try:return value.item()
		except Exception:pass
	raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
def _load_json(path:str|Path)->dict[str,Any]:return json.loads(Path(path).read_text(encoding='utf-8'))
def _write_json(path:str|Path,payload:dict[str,Any])->Path:target_path=Path(path);target_path.parent.mkdir(parents=True,exist_ok=True);target_path.write_text(json.dumps(payload,indent=2,default=_json_default),encoding='utf-8');return target_path
def _kill_process_tree(process:subprocess.Popen[str])->None:
	if process.poll()is not None:return
	if os.name=='nt':
		subprocess.run(['taskkill','/T','/F','/PID',str(process.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
		return
	process.terminate()
	try:process.wait(timeout=5)
	except subprocess.TimeoutExpired:process.kill()
def _stream_subprocess_output(*,process:subprocess.Popen[str],log_path:Path)->int:
	with log_path.open('w',encoding='utf-8')as log_handle:
		stream=process.stdout
		if stream is None:return process.wait()
		try:
			for line in stream:
				print(line,end='')
				sys.stdout.flush()
				log_handle.write(line)
				log_handle.flush()
			return process.wait()
		except BaseException:
			_kill_process_tree(process)
			raise
		finally:
			stream.close()
def resolve_madrl_model_root(*,algorithm:str,prediction_mode:str,experiment_name:str,model_root=None,root=None,checkpoint_root=None)->Path:
	if model_root is not None:
		candidate=Path(model_root).resolve()
		if not candidate.exists():raise FileNotFoundError(f"Requested model_root does not exist: '{candidate}'")
		if(candidate/algorithm).exists():return candidate
		if candidate.name==algorithm and any(candidate.glob('actor_agent_*_ep_*.pth')):return candidate.parent
		raise FileNotFoundError(f"model_root should point to a run directory that contains the algorithm subdirectory or directly to that algorithm subdirectory. Got '{candidate}'.")
	resolved_checkpoint_root=Path(checkpoint_root).resolve()if checkpoint_root is not None else get_checkpoint_root(root).resolve();return find_latest_training_run(resolved_checkpoint_root,algorithm=algorithm,prediction_mode=prediction_mode,experiment_name=experiment_name)
def _infer_checkpoint_action_dim(checkpoint_info:dict)->int|None:
	actor_path=Path(checkpoint_info['algo_dir'])/f"actor_agent_0_ep_{checkpoint_info['saved_episode_tag']}.pth"
	if not actor_path.exists():return
	state_dict=torch.load(actor_path,map_location='cpu')
	for tensor in(state_dict.get('head.fc.bias'),state_dict.get('head.fc.weight')):
		if isinstance(tensor,torch.Tensor)and tensor.ndim>=1:return int(tensor.shape[0])
def load_madrl_controller(cfg,model_root=None,*,algorithm:str|None=None,episode_tag:int|None=None,device=None,prediction_mode:str|None=None,experiment_name:str='grid_mainline',checkpoint_root=None,root=None):
	load_cfg=deepcopy(cfg);load_cfg.algo.name=algorithm or load_cfg.algo.name
	if device is not None:load_cfg.runtime.device=resolve_device(device)
	load_cfg.runtime.forecast_ready=None if getattr(load_cfg.runtime,'shared_data_dir',None)else ensure_mainline_forecast_ready(load_cfg);resolved_prediction_mode=prediction_mode or('perfect'if str(load_cfg.forecast.type).strip().lower()=='perfect'else'normal');resolved_model_root=resolve_madrl_model_root(algorithm=load_cfg.algo.name,prediction_mode=resolved_prediction_mode,experiment_name=experiment_name,model_root=model_root,root=root,checkpoint_root=checkpoint_root);env=build_env(load_cfg,mode='test')
	try:
		load_cfg.runtime.observation_schema=dict(env.observation_schema);load_cfg.runtime.observation_layout=dict(env.observation_layout);checkpoint_info=resolve_checkpoint_to_load(resolved_model_root,load_cfg.algo.name,episode_tag=episode_tag,expected_training_contract_signature=build_training_contract_signature(load_cfg));env_action_dim=int(env.action_space[0].shape[0]);load_cfg.runtime.action_dim=int(_infer_checkpoint_action_dim(checkpoint_info)or env_action_dim);agent_cls=get_agent_cls(load_cfg.algo.name);agents=[agent_cls(load_cfg,agent_id=i)for i in range(load_cfg.env.num_agents)]
		for agent in agents:agent.load_model(checkpoint_info['algo_dir'],checkpoint_info['saved_episode_tag'])
		controller=MADRLController(agents,noise_std=.0);return{'controller':controller,'checkpoint_info':checkpoint_info,'cfg':load_cfg,'model_root':str(resolved_model_root)}
	finally:env.close()
def load_madrl_training_result(*,algorithm:str,prediction_mode:str,experiment_name:str='grid_mainline',model_root=None,checkpoint_root=None,root=None)->dict[str,Any]:
	from predictors.shared_data import PRICE_OBSERVATION_CONTRACT
	resolved_model_root=resolve_madrl_model_root(algorithm=algorithm,prediction_mode=prediction_mode,experiment_name=experiment_name,model_root=model_root,root=root,checkpoint_root=checkpoint_root);result_json_path=(resolved_model_root/'_meta'/'train_result.json').resolve()
	if not result_json_path.exists():raise FileNotFoundError(f"Training result metadata is missing: '{result_json_path}'. Re-run the canonical MADRL notebook for '{experiment_name}'.")
	result=_load_json(result_json_path)
	if'training_contract_signature'not in result:raise KeyError(f"Old training result '{result_json_path}' is missing 'training_contract_signature'. New MADRL training contract expects checkpoint/result metadata with shared_data schema, observation features, storage price mode, and train_init_soc range without terminal SoC shaping. Re-run notebooks/madrl/train_base.ipynb.")
	stored_reward_controls=dict(dict(result.get('experiment_controls',{})).get('reward_controls',{}))
	removed_reward_keys=sorted(set(stored_reward_controls).intersection({'w_soc_pen','w_action_pen','lambda_throughput','local_action_penalty_mode','local_action_penalty_weight','terminal_soc_value_weight','action_feasibility_regularization_weight'}))
	if removed_reward_keys:raise ValueError(f"Old training result '{result_json_path}' contains removed reward_controls key(s) {removed_reward_keys}. Expected action_boundary_penalty_weight under the restored MADRL baseline contract. Re-run notebooks/madrl/train_base.ipynb.")
	required_reward_keys=('action_boundary_penalty_weight','soc_boundary_regularization_weight','throughput_bonus_eur_per_kwh_max','soc_boundary_epsilon','soc_boundary_margin','export_subsidy_eur_per_kwh','import_price_markup_eur_per_kwh','storage_objective_mode','storage_price_mode','storage_profit_weight','w_voltage_pen','w_line_pen','w_trafo_pen')
	missing_reward_keys=[field_name for field_name in required_reward_keys if field_name not in stored_reward_controls]
	if missing_reward_keys:raise ValueError(f"Old training result '{result_json_path}' is missing Step 3 reward_controls key(s) {missing_reward_keys}. Re-run notebooks/madrl/train_base.ipynb.")
	if 'terminal_soc_value_weight' in dict(result.get('training_contract',{})):raise ValueError(f"Old training result '{result_json_path}' contains terminal_soc_value_weight in training_contract. New MADRL reward contract removed terminal SoC shaping. Re-run notebooks/madrl/train_base.ipynb.")
	training_contract=dict(result.get('training_contract',{}))
	removed_contract_keys=sorted(set(training_contract).intersection({'action_feasibility_regularization_weight','action_mapping_contract'}))
	if removed_contract_keys:raise ValueError(f"Old training result '{result_json_path}' contains removed contract field(s) {removed_contract_keys}. Old field 'action_mapping_contract' is not valid; expected 'actor_action_mapping_contract'={ACTOR_ACTION_MAPPING_CONTRACT!r} and 'training_health_contract'={TRAINING_HEALTH_CONTRACT!r}. Re-run notebooks/madrl/train_base.ipynb.")
	for field_name,expected_value in{'actor_action_mapping_contract':ACTOR_ACTION_MAPPING_CONTRACT,'training_health_contract':TRAINING_HEALTH_CONTRACT,'price_observation_contract':PRICE_OBSERVATION_CONTRACT}.items():
		if field_name not in training_contract:raise ValueError(f"Old training result '{result_json_path}' is missing {field_name!r}. Expected {field_name}={expected_value!r}. Re-run notebooks/madrl/train_base.ipynb.")
		if str(training_contract[field_name])!=expected_value:raise ValueError(f"Training result '{result_json_path}' has {field_name}={training_contract[field_name]!r}; expected {expected_value!r}. Re-run notebooks/madrl/train_base.ipynb.")
	if 'discount_gamma' not in training_contract:raise ValueError(f"Old training result '{result_json_path}' is missing 'discount_gamma'. Expected explicit discount_gamma in the MADRL training contract so gamma-specific checkpoints are not mixed. Re-run notebooks/madrl/train_base.ipynb.")
	try:float(training_contract['discount_gamma'])
	except(TypeError,ValueError)as exc:raise ValueError(f"Training result '{result_json_path}' has invalid discount_gamma={training_contract['discount_gamma']!r}. Expected numeric discount_gamma in training_contract. Re-run notebooks/madrl/train_base.ipynb.")from exc
	return{'model_root':str(resolved_model_root),'result_json_path':str(result_json_path),'result':result}
def _format_power_flow_summary_message(summary:dict[str,Any])->str:
	total_steps=int(summary.get('steps',0));nonconverged_steps=int(summary.get('nonconverged_steps',0))
	if total_steps<=0:return'[power-flow] training convergence was not tracked.'
	if nonconverged_steps==0:return f'[power-flow] training convergence: all {total_steps} tracked step(s) converged.'
	fraction_pct=100.*float(nonconverged_steps)/float(max(total_steps,1));first_error=str(summary.get('first_pf_error')or'')
	error_suffix=''if not first_error else f' first_pf_error={first_error}'
	return f'[power-flow] training non-converged steps: {nonconverged_steps}/{total_steps} ({fraction_pct:.3f}%).{error_suffix}'
def _build_result_payload(*,cfg,experiment_controls:dict[str,Any],data_controls:dict[str,Any],battery_controls:dict[str,Any],train_controls:dict[str,Any],checkpoint_controls:dict[str,Any],runtime_state,forecast_ready,summary:dict[str,Any],runner,episodes_completed:int,save_dir:Path,meta_dir:Path,log_path:Path,reward_summary_path:Path,env_name:str,run_number:int,seed:int,applied_controls:dict[str,Any])->dict[str,Any]:experiment_name=slugify_checkpoint_token(checkpoint_controls.get('experiment_name',save_dir.parent.name),default='grid_mainline');run_metadata=dict(getattr(runner,'run_metadata',{}));training_contract=build_training_contract(cfg);training_contract_signature=build_training_contract_signature(cfg);return{'algorithm':str(cfg.algo.name),'prediction_mode':str(applied_controls['prediction_mode']),'evaluation_mode':str(applied_controls['evaluation_mode']),'experiment_name':experiment_name,'run_label':str(save_dir.name),'episodes_completed':int(episodes_completed),'saved_episode_tag':int(episodes_completed),'save_dir':str(save_dir),'model_root':str(save_dir),'meta_dir':str(meta_dir),'log_path':str(log_path),'reward_summary_path':str(reward_summary_path),'checkpoint_info':resolve_checkpoint_to_load(save_dir,cfg.algo.name,episode_tag=episodes_completed,expected_training_contract_signature=training_contract_signature),'training_contract_signature':training_contract_signature,'training_contract':training_contract,'training_health':runner.build_training_health_summary(),'power_flow_summary':runner.build_power_flow_summary(),'tensorboard_dir':str(get_tensorboard_run_dir(cfg.algo.name,env_name,run_number=run_number,seed=seed,root=project_root())),'device':str(cfg.runtime.device),'vec_env':type(runner.env).__name__,'perf_summary':dict(runner.perf_summary),'shared_data_dir':getattr(cfg.runtime,'shared_data_dir',None),'shared_data_signature':getattr(cfg.runtime,'shared_data_signature',None),'started_at':run_metadata.get('started_at'),'finished_at':run_metadata.get('finished_at'),'elapsed_seconds':run_metadata.get('elapsed_seconds'),'estimated_end_time':run_metadata.get('estimated_end_time'),'summary':summary,'safety_summary':runner.build_safety_summary(),'device_info':describe_device(runtime_state),'experiment_controls':dict(experiment_controls),'data_controls':dict(data_controls),'battery_controls':dict(battery_controls),'train_controls':dict(train_controls),'checkpoint_controls':dict(checkpoint_controls),'safety_controls':_serialize_safety_cfg(cfg),'forecast_ready':forecast_ready,'pid':int(os.getpid())}
def run_external_train_mainline(*,project_root,experiment_controls:dict[str,Any],data_controls:dict[str,Any],train_controls:dict[str,Any],battery_controls:dict[str,Any]|None=None,checkpoint_controls:dict[str,Any]|None=None,data_dir=None,env_name:str='GridTrainMainline',run_number:int=1)->dict[str,Any]:
	root=Path(project_root).resolve();resolved_data_dir=Path(data_dir).resolve()if data_dir is not None else(root/'data').resolve();checkpoint_controls=dict(checkpoint_controls or{});algorithm=_default_algorithm(experiment_controls,train_controls);prediction_mode=str(data_controls.get('prediction_mode','perfect')).strip().lower();checkpoint_root=Path(checkpoint_controls.get('checkpoint_root')or get_checkpoint_root(root)).resolve();experiment_name=slugify_checkpoint_token(checkpoint_controls.get('experiment_name','grid_mainline'),default='grid_mainline');run_paths=build_training_run_paths(checkpoint_root,algorithm=algorithm,prediction_mode=prediction_mode,experiment_name=experiment_name,train_episodes=train_controls.get('train_episodes'),max_train_steps=train_controls.get('max_train_steps'));model_root=Path(run_paths['model_root']).resolve();meta_dir=Path(run_paths['meta_dir']).resolve();run_label,prediction_mode,experiment_name=(str(run_paths[key])for key in('run_label','prediction_mode','experiment_name'));meta_dir.mkdir(parents=True,exist_ok=True);checkpoint_payload={'experiment_name':experiment_name,'checkpoint_root':str(checkpoint_root),'prediction_mode':prediction_mode,'run_label':run_label,'model_root':str(model_root)};battery_controls=dict(battery_controls or{});launch_info={'project_root':str(root),'experiment_controls_path':str(_write_json(meta_dir/'experiment_controls.json',experiment_controls)),'data_controls_path':str(_write_json(meta_dir/'data_controls.json',data_controls)),'battery_controls_path':str(_write_json(meta_dir/'battery_controls.json',battery_controls)),'train_controls_path':str(_write_json(meta_dir/'train_controls.json',train_controls)),'checkpoint_controls_path':str(_write_json(meta_dir/'checkpoint_controls.json',checkpoint_payload)),'result_json_path':str(meta_dir/'train_result.json'),'log_path':str(meta_dir/'train.log'),'data_dir':str(resolved_data_dir),'save_dir':str(model_root),'model_root':str(model_root),'meta_dir':str(meta_dir),'checkpoint_root':str(checkpoint_root),'prediction_mode':str(prediction_mode),'experiment_name':str(experiment_name),'run_label':str(run_label),'env_name':str(env_name),'run_number':int(run_number)};command=[sys.executable,'-m','scripts.mainline_madrl','--experiment-controls',str(launch_info['experiment_controls_path']),'--data-controls',str(launch_info['data_controls_path']),'--battery-controls',str(launch_info['battery_controls_path']),'--train-controls',str(launch_info['train_controls_path']),'--checkpoint-controls',str(launch_info['checkpoint_controls_path']),'--data-dir',str(launch_info['data_dir']),'--save-dir',str(launch_info['save_dir']),'--result-json',str(launch_info['result_json_path']),'--env-name',str(launch_info['env_name']),'--run-number',str(launch_info['run_number'])];env=os.environ.copy();env.setdefault('PYTHONUNBUFFERED','1');log_path=Path(launch_info['log_path']);log_path.parent.mkdir(parents=True,exist_ok=True)
	creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt'and hasattr(subprocess,'CREATE_NEW_PROCESS_GROUP')else 0
	process=subprocess.Popen(command,cwd=str(project_root),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,creationflags=creationflags)
	returncode=_stream_subprocess_output(process=process,log_path=log_path)
	result=_load_json(launch_info['result_json_path'])if Path(launch_info['result_json_path']).exists()else None;launch_metadata={'pid':int(process.pid),'returncode':int(returncode),'command':command,'launch_info':launch_info,'result':result}
	if returncode!=0:raise subprocess.CalledProcessError(returncode,command)
	return launch_metadata
def parse_args(argv:list[str]|None=None)->argparse.Namespace:
	parser=argparse.ArgumentParser(description='Run the Grid MADRL training mainline.')
	for flag in('--experiment-controls','--data-controls','--train-controls'):parser.add_argument(flag,required=True)
	for flag in('--battery-controls','--checkpoint-controls','--data-dir','--save-dir','--result-json'):parser.add_argument(flag)
	parser.add_argument('--env-name',default='GridTrainMainline');parser.add_argument('--run-number',type=int,default=1);return parser.parse_args(argv)
def main(argv:list[str]|None=None)->int:
	args=parse_args(argv);torch.set_num_threads(1)
	try:torch.set_num_interop_threads(1)
	except(AttributeError,RuntimeError):pass
	experiment_controls,data_controls,train_controls=_load_json(args.experiment_controls),_load_json(args.data_controls),_load_json(args.train_controls);battery_controls,checkpoint_controls=_load_json(args.battery_controls)if args.battery_controls else{},_load_json(args.checkpoint_controls)if args.checkpoint_controls else{};seed=int(experiment_controls.get('seed',0));data_dir=Path(args.data_dir).resolve()if args.data_dir is not None else project_root()/'data';save_dir=Path(args.save_dir).resolve()if args.save_dir is not None else Path(build_training_run_paths(Path(checkpoint_controls.get('checkpoint_root')or get_checkpoint_root(project_root())).resolve(),algorithm=_default_algorithm(experiment_controls,train_controls),prediction_mode=str(data_controls.get('prediction_mode','perfect')),experiment_name=checkpoint_controls.get('experiment_name','grid_mainline'),train_episodes=train_controls.get('train_episodes'),max_train_steps=train_controls.get('max_train_steps'))['model_root']).resolve();meta_dir=(save_dir/'_meta').resolve();log_path=meta_dir/'train.log';reward_summary_path=meta_dir/'train_reward_summary.json';result_json=Path(args.result_json).resolve()if args.result_json is not None else meta_dir/'train_result.json';meta_dir.mkdir(parents=True,exist_ok=True);cfg=compose_experiment_config(profile=str(train_controls.get('profile','base')),algorithm=experiment_controls.get('algorithm'),model_family=str(train_controls.get('model_family','mlp')),vec_env_type=train_controls.get('vec_env_type'),data_dir=data_dir,device=experiment_controls.get('device_request'),runtime_mode=str(experiment_controls.get('runtime_mode','performance')),seed=seed,require_cuda=experiment_controls.get('require_cuda'))
	for(apply_fn,key)in((_apply_model_controls,'model_controls'),(_apply_runtime_controls,'runtime_controls'),(_apply_reward_controls,'reward_controls'),(_apply_safety_controls,'safety_controls')):apply_fn(cfg,experiment_controls.get(key))
	agent_profiles=list(data_controls.get('agent_profiles',cfg.data.agent_profiles));n_requested_agents=len(agent_profiles);applied_controls=apply_mainline_experiment_settings(cfg,prediction_mode=str(data_controls.get('prediction_mode','perfect')),test_start_date=data_controls.get('test_start_date'),test_end_date=data_controls.get('test_end_date'),agent_profiles=agent_profiles,agent_bus_ids=data_controls.get('agent_bus_ids'),load_scale=data_controls.get('load_scale',cfg.data.load_scale or[1.]*n_requested_agents),pv_scale=data_controls.get('pv_scale',cfg.data.pv_scale or[1.]*n_requested_agents),battery_controls=battery_controls,forecast_controls=experiment_controls.get('forecast_controls'),future_horizon=int(data_controls.get('future_horizon',cfg.env.future_horizon)),train_year=data_controls.get('train_year'),test_year=data_controls.get('test_year'));_apply_train_controls(cfg,train_controls);cfg.train.noise_decay_steps=cfg.train.train_episodes*resolve_train_episode_limit(cfg);runtime_state=configure_torch_runtime(cfg,device=experiment_controls.get('device_request'),seed=seed,require_cuda=experiment_controls.get('require_cuda'));forecast_ready=None if getattr(cfg.runtime,'shared_data_dir',None)else ensure_mainline_forecast_ready(cfg);cfg.runtime.forecast_ready=forecast_ready;summary=summarize_experiment(cfg);summary.update({'applied_controls':applied_controls,'device_info':describe_device(runtime_state),'battery_controls':dict(battery_controls),'checkpoint_controls':dict(checkpoint_controls),'model_root':str(save_dir),'meta_dir':str(meta_dir),'log_path':str(log_path),'run_label':str(save_dir.name),'training_backend':'mainline','shared_data_dir':getattr(cfg.runtime,'shared_data_dir',None),'shared_data_signature':getattr(cfg.runtime,'shared_data_signature',None),'safety':_serialize_safety_cfg(cfg)});runner=build_train_runner(cfg,seed=seed,env_name=args.env_name,number=args.run_number);summary['shared_data_metadata']=dict(getattr(runner,'shared_data_metadata',{})or{});episodes_completed=0
	try:episodes_completed=runner.run();save_dir.mkdir(parents=True,exist_ok=True);runner.save_model(str(save_dir),episode=episodes_completed);_write_json(reward_summary_path,runner.build_reward_summary());result_payload=_build_result_payload(cfg=cfg,experiment_controls=experiment_controls,data_controls=data_controls,battery_controls=battery_controls,train_controls=train_controls,checkpoint_controls=checkpoint_controls,runtime_state=runtime_state,forecast_ready=forecast_ready,summary=summary,runner=runner,episodes_completed=episodes_completed,save_dir=save_dir,meta_dir=meta_dir,log_path=log_path,reward_summary_path=reward_summary_path,env_name=args.env_name,run_number=int(args.run_number),seed=seed,applied_controls=applied_controls);result_payload['perf_summary'].update({'shared_data_enabled':bool(getattr(cfg.runtime,'shared_data_dir',None)),'shared_data_dir':getattr(cfg.runtime,'shared_data_dir',None),'shared_data_signature':getattr(cfg.runtime,'shared_data_signature',None)});_write_json(result_json,result_payload);print(_format_power_flow_summary_message(dict(result_payload.get('power_flow_summary',{})or{})));return 0
	finally:runner.close()
__all__=['_apply_reward_controls','_apply_runtime_controls','_apply_train_controls','load_madrl_training_result','main','parse_args','run_external_train_mainline']
if __name__=='__main__':raise SystemExit(main())
