from __future__ import annotations
import hashlib,json,re
from datetime import datetime
from pathlib import Path
LATEST_CHECKPOINT_MANIFEST='latest_checkpoint.json'
TAGGED_CHECKPOINT_MANIFEST='checkpoint_ep_{episode_tag}.json'
TRAINING_CONTRACT_SIGNATURE_KEY='training_contract_signature'
TRAINING_CONTRACT_KEY='training_contract'
ACTOR_ACTION_MAPPING_CONTRACT='soc_affine_feasible_actor_mapping_v1'
TRAINING_HEALTH_CONTRACT='finite_loss_grad_health_v1'
ACTOR_TAG_PATTERN=re.compile('actor_agent_\\d+_ep_(\\d+)\\.pth$')
CRITIC_TAG_PATTERN=re.compile('critic_agent_\\d+_ep_(\\d+)\\.pth$')
RUN_TOKEN_PATTERN=re.compile('[^a-z0-9]+')
def slugify_checkpoint_token(value:str|None,*,default:str)->str:text=str(value or'').strip().lower();text=RUN_TOKEN_PATTERN.sub('_',text).strip('_');return text or default
def build_checkpoint_budget_token(*,train_episodes:int|None,max_train_steps:int|None)->str:return f"steps{int(max_train_steps)}"if max_train_steps is not None else f"ep{int(train_episodes or 0)}"
def _format_run_timestamp(timestamp:datetime|str|None=None)->str:
	if timestamp is None:return datetime.now().strftime('%Y%m%d_%H%M%S')
	return timestamp.strftime('%Y%m%d_%H%M%S')if isinstance(timestamp,datetime)else str(timestamp).strip()
def build_training_run_label(*,algorithm:str,prediction_mode:str,experiment_name:str,train_episodes:int|None,max_train_steps:int|None,timestamp:datetime|str|None=None)->str:algorithm_token=slugify_checkpoint_token(algorithm,default='model');prediction_token=slugify_checkpoint_token(prediction_mode,default='perfect');experiment_token=slugify_checkpoint_token(experiment_name,default='grid_mainline');budget_token=build_checkpoint_budget_token(train_episodes=train_episodes,max_train_steps=max_train_steps);time_token=slugify_checkpoint_token(_format_run_timestamp(timestamp),default='run');return'_'.join([algorithm_token,prediction_token,experiment_token,budget_token,time_token])
def build_training_run_paths(checkpoint_root,*,algorithm:str,prediction_mode:str,experiment_name:str,train_episodes:int|None,max_train_steps:int|None,timestamp:datetime|str|None=None)->dict[str,object]:checkpoint_root=Path(checkpoint_root).resolve();algorithm_token=str(algorithm).strip();prediction_token=slugify_checkpoint_token(prediction_mode,default='perfect');experiment_token=slugify_checkpoint_token(experiment_name,default='grid_mainline');run_label=build_training_run_label(algorithm=algorithm_token,prediction_mode=prediction_token,experiment_name=experiment_token,train_episodes=train_episodes,max_train_steps=max_train_steps,timestamp=timestamp);model_root=checkpoint_root/algorithm_token/prediction_token/experiment_token/run_label;meta_dir=model_root/'_meta';return{'checkpoint_root':checkpoint_root,'algorithm':algorithm_token,'prediction_mode':prediction_token,'experiment_name':experiment_token,'run_label':run_label,'model_root':model_root,'meta_dir':meta_dir,'result_json_path':meta_dir/'train_result.json','progress_json_path':meta_dir/'progress.json','log_path':meta_dir/'train.log'}
def _run_has_result_metadata(run_dir:Path)->bool:return(run_dir/'_meta'/'train_result.json').exists()
def find_latest_training_run(checkpoint_root,*,algorithm:str,prediction_mode:str,experiment_name:str)->Path:
	checkpoint_root=Path(checkpoint_root).resolve();base_dir=checkpoint_root/str(algorithm).strip()/slugify_checkpoint_token(prediction_mode,default='perfect')/slugify_checkpoint_token(experiment_name,default='grid_mainline')
	if not base_dir.exists():raise FileNotFoundError(f"No training runs found under '{base_dir}'.")
	candidates=sorted((path for path in base_dir.iterdir()if path.is_dir()and path.name!='_meta'),key=lambda path:('_'.join(path.name.rsplit('_',2)[-2:])if len(path.name.rsplit('_',2))>=3 else path.name,path.name))
	if not candidates:raise FileNotFoundError(f"No run directories found under '{base_dir}'.")
	latest_run=candidates[-1]
	result_json=latest_run/'_meta'/'train_result.json'
	if not result_json.exists():raise FileNotFoundError(f"Latest training run is incomplete: '{latest_run}'. Expected exact result metadata at '{result_json}'. Re-run the canonical MADRL notebook for experiment '{experiment_name}' to produce a current checkpoint bundle.")
	try:resolve_checkpoint_to_load(latest_run,str(algorithm).strip())
	except KeyError as exc:raise KeyError(f"Latest training run '{latest_run}' uses an old checkpoint contract. Expected a manifest with '{TRAINING_CONTRACT_SIGNATURE_KEY}' for entrypoint scripts.checkpoints.find_latest_training_run. Re-run the canonical MADRL notebook for experiment '{experiment_name}'. Original error: {exc}")from exc
	except ValueError as exc:raise ValueError(f"Latest training run '{latest_run}' does not match the current checkpoint contract for entrypoint scripts.checkpoints.find_latest_training_run. Re-run the canonical MADRL notebook for experiment '{experiment_name}'. Original error: {exc}")from exc
	except FileNotFoundError as exc:raise FileNotFoundError(f"Latest training run '{latest_run}' is not a complete checkpoint bundle for entrypoint scripts.checkpoints.find_latest_training_run. Re-run the canonical MADRL notebook for experiment '{experiment_name}'. Original error: {exc}")from exc
	return latest_run
def get_algorithm_checkpoint_dir(model_dir,algorithm:str)->Path:return Path(model_dir)/algorithm
def checkpoint_tag_exists(algo_dir,episode_tag:int)->bool:algo_dir=Path(algo_dir);actor_files=list(algo_dir.glob(f"actor_agent_*_ep_{episode_tag}.pth"));critic_files=list(algo_dir.glob(f"critic_agent_*_ep_{episode_tag}.pth"));return bool(actor_files)and bool(critic_files)
def infer_latest_checkpoint_tag(model_dir,algorithm:str)->int:
	algo_dir=get_algorithm_checkpoint_dir(model_dir,algorithm)
	if not algo_dir.exists():raise FileNotFoundError(f"Checkpoint directory does not exist: '{algo_dir}'")
	actor_tags={int(match.group(1))for path in algo_dir.glob('actor_agent_*_ep_*.pth')if(match:=ACTOR_TAG_PATTERN.match(path.name))is not None};critic_tags={int(match.group(1))for path in algo_dir.glob('critic_agent_*_ep_*.pth')if(match:=CRITIC_TAG_PATTERN.match(path.name))is not None};common_tags=sorted(actor_tags&critic_tags)
	if not common_tags:raise FileNotFoundError(f"No complete checkpoint tags were found under '{algo_dir}'.")
	return common_tags[-1]
def _normalize_for_contract(value):
	if isinstance(value,dict):return{str(key):_normalize_for_contract(item)for key,item in sorted(value.items(),key=lambda pair:str(pair[0]))}
	if isinstance(value,(list,tuple)):return[_normalize_for_contract(item)for item in value]
	if isinstance(value,Path):return str(value)
	if hasattr(value,'item'):
		try:return value.item()
		except Exception:pass
	return value
def build_training_contract(cfg)->dict:
	from data.loaders.registry import resolve_test_episode_limit,resolve_train_episode_limit
	from predictors.shared_data import PRICE_OBSERVATION_CONTRACT,SHARED_DATA_SCHEMA_VERSION
	normalization_state=getattr(getattr(cfg,'runtime',None),'observation_normalization_state',None)
	normalization_signature=dict(normalization_state.get('signature')or{})if isinstance(normalization_state,dict)else None
	train_episode_limit=int(resolve_train_episode_limit(cfg));test_episode_limit=int(resolve_test_episode_limit(cfg))
	return _normalize_for_contract({'reward_contract':'madrl_incremental_storage_reward_v1','rollout_soc_contract':'continuous_soc_v1','actor_action_mapping_contract':ACTOR_ACTION_MAPPING_CONTRACT,'training_health_contract':TRAINING_HEALTH_CONTRACT,'price_observation_contract':PRICE_OBSERVATION_CONTRACT,'discount_gamma':float(cfg.algo.gamma),'train_window_days':int(getattr(cfg.env,'train_window_days',1)),'window_stride_days':int(getattr(cfg.env,'window_stride_days',1)),'train_episode_limit':train_episode_limit,'test_episode_limit':test_episode_limit,'learning_starts_transitions':int(cfg.train.resolved_learning_starts_transitions(train_episode_limit)),'actor_learning_starts_transitions':int(cfg.train.resolved_actor_learning_starts_transitions(train_episode_limit)),'n_step_return':int(cfg.train.n_step_return),'feasible_random_exploration_start':float(cfg.train.feasible_random_exploration_start),'feasible_random_exploration_end':float(cfg.train.feasible_random_exploration_end),'feasible_random_exploration_decay_steps':int(cfg.train.feasible_random_exploration_decay_steps),'shared_data_schema_version':int(SHARED_DATA_SCHEMA_VERSION),'shared_data_signature':getattr(getattr(cfg,'runtime',None),'shared_data_signature',None),'observation_feature_set':{'local':list(cfg.obs.local_features),'sequence':list(cfg.obs.sequence_features)},'observation_normalization_signature':normalization_signature,'wholesale_price_spread_scale_eur_per_kwh':float(getattr(cfg.obs,'wholesale_price_spread_scale_eur_per_kwh',.20)),'storage_objective_mode':str(cfg.reward.storage_objective_mode),'storage_price_mode':str(cfg.reward.storage_price_mode),'storage_profit_weight':float(cfg.reward.storage_profit_weight),'action_boundary_penalty_weight':float(cfg.reward.action_boundary_penalty_weight),'soc_boundary_regularization_weight':float(cfg.reward.soc_boundary_regularization_weight),'throughput_bonus_eur_per_kwh_max':float(cfg.reward.throughput_bonus_eur_per_kwh_max),'soc_boundary_epsilon':float(cfg.reward.soc_boundary_epsilon),'soc_boundary_margin':float(cfg.reward.soc_boundary_margin),'export_subsidy_eur_per_kwh':float(cfg.reward.export_subsidy_eur_per_kwh),'import_price_markup_eur_per_kwh':float(cfg.reward.import_price_markup_eur_per_kwh),'w_voltage_pen':float(cfg.reward.w_voltage_pen),'w_line_pen':float(cfg.reward.w_line_pen),'w_trafo_pen':float(cfg.reward.w_trafo_pen),'train_init_soc_low':float(getattr(cfg.env,'train_init_soc_low',cfg.env.init_soc)),'train_init_soc_high':float(getattr(cfg.env,'train_init_soc_high',cfg.env.init_soc))})
def build_training_contract_signature(cfg)->str:
	payload=build_training_contract(cfg);return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=True).encode('utf-8')).hexdigest()[:16]
def validate_checkpoint_training_contract(manifest:dict,*,expected_signature:str|None=None,entrypoint:str='scripts.checkpoints.resolve_checkpoint_to_load')->dict:
	from predictors.shared_data import PRICE_OBSERVATION_CONTRACT
	if TRAINING_CONTRACT_SIGNATURE_KEY not in manifest:raise KeyError(f"{entrypoint} received an old checkpoint manifest without '{TRAINING_CONTRACT_SIGNATURE_KEY}'. New contract expects a training contract signature covering shared_data schema, observation features, storage price mode, and train_init_soc range without terminal SoC shaping. Re-run notebooks/madrl/train_base.ipynb.")
	training_contract=dict(manifest.get(TRAINING_CONTRACT_KEY)or{})
	removed_reward_keys=sorted(set(training_contract).intersection({'terminal_soc_value_weight','w_soc_pen','w_action_pen','lambda_throughput','local_action_penalty_mode','local_action_penalty_weight','action_feasibility_regularization_weight','action_mapping_contract'}))
	if removed_reward_keys:raise ValueError(f"{entrypoint} received a checkpoint training contract containing removed field(s) {removed_reward_keys}. Old field 'action_mapping_contract' is not valid; expected 'actor_action_mapping_contract'={ACTOR_ACTION_MAPPING_CONTRACT!r} and 'training_health_contract'={TRAINING_HEALTH_CONTRACT!r}. Re-run notebooks/madrl/train_base.ipynb.")
	required_fields=('reward_contract','rollout_soc_contract','actor_action_mapping_contract','training_health_contract','price_observation_contract','discount_gamma','train_window_days','window_stride_days','train_episode_limit','test_episode_limit','learning_starts_transitions','actor_learning_starts_transitions','n_step_return','feasible_random_exploration_start','feasible_random_exploration_end','feasible_random_exploration_decay_steps','shared_data_schema_version','shared_data_signature','observation_feature_set','observation_normalization_signature','wholesale_price_spread_scale_eur_per_kwh','storage_objective_mode','storage_price_mode','storage_profit_weight','action_boundary_penalty_weight','soc_boundary_regularization_weight','throughput_bonus_eur_per_kwh_max','soc_boundary_epsilon','soc_boundary_margin','export_subsidy_eur_per_kwh','import_price_markup_eur_per_kwh','w_voltage_pen','w_line_pen','w_trafo_pen','train_init_soc_low','train_init_soc_high')
	missing_fields=[field_name for field_name in required_fields if field_name not in training_contract]
	if missing_fields:raise ValueError(f"{entrypoint} received an incomplete checkpoint training contract missing {missing_fields}. New contract expects 'actor_action_mapping_contract'={ACTOR_ACTION_MAPPING_CONTRACT!r}, 'training_health_contract'={TRAINING_HEALTH_CONTRACT!r}, 'price_observation_contract'={PRICE_OBSERVATION_CONTRACT!r}, shared-data, observation-normalization, Step 3 reward, continuous-SoC fields, and from-scratch stability fields. Re-run notebooks/madrl/train_base.ipynb.")
	if str(training_contract['actor_action_mapping_contract'])!=ACTOR_ACTION_MAPPING_CONTRACT:raise ValueError(f"{entrypoint} received actor_action_mapping_contract={training_contract['actor_action_mapping_contract']!r}; expected {ACTOR_ACTION_MAPPING_CONTRACT!r}. Re-run notebooks/madrl/train_base.ipynb.")
	if str(training_contract['training_health_contract'])!=TRAINING_HEALTH_CONTRACT:raise ValueError(f"{entrypoint} received training_health_contract={training_contract['training_health_contract']!r}; expected {TRAINING_HEALTH_CONTRACT!r}. Re-run notebooks/madrl/train_base.ipynb.")
	if str(training_contract['price_observation_contract'])!=PRICE_OBSERVATION_CONTRACT:raise ValueError(f"{entrypoint} received price_observation_contract={training_contract['price_observation_contract']!r}; expected {PRICE_OBSERVATION_CONTRACT!r}. Re-run notebooks/madrl/train_base.ipynb.")
	try:float(training_contract['discount_gamma'])
	except(TypeError,ValueError)as exc:raise ValueError(f"{entrypoint} received invalid discount_gamma={training_contract['discount_gamma']!r}. New contract expects numeric discount_gamma recorded from cfg.algo.gamma. Re-run notebooks/madrl/train_base.ipynb.")from exc
	actual_signature=str(manifest[TRAINING_CONTRACT_SIGNATURE_KEY])
	if expected_signature is not None and actual_signature!=str(expected_signature):raise ValueError(f"{entrypoint} received checkpoint training_contract_signature={actual_signature!r}, but current config expects {str(expected_signature)!r}. Re-run notebooks/madrl/train_base.ipynb.")
	return training_contract
def build_checkpoint_manifest(*,algorithm:str,saved_episode_tag:int,episodes_completed:int,total_steps:int,num_envs:int,episode_limit:int,save_dir,training_contract:dict)->dict:
	contract=_normalize_for_contract(dict(training_contract));signature=hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=True).encode('utf-8')).hexdigest()[:16];return{'algorithm':algorithm,'saved_episode_tag':int(saved_episode_tag),'episodes_completed':int(episodes_completed),'total_steps':int(total_steps),'global_interaction_step':int(total_steps),'target_total_steps':None,'num_envs':int(num_envs),'episode_limit':int(episode_limit),'save_dir':str(Path(save_dir).resolve()),TRAINING_CONTRACT_SIGNATURE_KEY:signature,TRAINING_CONTRACT_KEY:contract}
def write_checkpoint_manifest(algo_dir,manifest:dict)->dict:algo_dir=Path(algo_dir);algo_dir.mkdir(parents=True,exist_ok=True);latest_path=algo_dir/LATEST_CHECKPOINT_MANIFEST;tagged_path=algo_dir/TAGGED_CHECKPOINT_MANIFEST.format(episode_tag=manifest['saved_episode_tag']);payload=dict(manifest);latest_path.write_text(json.dumps(payload,indent=2),encoding='utf-8');tagged_path.write_text(json.dumps(payload,indent=2),encoding='utf-8');return payload
def load_latest_checkpoint_manifest(model_dir,algorithm:str)->dict:
	algo_dir=get_algorithm_checkpoint_dir(model_dir,algorithm);manifest_path=algo_dir/LATEST_CHECKPOINT_MANIFEST
	if not manifest_path.exists():raise FileNotFoundError(f"Latest checkpoint manifest not found: '{manifest_path}'")
	return json.loads(manifest_path.read_text(encoding='utf-8'))
def resolve_checkpoint_to_load(model_dir,algorithm:str,episode_tag:int|None=None,*,expected_training_contract_signature:str|None=None)->dict:
	algo_dir=get_algorithm_checkpoint_dir(model_dir,algorithm)
	if episode_tag is None:manifest=load_latest_checkpoint_manifest(model_dir,algorithm);episode_tag=int(manifest['saved_episode_tag'])
	else:
		manifest_path=algo_dir/TAGGED_CHECKPOINT_MANIFEST.format(episode_tag=int(episode_tag))
		if not manifest_path.exists():raise FileNotFoundError(f"Tagged checkpoint manifest not found: '{manifest_path}'. Re-run notebooks/madrl/train_base.ipynb to produce a current checkpoint contract.")
		manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
	training_contract=validate_checkpoint_training_contract(manifest,expected_signature=expected_training_contract_signature)
	if not checkpoint_tag_exists(algo_dir,int(episode_tag)):raise FileNotFoundError(f"Checkpoint tag {episode_tag} was not found under '{algo_dir}'.")
	resolved={'algorithm':algorithm,'algo_dir':str(algo_dir),'saved_episode_tag':int(episode_tag)}
	resolved.update({'episodes_completed':int(manifest.get('episodes_completed',episode_tag)),'total_steps':int(manifest.get('total_steps',0)),'global_interaction_step':int(manifest.get('global_interaction_step',manifest.get('total_steps',0))),'target_total_steps':manifest.get('target_total_steps'),'num_envs':int(manifest.get('num_envs',0)),'episode_limit':int(manifest.get('episode_limit',0)),'save_dir':manifest.get('save_dir',str(algo_dir)),TRAINING_CONTRACT_SIGNATURE_KEY:str(manifest[TRAINING_CONTRACT_SIGNATURE_KEY]),TRAINING_CONTRACT_KEY:training_contract})
	return resolved
