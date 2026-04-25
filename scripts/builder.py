from __future__ import annotations
import json,sys
from pathlib import Path
from typing import Any
from data.loaders.registry import _resolve_split_dates,build_dataset,resolve_dataset_window_spec
from envs.grid.core.grid_core import GridCore
from envs.grid.deployments import build_agent_deployments
from envs.grid_env import GridEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.normalization import build_observation_normalizer
from envs.rewards.NormalReward import NormalReward
from envs.subproc_vec_env import DummyVecEnv,SubprocVecEnv
from models.assembly import validate_and_finalize_model_config
from predictors.registry import build_forecaster
from scripts.train import TrainRunner
from predictors.shared_data import load_madrl_shared_data_manifest,select_shared_data_episode_indices,validate_madrl_shared_data_runtime_contract
from scripts.utils.torch_runtime import configure_torch_runtime
def _resolve_shared_data_dir(cfg:Any)->Path|None:shared_data_dir=getattr(getattr(cfg,'runtime',None),'shared_data_dir',None);return None if shared_data_dir in(None,'')else Path(shared_data_dir).resolve()
def _shared_split_dir(cfg:Any,split:str)->Path|None:
	shared_data_dir=_resolve_shared_data_dir(cfg)
	if str(split)not in{'train','test'}or shared_data_dir is None:return
	split_dir=(shared_data_dir/str(split)).resolve();manifest_path=split_dir/'manifest.json'
	if not manifest_path.exists():raise FileNotFoundError(f"shared_data_dir is set to '{shared_data_dir}', but split='{split}' is missing '{manifest_path.name}'.")
	return split_dir
def _load_split_manifest(split_dir:Path)->dict[str,Any]:return json.loads((split_dir/'manifest.json').read_text(encoding='utf-8'))
def _clear_runtime_split_state(cfg:Any)->None:
	runtime_cfg=getattr(cfg,'runtime',None)
	if runtime_cfg is not None:runtime_cfg.effective_split_controls=runtime_cfg.selected_episode_indices=None
def _full_manifest_episode_indices(split_manifest:dict[str,Any])->list[int]:return[int(entry['episode_idx'])for entry in list(split_manifest.get('episodes')or[])]
def _cfg_runtime_split_payload(*,source:str,split:str,year:int,start_date:str|None,end_date:str|None,exclude_start_date:str|None,exclude_end_date:str|None,window_strategy:str='cfg_window',window_days:int=1,window_stride_days:int=1,episode_length:int|None=None,base_episode_length:int|None=None)->dict[str,Any]:return{'source':str(source),'split':str(split),'year':int(year),'start_date':start_date,'end_date':end_date,'exclude_start_date':exclude_start_date,'exclude_end_date':exclude_end_date,'window_strategy':str(window_strategy),'window_days':int(window_days),'window_stride_days':int(window_stride_days),'episode_length':None if episode_length is None else int(episode_length),'base_episode_length':None if base_episode_length is None else int(base_episode_length)}
def _shared_data_metadata(cfg:Any)->dict[str,Any]|None:
	shared_data_dir=_resolve_shared_data_dir(cfg)
	if shared_data_dir is None:return
	manifest=load_madrl_shared_data_manifest(shared_data_dir);return{'shared_data_dir':str(shared_data_dir),'shared_data_signature':str(getattr(getattr(cfg,'runtime',None),'shared_data_signature',manifest.get('signature_hash',''))),'manifest':manifest,'split_dirs':{split:str((shared_data_dir/split).resolve())for split in('train','test')if(shared_data_dir/split/'manifest.json').exists()}}
def _subproc_vec_env_is_supported_in_current_process()->tuple[bool,str|None]:
	main_module=sys.modules.get('__main__');main_file=getattr(main_module,'__file__',None)
	if'ipykernel'in sys.modules:return False,'Jupyter/IPython kernels do not reliably support spawn-based vector environments.'
	if main_file is None:return False,'the current __main__ module has no importable file path.'
	if str(main_file).startswith('<'):return False,f"the current __main__ entrypoint is {main_file!r}."
	return True,None
def build_env(cfg:Any,mode:str,dataset:Any|None=None,reward_fn:Any|None=None,forecaster:Any|None=None,obs_builder:Any|None=None)->Any:
	_clear_runtime_split_state(cfg);split_name=str(mode);split_precomputed_dir=_shared_split_dir(cfg,split_name)if split_name in{'train','test'}else None;shared_split_manifest=None
	if split_precomputed_dir is not None:shared_data_dir=_resolve_shared_data_dir(cfg);shared_split_manifest=_load_split_manifest(split_precomputed_dir);validate_madrl_shared_data_runtime_contract(cfg,shared_data_dir=shared_data_dir,split_dir=split_precomputed_dir,split_manifest=shared_split_manifest);split_controls=dict(shared_split_manifest.get('split_controls')or{});dataset=build_dataset(cfg,mode=mode,override_start_date=split_controls.get('start_date'),override_end_date=split_controls.get('end_date'),override_exclude_start_date=split_controls.get('exclude_start_date'),override_exclude_end_date=split_controls.get('exclude_end_date'));requested_start=cfg.data.test_start_date if split_name=='test'else cfg.data.train_start_date;requested_end=cfg.data.test_end_date if split_name=='test'else cfg.data.train_end_date;selected_episode_indices=select_shared_data_episode_indices(shared_split_manifest,start_date=requested_start,end_date=requested_end)if split_name=='test'else _full_manifest_episode_indices(shared_split_manifest);cfg.runtime.effective_split_controls=_cfg_runtime_split_payload(source='shared_data_manifest',split=split_name,year=int(split_controls.get('year',cfg.data.train_year if split_name=='train'else cfg.data.test_year)),start_date=split_controls.get('start_date'),end_date=split_controls.get('end_date'),exclude_start_date=split_controls.get('exclude_start_date'),exclude_end_date=split_controls.get('exclude_end_date'),window_strategy=str(split_controls.get('window_strategy','cfg_window')),window_days=int(split_controls.get('window_days',1)),window_stride_days=int(split_controls.get('window_stride_days',1)),episode_length=int(shared_split_manifest.get('episode_length',getattr(dataset,'episode_length',0))),base_episode_length=int(split_controls.get('base_episode_length',cfg.env.episode_limit)));cfg.runtime.selected_episode_indices=[int(index)for index in selected_episode_indices]
	elif dataset is None:dataset=build_dataset(cfg,mode=mode)
	else:shared_split_manifest=None
	if shared_split_manifest is None:selected_year,start_date,end_date,exclude_start_date,exclude_end_date=_resolve_split_dates(cfg,mode);window_spec=resolve_dataset_window_spec(cfg,split_name);cfg.runtime.effective_split_controls=_cfg_runtime_split_payload(source='cfg',split=split_name,year=int(selected_year),start_date=start_date,end_date=end_date,exclude_start_date=exclude_start_date,exclude_end_date=exclude_end_date,window_strategy=str(window_spec['window_strategy']),window_days=int(window_spec['window_days']),window_stride_days=int(window_spec['window_stride_days']),episode_length=int(window_spec['episode_length']),base_episode_length=int(window_spec['base_episode_length']));cfg.runtime.selected_episode_indices=None
	if reward_fn is None:reward_fn=NormalReward(cfg)
	if forecaster is None and split_precomputed_dir is None:forecaster=build_forecaster(cfg)
	if obs_builder is None:normalizer=build_observation_normalizer(cfg,dataset=dataset if split_name=='train'else None);obs_builder=DefaultObservationBuilder(local_features=cfg.obs.local_features,sequence_features=cfg.obs.sequence_features,future_horizon=cfg.env.future_horizon,adjacency_type=cfg.obs.adjacency_type,normalizer=normalizer,precomputed=split_precomputed_dir is not None,price_spread_scale_eur_per_kwh=float(getattr(cfg.obs,'wholesale_price_spread_scale_eur_per_kwh',.20)))
	grid_core=GridCore(build_agent_deployments(cfg),cfg.grid);env=GridEnv(cfg,mode=mode,dataset=dataset,reward_fn=reward_fn,forecaster=forecaster,obs_builder=obs_builder,grid_core=grid_core,precomputed_data_dir=split_precomputed_dir);return env
def _build_dummy_train_vec_env(cfg:Any,*,seed:int|None=None)->Any:
	def make_train_env():return build_env(cfg,mode='train')
	return DummyVecEnv(cfg.train.num_envs,make_train_env,seed=seed,parallel_episode_sampling=str(getattr(cfg.train,'parallel_episode_sampling','unique_active')))
def _build_train_vec_env(cfg:Any,*,seed:int)->Any:
	if cfg.train.vec_env_type=='dummy':return _build_dummy_train_vec_env(cfg,seed=seed)
	if cfg.train.vec_env_type=='subproc':
		supported,reason=_subproc_vec_env_is_supported_in_current_process()
		if not supported:raise RuntimeError(f"`train.vec_env_type='subproc'` is unsupported in this session: {reason}")
		return SubprocVecEnv(cfg.train.num_envs,cfg,mode='train',seed=seed)
	raise ValueError(f"Unknown train.vec_env_type '{cfg.train.vec_env_type}', expected 'dummy' or 'subproc'.")
def _finalize_runtime_from_env(cfg:Any,env:Any)->None:cfg.runtime.observation_schema=dict(env.observation_schema);cfg.runtime.observation_layout=dict(env.observation_layout);cfg.runtime.action_dim=int(env.action_space[0].shape[0])
def build_train_runner(cfg:Any,seed:int=0,env_name:str='GridEnv',number:int=1)->TrainRunner:cfg.runtime.seed=int(seed);configure_torch_runtime(cfg,seed=seed);validate_and_finalize_model_config(cfg);train_env=_build_train_vec_env(cfg,seed=seed);eval_dataset=build_dataset(cfg,mode='test');eval_env=build_env(cfg,mode='test',dataset=eval_dataset);_finalize_runtime_from_env(cfg,eval_env);validate_and_finalize_model_config(cfg);runner=TrainRunner(cfg,train_env=train_env,eval_env=eval_env,env_name=env_name,number=number,seed=seed);runner.shared_data_metadata=_shared_data_metadata(cfg);return runner
