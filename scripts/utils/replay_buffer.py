from __future__ import annotations
import numpy as np,torch
from scripts.utils.torch_runtime import NestedArray,to_torch_nested
def _allocate_nested_storage(schema:dict[str,tuple[int,...]],size:int)->dict[str,np.ndarray]:return{key:np.zeros((size,*tuple(shape)),dtype=np.float32)for(key,shape)in schema.items()}
def _store_nested_range(storage:dict[str,np.ndarray],payload:NestedArray,source:slice,target:slice)->None:
	assert isinstance(payload,dict)
	for(key,value)in payload.items():storage[key][target]=np.asarray(value[source],dtype=np.float32)
def _sample_nested(storage:dict[str,np.ndarray],indices:np.ndarray,out:dict[str,np.ndarray]|None=None)->dict[str,np.ndarray]:
	if out is None:return{key:np.asarray(value[indices],dtype=np.float32).copy()for(key,value)in storage.items()}
	for(key,value)in storage.items():np.take(value,indices,axis=0,out=out[key])
	return out
def _tensor_nested_to_device(payload:dict[str,torch.Tensor],device:torch.device,*,non_blocking:bool)->dict[str,torch.Tensor]:copy_on_cpu=device.type=='cpu';return{key:tensor.to(device=device,non_blocking=non_blocking,copy=copy_on_cpu)for(key,tensor)in payload.items()}
class ReplayBuffer:
	def __init__(self,cfg:object)->None:
		if not getattr(cfg.runtime,'observation_schema',None):raise ValueError('ReplayBuffer requires cfg.runtime.observation_schema to be populated.')
		self.buffer_size=int(cfg.train.buffer_size);self.batch_size=int(cfg.train.batch_size);self.num_agents=int(cfg.env.num_agents);self.action_dim=int(cfg.runtime.action_dim);self.n_step_return=max(1,int(getattr(cfg.train,'n_step_return',1)));self.gamma=float(getattr(cfg.algo,'gamma',1.));self.observation_schema={key:tuple(shape)for(key,shape)in dict(cfg.runtime.observation_schema).items()};self.obs_storage=_allocate_nested_storage(self.observation_schema,self.buffer_size);self.next_obs_storage=_allocate_nested_storage(self.observation_schema,self.buffer_size);shape=self.buffer_size,self.num_agents;self.action_storage=np.zeros((*shape,self.action_dim),dtype=np.float32);self.reward_storage=np.zeros((*shape,1),dtype=np.float32);self.terminated_storage=np.zeros((*shape,1),dtype=np.float32);self.truncated_storage=np.zeros((*shape,1),dtype=np.float32);self.env_id_storage=np.full((self.buffer_size,),-1,dtype=np.int32);self.interaction_id_storage=np.full((self.buffer_size,),-1,dtype=np.int64);self.position=0;self.current_size=0;self.total_inserted=0;self._last_batch_size:int|None=None;(self._torch_staging_cache):dict[tuple[object,...],dict[str,object]]={}
	def _sample_indices(self)->np.ndarray:
		if self.current_size<=0:raise ValueError('ReplayBuffer cannot sample before any transition is stored.')
		if self.n_step_return<=1:return np.random.choice(self.current_size,size=self.batch_size,replace=self.current_size<self.batch_size)
		num_envs=int(self._last_batch_size or 0);recent_guard=(self.n_step_return-1)*num_envs
		if num_envs>0 and self.current_size<self.buffer_size and self.current_size>recent_guard:
			eligible_size=self.current_size-recent_guard
			return np.random.choice(eligible_size,size=self.batch_size,replace=eligible_size<self.batch_size).astype(np.int64,copy=False)
		selected:list[int]=[];attempts=0;max_attempts=max(self.batch_size*200,1000)
		while len(selected)<self.batch_size and attempts<max_attempts:
			draw_count=max(self.batch_size-len(selected),self.batch_size)
			candidates=np.random.choice(self.current_size,size=draw_count,replace=self.current_size<draw_count).astype(np.int64,copy=False);valid=self._n_step_candidates_valid(candidates)
			selected.extend(int(value)for value in candidates[valid])
			attempts+=draw_count
		if len(selected)<self.batch_size:raise ValueError(f"ReplayBuffer cannot sample {self.n_step_return}-step transitions from {self.current_size} stored rows. Increase learning_starts_transitions or reduce n_step_return.")
		return np.asarray(selected[:self.batch_size],dtype=np.int64)
	def _n_step_candidates_valid(self,indices:np.ndarray)->np.ndarray:
		num_envs=int(self._last_batch_size or 0)
		if num_envs<=0:return np.zeros(indices.shape,dtype=bool)
		base_env=self.env_id_storage[indices];base_interaction=self.interaction_id_storage[indices];valid=np.ones(indices.shape,dtype=bool);active=np.ones(indices.shape,dtype=bool)
		for step in range(self.n_step_return):
			step_indices=(indices+step*num_envs)%self.buffer_size;step_valid=(self.env_id_storage[step_indices]==base_env)&(self.interaction_id_storage[step_indices]==base_interaction+step);valid&=(~active)|step_valid
			if not np.any(active&step_valid):break
			done=np.any((self.terminated_storage[step_indices]>.5)|(self.truncated_storage[step_indices]>.5),axis=(1,2));active&=~(step_valid&done)
		return valid
	def store_transitions_batched(self,obs:NestedArray,action:np.ndarray,reward:np.ndarray,next_obs:NestedArray,terminated:np.ndarray,truncated:np.ndarray)->None:
		action,reward,terminated,truncated=(np.asarray(value,dtype=np.float32)for value in(action,reward,terminated,truncated));num_envs=int(action.shape[0]);first_block=min(num_envs,self.buffer_size-self.position);blocks=[(slice(0,first_block),slice(self.position,self.position+first_block))]
		if self.n_step_return>1 and self._last_batch_size is not None and int(self._last_batch_size)!=num_envs:raise ValueError(f"ReplayBuffer n-step storage requires a stable parallel env count, got previous {self._last_batch_size} and current {num_envs}.")
		if self.n_step_return>1 and self.buffer_size%num_envs!=0:raise ValueError(f"ReplayBuffer n-step storage requires buffer_size divisible by num_envs, got {self.buffer_size} and {num_envs}.")
		self._last_batch_size=num_envs;interaction_id=int(self.total_inserted//max(num_envs,1));env_ids=np.arange(num_envs,dtype=np.int32)
		if num_envs>first_block:blocks.append((slice(first_block,num_envs),slice(0,num_envs-first_block)))
		for(source,target)in blocks:
			if source.stop==source.start:continue
			_store_nested_range(self.obs_storage,obs,source,target);_store_nested_range(self.next_obs_storage,next_obs,source,target);self.action_storage[target],self.reward_storage[target],self.terminated_storage[target],self.truncated_storage[target]=action[source],reward[source],terminated[source],truncated[source]
			self.env_id_storage[target]=env_ids[source];self.interaction_id_storage[target]=interaction_id
		self.position=(self.position+num_envs)%self.buffer_size;self.current_size=min(self.current_size+num_envs,self.buffer_size);self.total_inserted+=num_envs
	def _n_step_scalar_payload(self,indices:np.ndarray)->dict[str,np.ndarray]:
		if self.n_step_return<=1:
			terminated=np.asarray(self.terminated_storage[indices],dtype=np.float32).copy();return{'reward':np.asarray(self.reward_storage[indices],dtype=np.float32).copy(),'next_indices':indices,'terminated':terminated,'truncated':np.asarray(self.truncated_storage[indices],dtype=np.float32).copy(),'bootstrap_discount':(self.gamma*(1.-terminated)).astype(np.float32)}
		num_envs=int(self._last_batch_size or 0);batch_size=int(indices.shape[0]);reward=np.zeros((batch_size,self.num_agents,1),dtype=np.float32);terminated=np.zeros_like(reward);truncated=np.zeros_like(reward);running=np.ones((batch_size,self.num_agents,1),dtype=bool);discount=np.ones((batch_size,self.num_agents,1),dtype=np.float32);bootstrap_discount=np.zeros_like(reward);next_indices=indices.copy()
		for step in range(self.n_step_return):
			step_indices=(indices+step*num_envs)%self.buffer_size;env_running=np.any(running,axis=(1,2));next_indices[env_running]=step_indices[env_running];reward+=running*discount*np.asarray(self.reward_storage[step_indices],dtype=np.float32);step_terminated=np.asarray(self.terminated_storage[step_indices],dtype=np.float32);step_truncated=np.asarray(self.truncated_storage[step_indices],dtype=np.float32);stop=running&((step_terminated>.5)|(step_truncated>.5));terminated=np.maximum(terminated,stop.astype(np.float32)*step_terminated);truncated=np.maximum(truncated,stop.astype(np.float32)*step_truncated);next_discount=discount*np.float32(self.gamma);bootstrap_discount=np.where(stop,next_discount*(1.-step_terminated),bootstrap_discount);running&=~stop;discount=next_discount
			if not np.any(running):break
		bootstrap_discount=np.where(running,discount,bootstrap_discount).astype(np.float32)
		return{'reward':reward.astype(np.float32),'next_indices':next_indices.astype(np.int64),'terminated':terminated.astype(np.float32),'truncated':truncated.astype(np.float32),'bootstrap_discount':bootstrap_discount}
	def sample(self)->dict[str,NestedArray]:
		indices=self._sample_indices();n_step=self._n_step_scalar_payload(indices);return{'obs':_sample_nested(self.obs_storage,indices),'action':np.asarray(self.action_storage[indices],dtype=np.float32).copy(),'reward':n_step['reward'].copy(),'next_obs':_sample_nested(self.next_obs_storage,n_step['next_indices']),'terminated':n_step['terminated'].copy(),'truncated':n_step['truncated'].copy(),'bootstrap_discount':n_step['bootstrap_discount'].copy()}
	def sample_torch(self,device:torch.device|str,*,pin_memory:bool=False,non_blocking:bool=False)->dict[str,NestedArray]:
		resolved_device=torch.device(device);indices=self._sample_indices();staging=self._get_torch_staging_cache(resolved_device,pin_memory=pin_memory);_sample_nested(self.obs_storage,indices,staging['obs_numpy']);_sample_nested(self.next_obs_storage,indices,staging['next_obs_numpy'])
		n_step=self._n_step_scalar_payload(indices);_sample_nested(self.next_obs_storage,n_step['next_indices'],staging['next_obs_numpy']);np.take(self.action_storage,indices,axis=0,out=staging['action_numpy']);np.copyto(staging['reward_numpy'],n_step['reward']);np.copyto(staging['terminated_numpy'],n_step['terminated']);np.copyto(staging['truncated_numpy'],n_step['truncated']);np.copyto(staging['bootstrap_discount_numpy'],n_step['bootstrap_discount'])
		copy_on_cpu=resolved_device.type=='cpu';return{'obs':_tensor_nested_to_device(staging['obs_tensors'],resolved_device,non_blocking=non_blocking),'action':staging['action_tensor'].to(device=resolved_device,non_blocking=non_blocking,copy=copy_on_cpu),'reward':staging['reward_tensor'].to(device=resolved_device,non_blocking=non_blocking,copy=copy_on_cpu),'next_obs':_tensor_nested_to_device(staging['next_obs_tensors'],resolved_device,non_blocking=non_blocking),'terminated':staging['terminated_tensor'].to(device=resolved_device,non_blocking=non_blocking,copy=copy_on_cpu),'truncated':staging['truncated_tensor'].to(device=resolved_device,non_blocking=non_blocking,copy=copy_on_cpu),'bootstrap_discount':staging['bootstrap_discount_tensor'].to(device=resolved_device,non_blocking=non_blocking,copy=copy_on_cpu)}
	def _get_torch_staging_cache(self,device:torch.device,*,pin_memory:bool)->dict[str,object]:
		use_pinned=bool(pin_memory and device.type=='cuda');schema_signature=tuple(sorted((key,tuple(shape))for(key,shape)in self.observation_schema.items()));key=self.batch_size,schema_signature,device.type,use_pinned,self.num_agents,self.action_dim;cache=self._torch_staging_cache.get(key)
		if cache is not None:return cache
		def _alloc_nested()->tuple[dict[str,torch.Tensor],dict[str,np.ndarray]]:tensors={name:torch.empty((self.batch_size,*shape),dtype=torch.float32,pin_memory=use_pinned)for(name,shape)in self.observation_schema.items()};return tensors,{name:tensor.numpy()for(name,tensor)in tensors.items()}
		obs_tensors,obs_numpy=_alloc_nested();next_obs_tensors,next_obs_numpy=_alloc_nested();scalars={name:torch.empty((self.batch_size,self.num_agents,1 if name!='action'else self.action_dim),dtype=torch.float32,pin_memory=use_pinned)for name in('action','reward','terminated','truncated','bootstrap_discount')};cache={'obs_tensors':obs_tensors,'obs_numpy':obs_numpy,'next_obs_tensors':next_obs_tensors,'next_obs_numpy':next_obs_numpy,'action_tensor':scalars['action'],'action_numpy':scalars['action'].numpy(),'reward_tensor':scalars['reward'],'reward_numpy':scalars['reward'].numpy(),'terminated_tensor':scalars['terminated'],'terminated_numpy':scalars['terminated'].numpy(),'truncated_tensor':scalars['truncated'],'truncated_numpy':scalars['truncated'].numpy(),'bootstrap_discount_tensor':scalars['bootstrap_discount'],'bootstrap_discount_numpy':scalars['bootstrap_discount'].numpy()};self._torch_staging_cache[key]=cache;return cache
def to_torch_batch(batch:dict[str,NestedArray],device:torch.device|str)->dict[str,NestedArray]:
	names=('obs','action','reward','next_obs','terminated','truncated')+(()if'bootstrap_discount'not in batch else('bootstrap_discount',))
	return{name:to_torch_nested(batch[name],device)for name in names}
