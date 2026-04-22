from __future__ import annotations
import copy,os,time
from abc import ABC,abstractmethod
from typing import Any
import numpy as np,torch,torch.nn.functional as F
from controllers.madrl.safety_projector import JointGridSafetyProjector
from models.assembly import build_actor_network,build_critic_network
from scripts.utils.replay_buffer import to_torch_batch
from scripts.utils.torch_runtime import add_batch_dim,to_torch_nested
class BaseAgent(ABC):
	def _init_shared_modules(self,cfg:object,agent_id:int)->None:self.cfg=cfg;self.device=cfg.runtime.device;self.num_agents=int(cfg.env.num_agents);self.agent_id=int(agent_id);self.max_action=float(cfg.model.max_action);self.action_dim=int(cfg.runtime.action_dim);self.gamma=float(cfg.algo.gamma);self.tau=float(cfg.algo.tau);self.use_grad_clip=bool(cfg.model.use_grad_clip);self.grad_clip_norm=float(cfg.model.grad_clip_norm);self.actor=build_actor_network(cfg,self.agent_id).to(self.device);self.critic=build_critic_network(cfg).to(self.device);self.actor_target=copy.deepcopy(self.actor);self.critic_target=copy.deepcopy(self.critic);self._configure_runtime_acceleration();self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=float(cfg.train.actor_lr));self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=float(cfg.train.critic_lr))
	def _configure_runtime_acceleration(self)->None:self._actor_forward=self.actor;self._critic_forward=self.critic;self._actor_target_forward=self.actor_target;self._critic_target_forward=self.critic_target;self.performance_summary={'amp_enabled':False,'amp_dtype':None,'amp_fallback_reason':'Single-path runtime disables AMP.','compile_enabled':False,'compile_mode':None,'compile_fallback_reason':'Single-path runtime disables torch.compile.','compiled_modules':[]}
	def _actor_call(self,obs_t:dict)->torch.Tensor:return self._actor_forward(obs_t)
	def _actor_target_call(self,obs_t:dict)->torch.Tensor:return self._actor_target_forward(obs_t)
	def _critic_call(self,obs_t:dict,action_t:torch.Tensor):return self._critic_forward(obs_t,action_t)
	def _critic_target_call(self,obs_t:dict,action_t:torch.Tensor):return self._critic_target_forward(obs_t,action_t)
	def _prepare_obs(self,obs:dict)->tuple[dict,bool]:
		has_batch_dim=obs['local'].ndim==3
		if not has_batch_dim:obs=add_batch_dim(obs)
		return to_torch_nested(obs,self.device),has_batch_dim
	def choose_action(self,obs:dict,noise_std:float)->np.ndarray:
		obs_t,has_batch_dim=self._prepare_obs(obs)
		with torch.inference_mode():action=self.act_from_torch_obs(obs_t,noise_std=noise_std).to(dtype=torch.float32).cpu().numpy()
		if not has_batch_dim:action=action[0]
		return action.astype(np.float32)
	def _soft_update(self)->None:
		for(p,tp)in zip(self.critic.parameters(),self.critic_target.parameters()):tp.data.copy_(self.tau*p.data+(1-self.tau)*tp.data)
		for(p,tp)in zip(self.actor.parameters(),self.actor_target.parameters()):tp.data.copy_(self.tau*p.data+(1-self.tau)*tp.data)
	def _optimizer_step(self,optimizer:torch.optim.Optimizer,loss:torch.Tensor,params)->None:
		optimizer.zero_grad();loss.backward()
		if self.use_grad_clip:torch.nn.utils.clip_grad_norm_(params,self.grad_clip_norm)
		optimizer.step()
	def save_model(self,model_dir:str,episode:int)->None:os.makedirs(model_dir,exist_ok=True);actor_path=os.path.join(model_dir,f"actor_agent_{self.agent_id}_ep_{episode}.pth");critic_path=os.path.join(model_dir,f"critic_agent_{self.agent_id}_ep_{episode}.pth");torch.save(self.actor.state_dict(),actor_path);torch.save(self.critic.state_dict(),critic_path)
	def load_model(self,model_dir:str,episode:int)->None:actor_path=os.path.join(model_dir,f"actor_agent_{self.agent_id}_ep_{episode}.pth");critic_path=os.path.join(model_dir,f"critic_agent_{self.agent_id}_ep_{episode}.pth");self.actor.load_state_dict(torch.load(actor_path,map_location=self.device));self.critic.load_state_dict(torch.load(critic_path,map_location=self.device));self.actor_target.load_state_dict(self.actor.state_dict());self.critic_target.load_state_dict(self.critic.state_dict())
	def act_from_torch_obs(self,obs_t:dict,noise_std:float)->torch.Tensor:
		action=self._actor_call(obs_t)
		if noise_std>.0:action=action+torch.randn_like(action)*float(noise_std)
		return action.clamp(-self.max_action,self.max_action)
	def train(self,replay_buffer:Any,agent_n:list)->None:self.train_on_batch(to_torch_batch(replay_buffer.sample(),self.device),agent_n)
	@abstractmethod
	def train_on_batch(self,batch:dict,agent_n:list,shared_ctx:dict|None=None)->None:0
class MADDPG(BaseAgent):
	def __init__(self,cfg:object,agent_id:int)->None:self._init_shared_modules(cfg,agent_id)
	def train_on_batch(self,batch:dict,agent_n:list,shared_ctx:dict|None=None)->None:
		obs=batch['obs'];action=batch['action'];reward=batch['reward'];next_obs=batch['next_obs'];done=batch['done']
		with torch.no_grad():
			next_action=None if shared_ctx is None else shared_ctx.get('target_actor_actions_clean')
			if next_action is None:next_action=torch.stack([agent._actor_target_call(next_obs)for agent in agent_n],dim=1)
			target_q=reward[:,self.agent_id]+self.gamma*(1-done[:,self.agent_id])*self._critic_target_call(next_obs,next_action)
		current_q=self._critic_call(obs,action);critic_loss=F.mse_loss(current_q.float(),target_q.float());self._optimizer_step(self.critic_optimizer,critic_loss,self.critic.parameters());new_action=action.clone();new_action[:,self.agent_id]=self._actor_call(obs);actor_loss=-self._critic_call(obs,new_action).float().mean();self._optimizer_step(self.actor_optimizer,actor_loss,self.actor.parameters());self._soft_update()
class MATD3(BaseAgent):
	def __init__(self,cfg:object,agent_id:int)->None:self._init_shared_modules(cfg,agent_id);self.policy_noise=float(cfg.algo.policy_noise);self.noise_clip=float(cfg.algo.noise_clip);self.policy_update_freq=int(cfg.algo.policy_update_freq);self.actor_pointer=0
	def _smoothed_target_actions(self,clean_next_action:torch.Tensor)->torch.Tensor:noise=(torch.randn_like(clean_next_action)*self.policy_noise).clamp(-self.noise_clip,self.noise_clip);return(clean_next_action+noise).clamp(-self.max_action,self.max_action)
	def train_on_batch(self,batch:dict,agent_n:list,shared_ctx:dict|None=None)->None:
		self.actor_pointer+=1;obs=batch['obs'];action=batch['action'];reward=batch['reward'];next_obs=batch['next_obs'];done=batch['done']
		with torch.no_grad():
			clean_next_action=None if shared_ctx is None else shared_ctx.get('target_actor_actions_clean')
			if clean_next_action is None:clean_next_action=torch.stack([agent._actor_target_call(next_obs)for agent in agent_n],dim=1)
			next_action=self._smoothed_target_actions(clean_next_action);q1_next,q2_next=self._critic_target_call(next_obs,next_action);target_q=reward[:,self.agent_id]+self.gamma*(1-done[:,self.agent_id])*torch.min(q1_next,q2_next)
		current_q1,current_q2=self._critic_call(obs,action);target_q_fp32=target_q.float();critic_loss=F.mse_loss(current_q1.float(),target_q_fp32)+F.mse_loss(current_q2.float(),target_q_fp32);self._optimizer_step(self.critic_optimizer,critic_loss,self.critic.parameters())
		if self.actor_pointer%self.policy_update_freq!=0:return
		new_action=action.clone();new_action[:,self.agent_id]=self._actor_call(obs);q1_policy,_=self._critic_call(obs,new_action);actor_loss=-q1_policy.float().mean();self._optimizer_step(self.actor_optimizer,actor_loss,self.actor.parameters());self._soft_update()
class MATD3SafePOC(MATD3):
	def __init__(self,cfg:object,agent_id:int)->None:super().__init__(cfg,agent_id);self.safety_projector=JointGridSafetyProjector.from_cfg(cfg,device=self.device);self.safety_enabled=True
	def _record_projection_event(self,shared_ctx:dict|None,*,stage:str,batch_size:int,elapsed_s:float)->None:
		if shared_ctx is None:return
		recorder=shared_ctx.get('projection_event_recorder')
		if recorder is not None:recorder(stage=stage,batch_size=batch_size,elapsed_s=elapsed_s)
	def _get_projected_target_actions(self,next_obs:dict,agent_n:list,shared_ctx:dict|None)->torch.Tensor:
		if shared_ctx is not None and shared_ctx.get('projected_target_actions')is not None:return shared_ctx['projected_target_actions']
		with torch.no_grad():
			clean_next_action=None if shared_ctx is None else shared_ctx.get('target_actor_actions_clean')
			if clean_next_action is None:clean_next_action=torch.stack([agent._actor_target_call(next_obs)for agent in agent_n],dim=1)
			next_action=self._smoothed_target_actions(clean_next_action);started=time.perf_counter();projected=self.safety_projector.project_actions_from_safety_local(next_obs['safety_local'],next_action)
		self._record_projection_event(shared_ctx,stage='target',batch_size=int(next_action.shape[0]),elapsed_s=time.perf_counter()-started)
		if shared_ctx is not None:shared_ctx['projected_target_actions']=projected
		return projected
	def _get_projected_policy_actions(self,obs:dict,action:torch.Tensor,agent_n:list,shared_ctx:dict|None)->torch.Tensor:
		if shared_ctx is not None and shared_ctx.get('projected_policy_actions_all')is not None:return shared_ctx['projected_policy_actions_all']
		batch_size=int(action.shape[0]);n_agents=int(len(agent_n));candidate_joint_actions=action.unsqueeze(0).expand(n_agents,-1,-1,-1).clone()
		for(agent_id,agent)in enumerate(agent_n):candidate_joint_actions[agent_id,:,agent_id]=agent._actor_call(obs)
		flat_candidate_actions=candidate_joint_actions.reshape(n_agents*batch_size,self.num_agents,self.action_dim);safety_local=obs['safety_local'].unsqueeze(0).expand(n_agents,-1,-1,-1).reshape(n_agents*batch_size,self.num_agents,obs['safety_local'].shape[-1]);started=time.perf_counter();projected=self.safety_projector.project_actions_from_safety_local(safety_local,flat_candidate_actions);self._record_projection_event(shared_ctx,stage='actor',batch_size=int(flat_candidate_actions.shape[0]),elapsed_s=time.perf_counter()-started);projected_policy_actions=projected.reshape(n_agents,batch_size,self.num_agents,self.action_dim)
		if shared_ctx is not None:shared_ctx['projected_policy_actions_all']=projected_policy_actions
		return projected_policy_actions
	def train_on_batch(self,batch:dict,agent_n:list,shared_ctx:dict|None=None)->None:
		self.actor_pointer+=1;obs=batch['obs'];action=batch['action'];reward=batch['reward'];next_obs=batch['next_obs'];done=batch['done']
		with torch.no_grad():projected_next_action=self._get_projected_target_actions(next_obs,agent_n,shared_ctx);q1_next,q2_next=self._critic_target_call(next_obs,projected_next_action);target_q=reward[:,self.agent_id]+self.gamma*(1-done[:,self.agent_id])*torch.min(q1_next,q2_next)
		current_q1,current_q2=self._critic_call(obs,action);target_q_fp32=target_q.float();critic_loss=F.mse_loss(current_q1.float(),target_q_fp32)+F.mse_loss(current_q2.float(),target_q_fp32);self._optimizer_step(self.critic_optimizer,critic_loss,self.critic.parameters())
		if self.actor_pointer%self.policy_update_freq!=0:return
		projected_policy_actions=self._get_projected_policy_actions(obs,action,agent_n,shared_ctx);q1_policy,_=self._critic_call(obs,projected_policy_actions[self.agent_id]);actor_loss=-q1_policy.float().mean()
		if shared_ctx is None:self._optimizer_step(self.actor_optimizer,actor_loss,self.actor.parameters());self._soft_update();return
		self.actor_optimizer.zero_grad();shared_ctx['actor_backward_count']=int(shared_ctx.get('actor_backward_count',0))+1;actor_loss.backward(retain_graph=self.agent_id<len(agent_n)-1)
		if shared_ctx['actor_backward_count']<len(agent_n):return
		for agent in agent_n:
			if getattr(agent,'use_grad_clip',False):torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),agent.grad_clip_norm)
			agent.actor_optimizer.step();agent._soft_update()
def get_agent_cls(name:str)->type[BaseAgent]:
	normalized=str(name)
	if normalized=='MADDPG':return MADDPG
	if normalized=='MATD3':return MATD3
	if normalized=='MATD3_SAFE_POC':return MATD3SafePOC
	raise ValueError(f"Unknown algorithm '{normalized}', available: ['MADDPG', 'MATD3', 'MATD3_SAFE_POC']")
