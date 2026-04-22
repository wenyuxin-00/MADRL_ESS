from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from controllers.madrl.safety_projector import JointGridSafetyProjector,_local_bounds_numpy,build_safety_local_numpy
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import WHOLESALE_PRICE_SEQ_FIELD,derive_import_price_seq,get_import_price_markup
_THROUGHPUT_TIEBREAKER_EUR_PER_KWH=1e-08
_NPOS_TIEBREAKER_EUR_PER_KWH=1e-09
_RHO_BALANCE_MU=1e1
_RHO_BALANCE_TAU=2.
@dataclass(frozen=True)
class AdmmMpcWindowData:import_price_eur_per_kwh:np.ndarray;load_seq:np.ndarray;pv_seq:np.ndarray;battery_capacity_kwh:np.ndarray;p_max_kw:np.ndarray;efficiency:float;energy_init_kwh:np.ndarray;energy_ref_kwh:np.ndarray;energy_min_kwh:np.ndarray;energy_max_kwh:np.ndarray;export_subsidy_eur_per_kwh:float;dt_hours:float
@dataclass(frozen=True)
class AdmmMpcSurrogateCache:trafo_limit_kw:float;trafo_base_kw:float;alpha_netload_window_kw:np.ndarray
@dataclass(frozen=True)
class AdmmMpcStepResult:executed_action_array:np.ndarray;converged:bool;iterations:int;final_primal_residual:float;final_dual_residual:float;solve_time_sec:float;rho_final:float
def _as_float32(values,*,name:str,ndim:int|None=None)->np.ndarray:
	array=np.asarray(values,dtype=np.float32)
	if ndim is not None and array.ndim!=ndim:raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}.")
	return array.astype(np.float32,copy=False)
def _day_episode_length(dt_hours:float)->int:
	resolved=int(round(24./float(dt_hours)))
	if abs(float(dt_hours)*resolved-24.)>1e-09:raise ValueError(f"ADMM MPC requires dt in hours with one full day per episode, got dt_hours={dt_hours} and resolved episode_length={resolved}.")
	return resolved
def build_admm_mpc_surrogate_cache(cfg,env,*,horizon_steps:int)->AdmmMpcSurrogateCache:
	projector=getattr(env,'_grid_notebook_projector',None)
	if not isinstance(projector,JointGridSafetyProjector):projector=JointGridSafetyProjector.from_cfg(cfg,device='cpu');setattr(env,'_grid_notebook_projector',projector)
	trafo_sensitivity=np.asarray(projector.trafo_power_sensitivity.detach().cpu().numpy(),dtype=np.float32)
	if trafo_sensitivity.size==0:alpha_window=np.zeros((int(projector.n_agents),int(horizon_steps)),dtype=np.float32)
	elif trafo_sensitivity.shape[0]!=1:raise ValueError(f"Notebook surrogate helpers expect exactly one transformer sensitivity row, got shape {trafo_sensitivity.shape}.")
	else:alpha_window=np.repeat(trafo_sensitivity[[0],:int(projector.n_agents)].T,int(horizon_steps),axis=1).astype(np.float32,copy=False)
	trafo_base_kw=np.asarray(projector.trafo_power_base_kw.detach().cpu().numpy(),dtype=np.float32).reshape(-1)
	if trafo_base_kw.size==0:raise ValueError('ADMM MPC surrogate requires transformer baseline data.')
	trafo_limit_kw=grid_nb._approx_trafo_limit_kw(env,loading_limit_pct=float(cfg.grid.line_max_loading_pct))
	if trafo_limit_kw is None or not np.isfinite(float(trafo_limit_kw)):raise ValueError('Unable to resolve a transformer power reference limit for ADMM MPC.')
	return AdmmMpcSurrogateCache(float(trafo_limit_kw),float(trafo_base_kw[0]),alpha_window)
def build_admm_mpc_window_data(cfg,env,raw_obs)->AdmmMpcWindowData:
	wholesale_price=_as_float32(raw_obs[WHOLESALE_PRICE_SEQ_FIELD],name=WHOLESALE_PRICE_SEQ_FIELD,ndim=1).reshape(-1);load_seq,pv_seq=_as_float32(raw_obs['load_seq'],name='load_seq',ndim=2),_as_float32(raw_obs['pv_seq'],name='pv_seq',ndim=2)
	if load_seq.shape!=pv_seq.shape:raise ValueError(f"load_seq shape {load_seq.shape} must match pv_seq shape {pv_seq.shape}.")
	n_agents,horizon=load_seq.shape
	if horizon!=wholesale_price.size:raise ValueError(f"load_seq horizon {horizon} must match {WHOLESALE_PRICE_SEQ_FIELD} horizon {wholesale_price.size}.")
	battery_capacity_kwh,p_max_kw=_as_float32(env.agent_c_bat,name='agent_c_bat',ndim=1).reshape(-1),_as_float32(env.agent_p_max,name='agent_p_max',ndim=1).reshape(-1)
	if battery_capacity_kwh.size!=n_agents or p_max_kw.size!=n_agents:raise ValueError(f"Battery arrays must match agent dimension {n_agents}, got {battery_capacity_kwh.size} and {p_max_kw.size}.")
	import_price=derive_import_price_seq(wholesale_price,markup_eur_per_kwh=float(get_import_price_markup(cfg)));export_subsidy=float(getattr(cfg.reward,'export_subsidy_eur_per_kwh',.079));min_gap=float(np.min(import_price-np.float32(export_subsidy)))
	if min_gap<-1e-09:raise ValueError(f"Rolling ADMM-MPC requires import_price_eur_per_kwh >= export_subsidy_eur_per_kwh, got min(import_price - subsidy)={min_gap:.6f}.")
	energy_init_kwh=(_as_float32(env.soc,name='env.soc',ndim=1).reshape(-1)*battery_capacity_kwh).astype(np.float32,copy=False)
	if energy_init_kwh.size!=n_agents:raise ValueError(f"env.soc agent dimension {energy_init_kwh.size} must match load_seq agents {n_agents}.")
	return AdmmMpcWindowData(import_price_eur_per_kwh=import_price,load_seq=load_seq,pv_seq=pv_seq,battery_capacity_kwh=battery_capacity_kwh,p_max_kw=p_max_kw,efficiency=float(env.eff),energy_init_kwh=energy_init_kwh,energy_ref_kwh=(float(cfg.env.soc_target)*battery_capacity_kwh).astype(np.float32,copy=False),energy_min_kwh=(float(env.soc_min)*battery_capacity_kwh).astype(np.float32,copy=False),energy_max_kwh=(float(env.soc_max)*battery_capacity_kwh).astype(np.float32,copy=False),export_subsidy_eur_per_kwh=export_subsidy,dt_hours=float(env.dt))
def resolve_default_terminal_cost_weight(window_data:AdmmMpcWindowData,*,multiplier:float=1.)->np.ndarray:mean_import_price=float(np.mean(np.asarray(window_data.import_price_eur_per_kwh,dtype=np.float32)));return np.asarray(float(multiplier)*2.*mean_import_price/np.maximum(np.asarray(window_data.battery_capacity_kwh,dtype=np.float32),1.),dtype=np.float32)
def _project_contribution_copies(values:np.ndarray,lower_bound_kw:np.ndarray)->np.ndarray:
	projected=_as_float32(values,name='values',ndim=2).copy();lower_bound=_as_float32(lower_bound_kw,name='lower_bound_kw',ndim=1).reshape(-1)
	if projected.shape[1]!=lower_bound.size:raise ValueError(f"values horizon {projected.shape[1]} must match lower_bound horizon {lower_bound.size}.")
	if projected.shape[0]>0:projected+=np.maximum(lower_bound-projected.sum(axis=0),.0).astype(np.float32,copy=False)/projected.shape[0]
	return projected.astype(np.float32,copy=False)
def _compute_default_rho(window_data:AdmmMpcWindowData,surrogate_cache:AdmmMpcSurrogateCache)->float:alpha_window=np.asarray(surrogate_cache.alpha_netload_window_kw,dtype=np.float32);baseline_net_load=(np.asarray(window_data.load_seq,dtype=np.float32)-np.asarray(window_data.pv_seq,dtype=np.float32)).astype(np.float32,copy=False);denominator=max(1.,float(np.mean(np.abs(alpha_window*baseline_net_load))));return float(np.mean(np.asarray(window_data.import_price_eur_per_kwh,dtype=np.float32))*float(window_data.dt_hours)/denominator)
class _ReusableAdmmLocalSolver:
	def __init__(self,*,agent_idx:int,horizon:int,p_max_kw:float,dt_hours:float,efficiency:float,energy_min_kwh:float,energy_max_kwh:float)->None:
		import gurobipy as gp;self.agent_idx=int(agent_idx);self.horizon=int(horizon);self.gp=gp;self.grb=gp.GRB;self.model=gp.Model(f"admm_mpc_local_agent_{self.agent_idx}");self.model.Params.OutputFlag=0;self.charge=self.model.addVars(self.horizon,lb=.0,ub=float(p_max_kw),name='charge_kw');self.discharge=self.model.addVars(self.horizon,lb=.0,ub=float(p_max_kw),name='discharge_kw');self.pv_curtail=self.model.addVars(self.horizon,lb=.0,name='pv_curtail_kw');self.net_load=self.model.addVars(self.horizon,lb=-self.grb.INFINITY,name='net_load_kw');self.n_pos=self.model.addVars(self.horizon,lb=.0,name='n_pos_kw');self.energy=self.model.addVars(self.horizon+1,lb=float(energy_min_kwh),ub=float(energy_max_kwh),name='energy_kwh');self._energy_init=self.model.addConstr(self.energy[0]==.0,name='energy_init');self._pv_curtail_upper=[];self._net_load_balance=[];dt=float(dt_hours);eff=max(float(efficiency),1e-06)
		for step_idx in range(self.horizon):self._pv_curtail_upper.append(self.model.addConstr(self.pv_curtail[step_idx]<=.0,name=f"curtail_ub_{step_idx}"));self._net_load_balance.append(self.model.addConstr(self.net_load[step_idx]-self.pv_curtail[step_idx]-self.charge[step_idx]+self.discharge[step_idx]==.0,name=f"net_load_balance_{step_idx}"));self.model.addConstr(self.n_pos[step_idx]>=self.net_load[step_idx],name=f"n_pos_lb_{step_idx}");self.model.addConstr(self.energy[step_idx+1]==self.energy[step_idx]+eff*dt*self.charge[step_idx]-dt/eff*self.discharge[step_idx],name=f"energy_balance_{step_idx}")
		self.model.ModelSense=self.grb.MINIMIZE;self.model.update()
	def dispose(self)->None:
		dispose=getattr(self.model,'dispose',None)
		if callable(dispose):dispose()
	def solve(self,*,window_data:AdmmMpcWindowData,alpha_kw:np.ndarray,z_kw:np.ndarray,u_kw:np.ndarray,rho:float,terminal_ref_kwh:float,terminal_cost_weight:float)->tuple[np.ndarray,np.float32,np.float32]:
		alpha=_as_float32(alpha_kw,name='alpha_kw',ndim=1).reshape(-1);z=_as_float32(z_kw,name='z_kw',ndim=1).reshape(-1);u=_as_float32(u_kw,name='u_kw',ndim=1).reshape(-1)
		if alpha.size!=self.horizon or z.size!=self.horizon or u.size!=self.horizon:raise ValueError(f"alpha/z/u must have horizon {self.horizon}, got {alpha.size}, {z.size}, {u.size}.")
		self._energy_init.RHS=float(window_data.energy_init_kwh[self.agent_idx]);import_minus_subsidy=np.maximum(np.asarray(window_data.import_price_eur_per_kwh,dtype=np.float32)-float(window_data.export_subsidy_eur_per_kwh),.0);objective=self.gp.QuadExpr()
		for step_idx in range(self.horizon):
			self._pv_curtail_upper[step_idx].RHS=float(window_data.pv_seq[self.agent_idx,step_idx]);self._net_load_balance[step_idx].RHS=float(window_data.load_seq[self.agent_idx,step_idx]-window_data.pv_seq[self.agent_idx,step_idx]);objective+=float(window_data.dt_hours*window_data.export_subsidy_eur_per_kwh)*self.net_load[step_idx];objective+=float(window_data.dt_hours*import_minus_subsidy[step_idx])*self.n_pos[step_idx];objective+=float(window_data.dt_hours*_THROUGHPUT_TIEBREAKER_EUR_PER_KWH)*(self.charge[step_idx]+self.discharge[step_idx]);objective+=float(window_data.dt_hours*_NPOS_TIEBREAKER_EUR_PER_KWH)*self.n_pos[step_idx]
			if abs(float(alpha[step_idx]))>1e-12 and float(rho)>.0:objective+=float(.5*rho*alpha[step_idx]*alpha[step_idx])*(self.net_load[step_idx]*self.net_load[step_idx]);objective+=float(rho*alpha[step_idx]*(u[step_idx]-z[step_idx]))*self.net_load[step_idx]
		if float(terminal_cost_weight)>.0:objective+=float(terminal_cost_weight)*(self.energy[self.horizon]*self.energy[self.horizon]-2.*float(terminal_ref_kwh)*self.energy[self.horizon])
		self.model.setObjective(objective,self.grb.MINIMIZE);self.model.optimize()
		if int(self.model.Status)not in{int(self.grb.OPTIMAL),int(self.grb.SUBOPTIMAL)}:raise RuntimeError(f"ADMM MPC local model for agent {self.agent_idx} failed with status={int(self.model.Status)}.")
		net_load_kw=np.asarray([self.net_load[t].X for t in range(self.horizon)],dtype=np.float32);return(alpha*net_load_kw).astype(np.float32,copy=False),np.float32(float(self.charge[0].X)-float(self.discharge[0].X)),np.float32(float(self.pv_curtail[0].X))
def _first_step_action(env,*,battery_power_kw:np.ndarray,pv_curtail_kw:np.ndarray,pv_step_kw:np.ndarray)->np.ndarray:requested_battery_power_kw=np.asarray(battery_power_kw,dtype=np.float32).reshape(-1);safety_local=build_safety_local_numpy(soc=np.asarray(env.soc,dtype=np.float32),load_raw=np.zeros_like(requested_battery_power_kw),pv_raw=np.zeros_like(requested_battery_power_kw),battery_capacity_kwh=np.asarray(env.agent_c_bat,dtype=np.float32),p_max_kw=np.asarray(env.agent_p_max,dtype=np.float32));battery_lower,battery_upper,p_max_kw,_=_local_bounds_numpy(safety_local,efficiency=float(env.eff),dt_hours=float(env.dt),soc_min=float(env.soc_min),soc_max=float(env.soc_max));battery_action=np.clip(np.clip(requested_battery_power_kw,battery_lower,battery_upper)/np.maximum(p_max_kw,1e-06),-1.,1.).astype(np.float32,copy=False);pv_raw_kw=np.maximum(np.asarray(pv_step_kw,dtype=np.float32).reshape(-1),.0);pv_effective_kw=np.maximum(pv_raw_kw-np.asarray(pv_curtail_kw,dtype=np.float32).reshape(-1),.0).astype(np.float32,copy=False);pv_utilization=np.ones_like(pv_raw_kw,dtype=np.float32);valid_mask=pv_raw_kw>1e-06;pv_utilization[valid_mask]=pv_effective_kw[valid_mask]/pv_raw_kw[valid_mask];pv_action=np.clip(2.*pv_utilization-1.,-1.,1.).astype(np.float32,copy=False);return np.stack([battery_action,pv_action],axis=-1).astype(np.float32,copy=False)
def run_admm_mpc_step(env,window_data:AdmmMpcWindowData,*,surrogate_cache:AdmmMpcSurrogateCache,rho_init:float,rho_min:float,rho_max:float,rho_adaptation:str|None,max_iters:int,max_iters_first_step:int,primal_tol:float,dual_tol:float,terminal_cost_weight_eur_per_kwh2:np.ndarray)->AdmmMpcStepResult:
	n_agents,horizon=window_data.load_seq.shape;alpha_window=_as_float32(surrogate_cache.alpha_netload_window_kw,name='alpha_netload_window_kw',ndim=2)
	if alpha_window.shape!=(n_agents,horizon):raise ValueError(f"alpha_netload_window_kw shape {alpha_window.shape} must match {(n_agents,horizon)}.")
	terminal_weight=_as_float32(terminal_cost_weight_eur_per_kwh2,name='terminal_cost_weight_eur_per_kwh2',ndim=1).reshape(-1)
	if terminal_weight.size!=n_agents:raise ValueError(f"terminal_cost_weight_eur_per_kwh2 must provide {n_agents} values, got {terminal_weight.size}.")
	lower_bound=np.full((horizon,),-float(surrogate_cache.trafo_limit_kw)-float(surrogate_cache.trafo_base_kw),dtype=np.float32);baseline_contrib=(alpha_window*(np.asarray(window_data.load_seq,dtype=np.float32)-np.asarray(window_data.pv_seq,dtype=np.float32))).astype(np.float32,copy=False);z_kw=_project_contribution_copies(baseline_contrib,lower_bound);u_kw=np.zeros_like(z_kw,dtype=np.float32);rho=float(np.clip(float(rho_init),float(rho_min),float(rho_max)));iteration_budget=int(max_iters_first_step)if int(getattr(env,'cur_step',0))==0 else int(max_iters);use_residual_balancing=str(rho_adaptation or'').strip().lower()=='residual_balancing';final_battery_power=final_pv_curtail=None;final_primal=float('inf');final_dual=float('inf');iterations_done=0;converged=False;started_at=perf_counter();local_solvers=tuple(_ReusableAdmmLocalSolver(agent_idx=agent_idx,horizon=horizon,p_max_kw=float(window_data.p_max_kw[agent_idx]),dt_hours=float(window_data.dt_hours),efficiency=float(window_data.efficiency),energy_min_kwh=float(window_data.energy_min_kwh[agent_idx]),energy_max_kwh=float(window_data.energy_max_kwh[agent_idx]))for agent_idx in range(n_agents))
	try:
		for iteration in range(1,iteration_budget+1):
			local_results=[solver.solve(window_data=window_data,alpha_kw=alpha_window[agent_idx],z_kw=z_kw[agent_idx],u_kw=u_kw[agent_idx],rho=float(rho),terminal_ref_kwh=float(window_data.energy_ref_kwh[agent_idx]),terminal_cost_weight=float(terminal_weight[agent_idx]))for(agent_idx,solver)in enumerate(local_solvers)];contribution_kw=np.asarray([result[0]for result in local_results],dtype=np.float32);final_battery_power=np.asarray([result[1]for result in local_results],dtype=np.float32);final_pv_curtail=np.asarray([result[2]for result in local_results],dtype=np.float32);z_next=_project_contribution_copies(contribution_kw+u_kw,lower_bound);u_kw=(u_kw+contribution_kw-z_next).astype(np.float32,copy=False);final_primal=float(np.linalg.norm((contribution_kw-z_next).reshape(-1),ord=2));final_dual=float(rho*np.linalg.norm((z_next-z_kw).reshape(-1),ord=2));iterations_done=int(iteration)
			if final_primal<=float(primal_tol)and final_dual<=float(dual_tol):converged=True;break
			if use_residual_balancing and rho>.0:
				rho_before=float(rho)
				if final_primal>_RHO_BALANCE_MU*final_dual:rho=float(np.clip(rho_before*_RHO_BALANCE_TAU,float(rho_min),float(rho_max)))
				elif final_dual>_RHO_BALANCE_MU*final_primal:rho=float(np.clip(rho_before/_RHO_BALANCE_TAU,float(rho_min),float(rho_max)))
				if not np.isclose(rho,rho_before):u_kw=(u_kw*np.float32(rho_before/max(rho,1e-09))).astype(np.float32,copy=False)
			z_kw=z_next
	finally:
		for solver in local_solvers:solver.dispose()
	if final_battery_power is None or final_pv_curtail is None:raise RuntimeError('ADMM MPC did not produce any iterate.')
	return AdmmMpcStepResult(executed_action_array=_first_step_action(env,battery_power_kw=final_battery_power,pv_curtail_kw=final_pv_curtail,pv_step_kw=window_data.pv_seq[:,0]),converged=bool(converged),iterations=int(iterations_done),final_primal_residual=float(final_primal),final_dual_residual=float(final_dual),solve_time_sec=float(perf_counter()-started_at),rho_final=float(rho))
