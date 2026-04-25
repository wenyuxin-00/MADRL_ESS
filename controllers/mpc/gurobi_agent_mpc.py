from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np
_GUROBI_ERROR_PREFIX='MPC rollout requires a working Gurobi installation/license'
LOCAL_MPC_TIME_LIMIT_SEC=2.
LOCAL_MPC_MIP_GAP=0.01
BATTERY_THROUGHPUT_TIEBREAKER_EPS_EUR_PER_KWH=5e-05
def _load_gurobi():import gurobipy as gp;gp.setParam('OutputFlag',0);return gp,gp.GRB
def _create_model(gp:Any):return gp.Model('local_mpc')
def _wrap_gurobi_error(detail:str,exc:Exception)->RuntimeError:error=RuntimeError(f"{_GUROBI_ERROR_PREFIX}: {detail}: {exc}");error.__cause__=exc;return error
@dataclass(frozen=True)
class _LocalMPCPrimalSolution:charge_kw:np.ndarray;discharge_kw:np.ndarray;pv_curtail_kw:np.ndarray;energy_kwh:np.ndarray
@dataclass(frozen=True)
class LocalMPCFullHorizonResult:charge_kw:np.ndarray;discharge_kw:np.ndarray;pv_curtail_kw:np.ndarray;pv_effective_kw:np.ndarray;pv_utilization:np.ndarray;signed_battery_kw:np.ndarray;net_load_kw:np.ndarray;energy_kwh:np.ndarray;objective_eur:float;solve_time_sec:float;feasible:bool;grid_import_kw:np.ndarray;grid_export_kw:np.ndarray;net_load_floor_kw:np.ndarray|None=None
def _empty_full_horizon_result(horizon:int,*,solve_time_sec:float=.0,net_load_floor_kw:np.ndarray|None=None)->LocalMPCFullHorizonResult:empty_horizon=np.zeros((max(int(horizon),0),),dtype=np.float32);empty_energy=np.zeros((max(int(horizon)+1,0),),dtype=np.float32);floor=None if net_load_floor_kw is None else np.asarray(net_load_floor_kw,dtype=np.float32).reshape(-1).copy();return LocalMPCFullHorizonResult(charge_kw=empty_horizon.copy(),discharge_kw=empty_horizon.copy(),pv_curtail_kw=empty_horizon.copy(),pv_effective_kw=empty_horizon.copy(),pv_utilization=empty_horizon.copy(),signed_battery_kw=empty_horizon.copy(),net_load_kw=empty_horizon.copy(),energy_kwh=empty_energy,objective_eur=.0,solve_time_sec=float(solve_time_sec),feasible=False,grid_import_kw=empty_horizon.copy(),grid_export_kw=empty_horizon.copy(),net_load_floor_kw=floor)
def _sanitize_problem_inputs(*,import_price_seq:np.ndarray,load_seq:np.ndarray,pv_seq:np.ndarray,soc:float,battery_capacity_kwh:float,p_max_kw:float,dt_hours:float,efficiency:float,soc_min:float,soc_max:float)->dict[str,Any]|None:
	capacity=float(battery_capacity_kwh);power_limit=float(p_max_kw);dt=float(dt_hours)
	if capacity<=.0 or power_limit<=.0 or dt<=.0:return
	prices=np.asarray(import_price_seq,dtype=np.float32).reshape(-1);load=np.asarray(load_seq,dtype=np.float32).reshape(-1);pv=np.asarray(pv_seq,dtype=np.float32).reshape(-1);horizon=int(min(len(prices),len(load),len(pv)))
	if horizon<=0:return
	prices=prices[:horizon].astype(np.float32,copy=False);load=load[:horizon].astype(np.float32,copy=False);pv=pv[:horizon].astype(np.float32,copy=False);net_load=(load-pv).astype(np.float32,copy=False);eff=max(float(efficiency),1e-06);energy_min=float(np.clip(soc_min,.0,1.))*capacity;energy_max=float(np.clip(soc_max,.0,1.))*capacity
	if energy_max<=energy_min+1e-09:return
	energy_now=float(np.clip(soc,soc_min,soc_max))*capacity;return{'prices':prices,'pv':pv,'net_load':net_load,'horizon':horizon,'capacity':capacity,'power_limit':power_limit,'dt':dt,'eff':eff,'energy_now':energy_now}
def _shift_primal_solution_start(solution:_LocalMPCPrimalSolution,*,current_energy_kwh:float)->_LocalMPCPrimalSolution:
	def _shift_control(values:np.ndarray)->np.ndarray:array=np.asarray(values,dtype=np.float32).reshape(-1);return array.copy()if array.size==0 else np.concatenate([array[1:],np.zeros((1,),dtype=np.float32)]).astype(np.float32,copy=False)
	energy=np.asarray(solution.energy_kwh,dtype=np.float32).reshape(-1);shifted_energy=energy.copy()
	if shifted_energy.size:
		shifted_energy[0]=np.float32(current_energy_kwh)
		if shifted_energy.size>2:shifted_energy[1:-1]=energy[2:]
		if shifted_energy.size>1:shifted_energy[-1]=energy[-2]
	return _LocalMPCPrimalSolution(charge_kw=_shift_control(solution.charge_kw),discharge_kw=_shift_control(solution.discharge_kw),pv_curtail_kw=_shift_control(solution.pv_curtail_kw),energy_kwh=shifted_energy)
def _sanitize_net_load_floor(net_load_floor_kw:np.ndarray|None,*,horizon:int)->np.ndarray|None:
	if net_load_floor_kw is None:return
	floor=np.asarray(net_load_floor_kw,dtype=np.float32).reshape(-1)
	if int(floor.size)!=int(horizon):raise ValueError(f"net_load_floor_kw must match the MPC horizon, got {floor.size} values for horizon={horizon}.")
	if not np.all(np.isfinite(floor)):raise ValueError('net_load_floor_kw must contain only finite values.')
	return floor.astype(np.float32,copy=False)
def _sanitize_pv_curtail_upper(pv_curtail_upper_kw:np.ndarray|None,*,pv_available_kw:np.ndarray,horizon:int)->np.ndarray:
	pv_available=np.maximum(np.asarray(pv_available_kw,dtype=np.float32).reshape(-1),.0)
	if int(pv_available.size)!=int(horizon):raise ValueError(f"pv_available_kw must match the MPC horizon, got {pv_available.size} values for horizon={horizon}.")
	if pv_curtail_upper_kw is None:return np.zeros((horizon,),dtype=np.float32)
	curtail_upper=np.asarray(pv_curtail_upper_kw,dtype=np.float32).reshape(-1)
	if int(curtail_upper.size)!=int(horizon):raise ValueError(f"pv_curtail_upper_kw must match the MPC horizon, got {curtail_upper.size} values for horizon={horizon}.")
	if not np.all(np.isfinite(curtail_upper)):raise ValueError('pv_curtail_upper_kw must contain only finite values.')
	return np.minimum(np.maximum(curtail_upper,.0),pv_available).astype(np.float32,copy=False)
class _ReusableLocalMPCSolver:
	def __init__(self,*,horizon:int,battery_capacity_kwh:float,p_max_kw:float,dt_hours:float,efficiency:float,soc_min:float,soc_max:float,export_subsidy_eur_per_kwh:float)->None:
		self.horizon=int(horizon);self.capacity=float(battery_capacity_kwh);self.power_limit=float(p_max_kw);self.dt=float(dt_hours);self.eff=max(float(efficiency),1e-06);self.soc_min=float(soc_min);self.soc_max=float(soc_max);self.energy_min=float(np.clip(self.soc_min,.0,1.))*self.capacity;self.energy_max=float(np.clip(self.soc_max,.0,1.))*self.capacity;self.export_subsidy_eur_per_kwh=float(export_subsidy_eur_per_kwh);(self._last_solution):_LocalMPCPrimalSolution|None=None
		try:self._gp,self._grb=_load_gurobi()
		except ModuleNotFoundError as exc:raise RuntimeError(f"{_GUROBI_ERROR_PREFIX}: gurobipy import failed")from exc
		try:self.model=_create_model(self._gp);self.model.Params.OutputFlag=0
		except Exception as exc:raise _wrap_gurobi_error('unable to create Gurobi model',exc)
		self.model.Params.TimeLimit=LOCAL_MPC_TIME_LIMIT_SEC;self.model.Params.MIPGap=LOCAL_MPC_MIP_GAP;self.charge=self.model.addVars(self.horizon,lb=.0,ub=self.power_limit,name='p_charge');self.discharge=self.model.addVars(self.horizon,lb=.0,ub=self.power_limit,name='p_discharge');self.pv_curtail=self.model.addVars(self.horizon,lb=.0,name='pv_curtail');self.energy=self.model.addVars(self.horizon+1,lb=self.energy_min,ub=self.energy_max,name='energy');self.grid_import=self.model.addVars(self.horizon,lb=.0,name='grid_import');self.grid_export=self.model.addVars(self.horizon,lb=.0,name='grid_export');self.grid_mode=self.model.addVars(self.horizon,vtype=self._grb.BINARY,name='grid_mode');(self._grid_balance_constrs):list[Any]=[];(self._grid_import_ub_constrs):list[Any]=[];(self._grid_export_ub_constrs):list[Any]=[];(self._fallback_import_gate_constrs):list[Any]=[];(self._fallback_export_gate_constrs):list[Any]=[];(self._net_load_floor_constrs):list[Any]=[];(self._pv_curtail_upper_constrs):list[Any]=[];self._energy_init_constr=self.model.addConstr(self.energy[0]==.0,name='energy_init')
		for step_idx in range(self.horizon):self.model.addConstr(self.energy[step_idx+1]==self.energy[step_idx]+self.eff*self.charge[step_idx]*self.dt-self.discharge[step_idx]/self.eff*self.dt,name=f"energy_balance_{step_idx}");self._net_load_floor_constrs.append(self.model.addConstr(self.charge[step_idx]-self.discharge[step_idx]+self.pv_curtail[step_idx]>=-2.*self.power_limit,name=f"net_load_floor_{step_idx}"));self._pv_curtail_upper_constrs.append(self.model.addConstr(self.pv_curtail[step_idx]<=.0,name=f"pv_curtail_ub_{step_idx}"))
		for step_idx in range(self.horizon):self._grid_balance_constrs.append(self.model.addConstr(self.grid_import[step_idx]-self.grid_export[step_idx]-self.charge[step_idx]+self.discharge[step_idx]-self.pv_curtail[step_idx]==.0,name=f"grid_balance_{step_idx}"));self._grid_import_ub_constrs.append(self.model.addConstr(self.grid_import[step_idx]<=.0,name=f"grid_import_ub_{step_idx}"));self._grid_export_ub_constrs.append(self.model.addConstr(self.grid_export[step_idx]<=.0,name=f"grid_export_ub_{step_idx}"));self._fallback_import_gate_constrs.append(self.model.addConstr(self.grid_import[step_idx]-self.grid_mode[step_idx]<=.0,name=f"grid_import_gate_{step_idx}"));self._fallback_export_gate_constrs.append(self.model.addConstr(self.grid_export[step_idx]+self.grid_mode[step_idx]<=1.,name=f"grid_export_gate_{step_idx}"))
		self.model.ModelSense=self._grb.MINIMIZE;self.model.update()
	def dispose(self)->None:
		dispose=getattr(self.model,'dispose',None)
		if callable(dispose):dispose()
	def _apply_problem_data(self,*,prices:np.ndarray,net_load:np.ndarray,pv_available_kw:np.ndarray,energy_now:float,net_load_floor_kw:np.ndarray|None=None,pv_curtail_upper_kw:np.ndarray|None=None)->None:
		self._energy_init_constr.RHS=float(energy_now);self.energy[0].Obj=.0;floor=_sanitize_net_load_floor(net_load_floor_kw,horizon=self.horizon);curtail_upper=_sanitize_pv_curtail_upper(pv_curtail_upper_kw,pv_available_kw=pv_available_kw,horizon=self.horizon)
		for step_idx in range(self.horizon):
			price_t=float(prices[step_idx]);throughput_penalty=BATTERY_THROUGHPUT_TIEBREAKER_EPS_EUR_PER_KWH*self.dt;gross_flow_limit_kw=abs(float(net_load[step_idx]))+self.power_limit+float(curtail_upper[step_idx])
			if floor is None:self._net_load_floor_constrs[step_idx].RHS=-2.*self.power_limit
			else:self._net_load_floor_constrs[step_idx].RHS=float(floor[step_idx]-net_load[step_idx])
			self._pv_curtail_upper_constrs[step_idx].RHS=float(curtail_upper[step_idx]);self._grid_balance_constrs[step_idx].RHS=float(net_load[step_idx]);self._grid_import_ub_constrs[step_idx].RHS=gross_flow_limit_kw;self._grid_export_ub_constrs[step_idx].RHS=gross_flow_limit_kw;self.charge[step_idx].Obj=price_t*self.dt+throughput_penalty;self.discharge[step_idx].Obj=-price_t*self.dt+throughput_penalty;self.pv_curtail[step_idx].Obj=.0;self.energy[step_idx+1].Obj=.0;self.model.chgCoeff(self._fallback_import_gate_constrs[step_idx],self.grid_mode[step_idx],-gross_flow_limit_kw);self._fallback_import_gate_constrs[step_idx].RHS=.0;self.model.chgCoeff(self._fallback_export_gate_constrs[step_idx],self.grid_mode[step_idx],gross_flow_limit_kw);self._fallback_export_gate_constrs[step_idx].RHS=gross_flow_limit_kw;self.grid_import[step_idx].Obj=.0;self.grid_export[step_idx].Obj=.0;self.grid_mode[step_idx].Obj=.0
	def _apply_warm_start(self,*,energy_now:float)->None:
		if self._last_solution is None:return
		shifted=_shift_primal_solution_start(self._last_solution,current_energy_kwh=energy_now)
		for step_idx in range(self.horizon):self.charge[step_idx].Start=float(shifted.charge_kw[step_idx]);self.discharge[step_idx].Start=float(shifted.discharge_kw[step_idx]);self.pv_curtail[step_idx].Start=float(shifted.pv_curtail_kw[step_idx])
		for step_idx in range(self.horizon+1):self.energy[step_idx].Start=float(shifted.energy_kwh[step_idx])
	def _status_has_usable_solution(self)->bool:
		status=int(self.model.Status)
		if status==int(self._grb.OPTIMAL):return True
		sol_count=int(getattr(self.model,'SolCount',0));return sol_count>0 and status in{int(self._grb.TIME_LIMIT),int(self._grb.SUBOPTIMAL)}
	def _extract_solution(self)->_LocalMPCPrimalSolution:charge_kw=np.asarray([self.charge[idx].X for idx in range(self.horizon)],dtype=np.float32);discharge_kw=np.asarray([self.discharge[idx].X for idx in range(self.horizon)],dtype=np.float32);pv_curtail_kw=np.asarray([self.pv_curtail[idx].X for idx in range(self.horizon)],dtype=np.float32);energy_kwh=np.asarray([self.energy[idx].X for idx in range(self.horizon+1)],dtype=np.float32);return _LocalMPCPrimalSolution(charge_kw=charge_kw,discharge_kw=discharge_kw,pv_curtail_kw=pv_curtail_kw,energy_kwh=energy_kwh)
	def _solve_prepared(self,*,prepared:dict[str,Any]|None,net_load_floor_kw:np.ndarray|None=None,pv_curtail_upper_kw:np.ndarray|None=None)->LocalMPCFullHorizonResult:
		if prepared is None:return _empty_full_horizon_result(self.horizon,net_load_floor_kw=net_load_floor_kw)
		if int(prepared['horizon'])!=self.horizon:raise ValueError(f"Reusable local MPC solver was built with a different horizon: expected {self.horizon}, got {prepared['horizon']}.")
		prices=prepared['prices'];pv_available_kw=np.maximum(np.asarray(prepared['pv'],dtype=np.float32),.0);net_load=prepared['net_load'];energy_now=float(prepared['energy_now']);floor=_sanitize_net_load_floor(net_load_floor_kw,horizon=self.horizon);curtail_upper=_sanitize_pv_curtail_upper(pv_curtail_upper_kw,pv_available_kw=pv_available_kw,horizon=self.horizon);self._apply_problem_data(prices=prices,net_load=net_load,pv_available_kw=pv_available_kw,energy_now=energy_now,net_load_floor_kw=floor,pv_curtail_upper_kw=curtail_upper);self._apply_warm_start(energy_now=energy_now);started_at=perf_counter()
		try:self.model.optimize()
		except Exception as exc:raise _wrap_gurobi_error('Gurobi optimize failed',exc)
		solve_time_sec=float(getattr(self.model,'Runtime',perf_counter()-started_at))
		if not self._status_has_usable_solution():return _empty_full_horizon_result(self.horizon,solve_time_sec=solve_time_sec,net_load_floor_kw=floor)
		solution=self._extract_solution();self._last_solution=solution;pv_curtail_kw=np.minimum(np.maximum(solution.pv_curtail_kw,.0),curtail_upper).astype(np.float32);pv_effective_kw=np.maximum(pv_available_kw-pv_curtail_kw,.0).astype(np.float32);pv_utilization=np.ones_like(pv_available_kw,dtype=np.float32);valid_pv_mask=pv_available_kw>1e-06;pv_utilization[valid_pv_mask]=np.clip(pv_effective_kw[valid_pv_mask]/pv_available_kw[valid_pv_mask],.0,1.).astype(np.float32);signed_battery_kw=(solution.charge_kw-solution.discharge_kw).astype(np.float32);net_grid_kw=(net_load+pv_curtail_kw+signed_battery_kw).astype(np.float32);grid_import_kw=np.maximum(net_grid_kw,.0).astype(np.float32);grid_export_kw=np.maximum(-net_grid_kw,.0).astype(np.float32);objective_eur=float(np.sum((prices*solution.charge_kw-prices*solution.discharge_kw)*self.dt));return LocalMPCFullHorizonResult(charge_kw=solution.charge_kw.copy(),discharge_kw=solution.discharge_kw.copy(),pv_curtail_kw=pv_curtail_kw.copy(),pv_effective_kw=pv_effective_kw.copy(),pv_utilization=pv_utilization.copy(),signed_battery_kw=signed_battery_kw,net_load_kw=net_grid_kw,energy_kwh=solution.energy_kwh.copy(),objective_eur=objective_eur,solve_time_sec=solve_time_sec,feasible=True,grid_import_kw=grid_import_kw,grid_export_kw=grid_export_kw,net_load_floor_kw=None if floor is None else floor.copy())
	def solve_full_horizon(self,*,import_price_seq:np.ndarray,load_seq:np.ndarray,pv_seq:np.ndarray,soc:float,pv_curtail_upper_kw:np.ndarray|None=None)->LocalMPCFullHorizonResult:prepared=_sanitize_problem_inputs(import_price_seq=import_price_seq,load_seq=load_seq,pv_seq=pv_seq,soc=soc,battery_capacity_kwh=self.capacity,p_max_kw=self.power_limit,dt_hours=self.dt,efficiency=self.eff,soc_min=self.soc_min,soc_max=self.soc_max);return self._solve_prepared(prepared=prepared,pv_curtail_upper_kw=pv_curtail_upper_kw)
def solve_local_gurobi_mpc_full_horizon(*,import_price_seq:np.ndarray,load_seq:np.ndarray,pv_seq:np.ndarray,soc:float,battery_capacity_kwh:float,p_max_kw:float,dt_hours:float,efficiency:float,soc_min:float,soc_max:float,export_subsidy_eur_per_kwh:float,net_load_floor_kw:np.ndarray|None=None,pv_curtail_upper_kw:np.ndarray|None=None)->LocalMPCFullHorizonResult:
	prepared=_sanitize_problem_inputs(import_price_seq=import_price_seq,load_seq=load_seq,pv_seq=pv_seq,soc=soc,battery_capacity_kwh=battery_capacity_kwh,p_max_kw=p_max_kw,dt_hours=dt_hours,efficiency=efficiency,soc_min=soc_min,soc_max=soc_max)
	if prepared is None:return _empty_full_horizon_result(0,net_load_floor_kw=net_load_floor_kw)
	solver=_ReusableLocalMPCSolver(horizon=int(prepared['horizon']),battery_capacity_kwh=float(prepared['capacity']),p_max_kw=float(prepared['power_limit']),dt_hours=float(prepared['dt']),efficiency=float(prepared['eff']),soc_min=float(soc_min),soc_max=float(soc_max),export_subsidy_eur_per_kwh=float(export_subsidy_eur_per_kwh))
	try:return solver._solve_prepared(prepared=prepared,net_load_floor_kw=None if net_load_floor_kw is None else np.asarray(net_load_floor_kw,dtype=np.float32)[:int(prepared['horizon'])],pv_curtail_upper_kw=None if pv_curtail_upper_kw is None else np.asarray(pv_curtail_upper_kw,dtype=np.float32)[:int(prepared['horizon'])])
	finally:solver.dispose()
def solve_local_gurobi_mpc_action(*,import_price_seq:np.ndarray,load_seq:np.ndarray,pv_seq:np.ndarray,soc:float,battery_capacity_kwh:float,p_max_kw:float,dt_hours:float,efficiency:float,soc_min:float,soc_max:float,export_subsidy_eur_per_kwh:float)->float:
	result=solve_local_gurobi_mpc_full_horizon(import_price_seq=import_price_seq,load_seq=load_seq,pv_seq=pv_seq,soc=soc,battery_capacity_kwh=battery_capacity_kwh,p_max_kw=p_max_kw,dt_hours=dt_hours,efficiency=efficiency,soc_min=soc_min,soc_max=soc_max,export_subsidy_eur_per_kwh=export_subsidy_eur_per_kwh)
	if not result.feasible or result.signed_battery_kw.size==0:return .0
	return .0 if abs(float(result.signed_battery_kw[0]))<1e-08 else float(result.signed_battery_kw[0])
