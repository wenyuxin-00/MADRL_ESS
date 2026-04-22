from __future__ import annotations
import dataclasses,warnings
from dataclasses import dataclass,field
from typing import TYPE_CHECKING,Any
import numpy as np
from envs.grid.core.net_builder import build_simbench_net
if TYPE_CHECKING:from envs.grid.deployments import AgentDeployment
@dataclass
class GridStepResult:converged:bool;vm_pu:np.ndarray;line_loading_pct:np.ndarray;trafo_loading_pct:np.ndarray;v_violation:np.ndarray;trafo_p_signed_kw:np.ndarray=field(default_factory=lambda:np.zeros(0,dtype=np.float32));psi_v_raw:float=.0;psi_line_raw:float=.0;psi_trafo_raw:float=.0
class GridCore:
	def __init__(self,deployments:list['AgentDeployment'],grid_cfg:Any)->None:
		self.deployments=deployments;self.grid_cfg=grid_cfg;self.n_agents=len(deployments);self.agent_bus_ids=[deployment.bus_id for deployment in deployments];self.net=build_simbench_net(grid_cfg.sb_code);self.n_buses=int(len(self.net.bus));self.n_lines=int(len(self.net.line));self.n_trafos=int(len(getattr(self.net,'trafo',[])));bus_id_to_pos={bus_id:pos for(pos,bus_id)in enumerate(self.net.bus.index.tolist())};missing=[bus_id for bus_id in self.agent_bus_ids if bus_id not in bus_id_to_pos]
		if missing:raise ValueError(f"agent_bus_id(s) {missing} are not present in network bus index.")
		self._agent_bus_pos=np.asarray([bus_id_to_pos[bus_id]for bus_id in self.agent_bus_ids],dtype=np.int64);self._bus_rows=self._ensure_agent_bus_rows();self._last_valid=self._make_zero_result();self.last_pf_error=''
	def _ensure_agent_bus_rows(self)->list[tuple[int,int,int,float,float]]:
		import pandapower as pp;rows:list[tuple[int,int,int,float,float]]=[]
		for bus_id in self.agent_bus_ids:load_bus=self.net.load['bus']==bus_id;sgen_bus=self.net.sgen['bus']==bus_id;load_idx=int(self.net.load.index[load_bus][0])if load_bus.any()else int(pp.create_load(self.net,bus=bus_id,p_mw=.0,q_mvar=.0));sgen_idx=int(self.net.sgen.index[sgen_bus][0])if sgen_bus.any()else int(pp.create_sgen(self.net,bus=bus_id,p_mw=.0,q_mvar=.0));rows.append((int(bus_id),load_idx,sgen_idx,float(self.net.load.at[load_idx,'p_mw']),float(self.net.sgen.at[sgen_idx,'p_mw'])))
		return rows
	def reset(self,base_load_kw:np.ndarray,base_pv_kw:np.ndarray)->None:del base_load_kw,base_pv_kw;self._restore_agent_buses();self._last_valid=self._make_zero_result();self.last_pf_error=''
	def step(self,p_batt_kw:np.ndarray,base_load_kw:np.ndarray)->GridStepResult:
		self._restore_agent_buses();self._apply_agent_injections(-(np.asarray(base_load_kw,dtype=np.float32)+np.asarray(p_batt_kw,dtype=np.float32)))
		try:self._runpp();self.last_pf_error='';self._last_valid=self._extract_result(converged=True);return self._last_valid
		except Exception as exc:self.last_pf_error=f"{type(exc).__name__}: {exc}";return dataclasses.replace(self._last_valid,converged=False)
	def _restore_agent_buses(self)->None:
		for(_,load_idx,sgen_idx,load_p_mw,sgen_p_mw)in self._bus_rows:self.net.load.at[load_idx,'p_mw']=load_p_mw;self.net.sgen.at[sgen_idx,'p_mw']=sgen_p_mw
	def _apply_agent_injections(self,p_inject_kw:np.ndarray)->None:
		for((bus_id,load_idx,sgen_idx,_,_),p_kw)in zip(self._bus_rows,np.asarray(p_inject_kw,dtype=np.float32),strict=False):p_mw=float(p_kw)/1e3;self.net.sgen.at[sgen_idx,'p_mw']=max(.0,p_mw);self.net.sgen.at[sgen_idx,'q_mvar']=.0;self.net.load.at[load_idx,'p_mw']=max(.0,-p_mw);self.net.load.at[load_idx,'q_mvar']=.0
	def _runpp(self)->None:
		import pandapower as pp;attempts={'algorithm':self.grid_cfg.pf_solver,'init':'flat'},{'algorithm':self.grid_cfg.pf_solver,'init':'auto'},{'algorithm':'bfsw','init':'flat'};last_exc:Exception|None=None
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for extra_kwargs in attempts:
				try:pp.runpp(self.net,verbose=False,numba=False,lightsim2grid=False,calculate_voltage_angles=False,voltage_depend_loads=False,**extra_kwargs);return
				except Exception as exc:last_exc=exc
		if last_exc is not None:raise last_exc
		raise RuntimeError('pandapower.runpp failed without exposing an exception.')
	def _extract_result(self,*,converged:bool)->GridStepResult:
		vm_pu=self.net.res_bus['vm_pu'].to_numpy(dtype=np.float32);line_loading_pct=self.net.res_line['loading_percent'].to_numpy(dtype=np.float32);trafo_loading_pct=np.zeros(self.n_trafos,dtype=np.float32);trafo_p_signed_kw=np.zeros(self.n_trafos,dtype=np.float32)
		if self.n_trafos>0 and hasattr(self.net,'res_trafo')and not self.net.res_trafo.empty:
			trafo_loading_pct=self.net.res_trafo['loading_percent'].to_numpy(dtype=np.float32)
			if'p_hv_mw'in self.net.res_trafo.columns:trafo_p_signed_kw=self.net.res_trafo['p_hv_mw'].to_numpy(dtype=np.float32)*1e3
		v_min=float(self.grid_cfg.v_min_pu);v_max=float(self.grid_cfg.v_max_pu);limit=float(self.grid_cfg.line_max_loading_pct);agent_vm_pu=np.take(vm_pu,self._agent_bus_pos);v_violation=(np.maximum(.0,v_min-agent_vm_pu)+np.maximum(.0,agent_vm_pu-v_max)).astype(np.float32);bus_v_excess=(np.maximum(.0,v_min-vm_pu)+np.maximum(.0,vm_pu-v_max)).astype(np.float32);line_excess=(np.maximum(.0,line_loading_pct-limit)/1e2).astype(np.float32);trafo_excess=(np.maximum(.0,trafo_loading_pct-limit)/1e2).astype(np.float32);return GridStepResult(converged=converged,vm_pu=vm_pu,line_loading_pct=line_loading_pct,trafo_loading_pct=trafo_loading_pct,v_violation=v_violation,trafo_p_signed_kw=trafo_p_signed_kw.astype(np.float32,copy=False),psi_v_raw=float(np.sum(bus_v_excess**2)),psi_line_raw=float(np.sum(line_excess**2)),psi_trafo_raw=float(np.sum(trafo_excess**2)))
	def _make_zero_result(self)->GridStepResult:trafo_zero=np.zeros(self.n_trafos,dtype=np.float32);return GridStepResult(converged=False,vm_pu=np.ones(self.n_buses,dtype=np.float32),line_loading_pct=np.zeros(self.n_lines,dtype=np.float32),trafo_loading_pct=trafo_zero.copy(),v_violation=np.zeros(self.n_agents,dtype=np.float32),trafo_p_signed_kw=trafo_zero,psi_v_raw=.0,psi_line_raw=.0,psi_trafo_raw=.0)
