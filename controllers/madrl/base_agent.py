from __future__ import annotations

import copy
import math
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from controllers.madrl.safety_projector import (
	JointGridSafetyProjector,
	enforce_local_action_feasibility_torch,
	map_actor_output_to_soc_feasible_action_torch,
)
from models.assembly import build_actor_network, build_critic_network
from scripts.utils.replay_buffer import to_torch_batch
from scripts.utils.torch_runtime import add_batch_dim, to_torch_nested

_NEAR_ZERO_GRAD_NORM = 1e-10
_HEALTH_WINDOW_SIZE = 200
_GRAD_COLLAPSE_RATIO = 0.80
_SOC_WIDTH_COLLAPSE_RATIO = 0.05


class BaseAgent(ABC):
	def _init_shared_modules(self, cfg: object, agent_id: int) -> None:
		self.cfg = cfg
		self.device = cfg.runtime.device
		self.num_agents = int(cfg.env.num_agents)
		self.agent_id = int(agent_id)
		self.max_action = float(cfg.model.max_action)
		self.action_dim = int(cfg.runtime.action_dim)
		self.gamma = float(cfg.algo.gamma)
		self.tau = float(cfg.algo.tau)
		self.use_grad_clip = bool(cfg.model.use_grad_clip)
		self.grad_clip_norm = float(cfg.model.grad_clip_norm)
		self.actor = build_actor_network(cfg, self.agent_id).to(self.device)
		self.critic = build_critic_network(cfg).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.critic_target = copy.deepcopy(self.critic)
		self._configure_runtime_acceleration()
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr))
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))
		self._near_zero_grad_windows: dict[str, list[bool]] = {"actor": [], "critic": []}
		self._last_actor_feasible_width_ratio_mean: float | None = None
		self.training_health: dict[str, Any] = {
			"agent_id": self.agent_id,
			"nonfinite_failure_count": 0,
			"actor_gradient_collapse_flag": False,
			"actor_gradient_collapse_by_soc_boundary": False,
			"actor_gradient_collapse_without_soc_compression": False,
		}

	def _configure_runtime_acceleration(self) -> None:
		self._actor_forward = self.actor
		self._critic_forward = self.critic
		self._actor_target_forward = self.actor_target
		self._critic_target_forward = self.critic_target
		self.performance_summary = {
			"amp_enabled": False,
			"amp_dtype": None,
			"amp_fallback_reason": "Single-path runtime disables AMP.",
			"compile_enabled": False,
			"compile_mode": None,
			"compile_fallback_reason": "Single-path runtime disables torch.compile.",
			"compiled_modules": [],
		}

	def _actor_call(self, obs_t: dict) -> torch.Tensor:
		return self._actor_forward(obs_t)

	def _actor_target_call(self, obs_t: dict) -> torch.Tensor:
		return self._actor_target_forward(obs_t)

	def _critic_call(self, obs_t: dict, action_t: torch.Tensor):
		return self._critic_forward(obs_t, action_t)

	def _critic_target_call(self, obs_t: dict, action_t: torch.Tensor):
		return self._critic_target_forward(obs_t, action_t)

	def _prepare_obs(self, obs: dict) -> tuple[dict, bool]:
		has_batch_dim = obs["local"].ndim == 3
		if not has_batch_dim:
			obs = add_batch_dim(obs)
		return to_torch_nested(obs, self.device), has_batch_dim

	def choose_action(self, obs: dict, noise_std: float) -> np.ndarray:
		obs_t, has_batch_dim = self._prepare_obs(obs)
		with torch.inference_mode():
			action = self.act_from_torch_obs(obs_t, noise_std=noise_std).to(dtype=torch.float32).cpu().numpy()
		if not has_batch_dim:
			action = action[0]
		return action.astype(np.float32)

	def _soft_update(self) -> None:
		for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
			tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
		for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
			tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

	def _feasibility_kwargs(self) -> dict[str, float]:
		return {
			"efficiency": float(self.cfg.env.efficiency),
			"dt_hours": float(self.cfg.env.dt),
			"soc_min": float(self.cfg.env.soc_min),
			"soc_max": float(self.cfg.env.soc_max),
		}

	def _map_and_guard_joint_actions(self, obs: dict, raw_actions: torch.Tensor) -> torch.Tensor:
		if "safety_local" not in obs:
			return raw_actions.clamp(-self.max_action, self.max_action)
		mapped_actions, _ = map_actor_output_to_soc_feasible_action_torch(
			obs["safety_local"],
			raw_actions,
			**self._feasibility_kwargs(),
		)
		guarded_actions, _ = enforce_local_action_feasibility_torch(
			obs["safety_local"],
			mapped_actions,
			**self._feasibility_kwargs(),
		)
		return guarded_actions

	def _map_and_guard_agent_action(self, obs: dict, raw_action: torch.Tensor) -> torch.Tensor:
		if "safety_local" not in obs:
			return raw_action.clamp(-self.max_action, self.max_action)
		safety_local = obs["safety_local"][:, self.agent_id : self.agent_id + 1]
		mapped_action, mapping_info = map_actor_output_to_soc_feasible_action_torch(
			safety_local,
			raw_action.unsqueeze(1),
			**self._feasibility_kwargs(),
		)
		width_ratio = mapping_info.get("actor_feasible_width_ratio")
		if width_ratio is not None:
			self._last_actor_feasible_width_ratio_mean = float(width_ratio.detach().mean().item())
		guarded_action, _ = enforce_local_action_feasibility_torch(
			safety_local,
			mapped_action,
			**self._feasibility_kwargs(),
		)
		return guarded_action[:, 0]

	def _bootstrap_discount_for_agent(self, batch: dict, terminated: torch.Tensor) -> torch.Tensor:
		bootstrap_discount = batch.get("bootstrap_discount")
		if bootstrap_discount is None:
			return self.gamma * (1 - terminated[:, self.agent_id])
		return bootstrap_discount[:, self.agent_id]

	def _actor_update_allowed(self, shared_ctx: dict | None) -> bool:
		return shared_ctx is None or bool(shared_ctx.get("allow_actor_update", True))

	def _record_actor_update_skipped(self) -> None:
		key = "actor_update_skipped_by_learning_starts"
		self.training_health[key] = int(self.training_health.get(key, 0)) + 1

	def _mark_nonfinite_failure(self, role: str, detail: str) -> None:
		self.training_health["nonfinite_failure_count"] = int(self.training_health.get("nonfinite_failure_count", 0)) + 1
		self.training_health[f"{role}_last_nonfinite_detail"] = detail

	def _require_finite_tensor(self, tensor: torch.Tensor, *, role: str, name: str, backward_agent_id: int | None = None) -> None:
		if torch.isfinite(tensor.detach()).all():
			return
		owner = f"agent {self.agent_id}"
		backward = "" if backward_agent_id is None else f" during backward triggered by agent {backward_agent_id}"
		detail = f"{owner} produced non-finite {role} {name}{backward}."
		self._mark_nonfinite_failure(role, detail)
		raise FloatingPointError(detail)

	def _check_module_parameters_finite(self, module: torch.nn.Module, *, role: str) -> None:
		for name, param in module.named_parameters():
			if not torch.isfinite(param.detach()).all():
				detail = f"agent {self.agent_id} {role} parameter '{name}' became non-finite after optimizer step."
				self._mark_nonfinite_failure(role, detail)
				raise FloatingPointError(detail)

	def _gradient_stats(self, params: list[torch.nn.Parameter], *, role: str, backward_agent_id: int | None = None) -> dict[str, float | int]:
		total_sq = 0.0
		max_abs = 0.0
		param_with_grad_count = 0
		for param_index, param in enumerate(params):
			grad = param.grad
			if grad is None:
				continue
			param_with_grad_count += 1
			if not torch.isfinite(grad.detach()).all():
				backward = "" if backward_agent_id is None else f" during backward triggered by agent {backward_agent_id}"
				detail = f"agent {self.agent_id} produced non-finite {role} gradient in parameter #{param_index}{backward}."
				self._mark_nonfinite_failure(role, detail)
				raise FloatingPointError(detail)
			grad_fp32 = grad.detach().float()
			total_sq += float(torch.sum(grad_fp32 * grad_fp32).item())
			max_abs = max(max_abs, float(grad_fp32.abs().max().item()))
		return {
			"grad_norm": math.sqrt(max(total_sq, 0.0)),
			"grad_max_abs": max_abs,
			"param_with_grad_count": param_with_grad_count,
		}

	def _record_train_health(self, role: str, *, loss_value: float, grad_stats: dict[str, float | int]) -> None:
		step_key = f"{role}_optimizer_step_count"
		step_count = int(self.training_health.get(step_key, 0)) + 1
		self.training_health[step_key] = step_count
		self.training_health[f"{role}_loss_last"] = float(loss_value)
		self.training_health[f"{role}_grad_norm_last"] = float(grad_stats["grad_norm"])
		self.training_health[f"{role}_grad_max_abs_last"] = float(grad_stats["grad_max_abs"])
		self.training_health[f"{role}_param_with_grad_count_last"] = int(grad_stats["param_with_grad_count"])
		loss_sum_key = f"{role}_loss_sum"
		grad_sum_key = f"{role}_grad_norm_sum"
		self.training_health[loss_sum_key] = float(self.training_health.get(loss_sum_key, 0.0)) + float(loss_value)
		self.training_health[grad_sum_key] = float(self.training_health.get(grad_sum_key, 0.0)) + float(grad_stats["grad_norm"])
		self.training_health[f"{role}_loss_mean"] = self.training_health[loss_sum_key] / step_count
		self.training_health[f"{role}_grad_norm_mean"] = self.training_health[grad_sum_key] / step_count
		near_zero = float(grad_stats["grad_norm"]) <= _NEAR_ZERO_GRAD_NORM
		window = self._near_zero_grad_windows.setdefault(role, [])
		window.append(bool(near_zero))
		if len(window) > _HEALTH_WINDOW_SIZE:
			del window[0 : len(window) - _HEALTH_WINDOW_SIZE]
		near_zero_count = int(self.training_health.get(f"{role}_near_zero_grad_count", 0)) + int(near_zero)
		self.training_health[f"{role}_near_zero_grad_count"] = near_zero_count
		window_ratio = float(sum(window)) / max(len(window), 1)
		self.training_health[f"{role}_near_zero_grad_ratio_last_window"] = window_ratio
		if role == "actor":
			width_mean = self._last_actor_feasible_width_ratio_mean
			if width_mean is not None:
				width_sum_key = "actor_feasible_width_ratio_sum"
				self.training_health[width_sum_key] = float(self.training_health.get(width_sum_key, 0.0)) + width_mean
				self.training_health["actor_feasible_width_ratio_mean"] = self.training_health[width_sum_key] / step_count
				self.training_health["actor_feasible_width_ratio_last"] = width_mean
			if window_ratio > _GRAD_COLLAPSE_RATIO:
				self.training_health["actor_gradient_collapse_flag"] = True
				if width_mean is not None and width_mean <= _SOC_WIDTH_COLLAPSE_RATIO:
					self.training_health["actor_gradient_collapse_by_soc_boundary"] = True
				else:
					self.training_health["actor_gradient_collapse_without_soc_compression"] = True

	def _optimizer_step(
		self,
		optimizer: torch.optim.Optimizer,
		loss: torch.Tensor,
		params,
		*,
		role: str,
		module: torch.nn.Module,
		backward_agent_id: int | None = None,
		retain_graph: bool = False,
	) -> None:
		params_list = list(params)
		self._require_finite_tensor(loss, role=role, name="loss", backward_agent_id=backward_agent_id)
		optimizer.zero_grad(set_to_none=True)
		loss.backward(retain_graph=retain_graph)
		grad_stats = self._gradient_stats(params_list, role=role, backward_agent_id=backward_agent_id)
		if self.use_grad_clip:
			clip_norm = torch.nn.utils.clip_grad_norm_(params_list, self.grad_clip_norm)
			self._require_finite_tensor(torch.as_tensor(clip_norm, device=self.device), role=role, name="clip_grad_norm", backward_agent_id=backward_agent_id)
		optimizer.step()
		self._check_module_parameters_finite(module, role=role)
		self._record_train_health(role, loss_value=float(loss.detach().float().item()), grad_stats=grad_stats)

	def _step_optimizer_from_existing_grad(
		self,
		optimizer: torch.optim.Optimizer,
		loss_value: float,
		params,
		*,
		role: str,
		module: torch.nn.Module,
		backward_agent_id: int | None = None,
	) -> None:
		params_list = list(params)
		grad_stats = self._gradient_stats(params_list, role=role, backward_agent_id=backward_agent_id)
		if self.use_grad_clip:
			clip_norm = torch.nn.utils.clip_grad_norm_(params_list, self.grad_clip_norm)
			self._require_finite_tensor(torch.as_tensor(clip_norm, device=self.device), role=role, name="clip_grad_norm", backward_agent_id=backward_agent_id)
		optimizer.step()
		self._check_module_parameters_finite(module, role=role)
		self._record_train_health(role, loss_value=float(loss_value), grad_stats=grad_stats)

	def save_model(self, model_dir: str, episode: int) -> None:
		os.makedirs(model_dir, exist_ok=True)
		actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
		critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
		torch.save(self.actor.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)

	def load_model(self, model_dir: str, episode: int) -> None:
		actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
		critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
		self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
		self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic_target.load_state_dict(self.critic.state_dict())

	def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
		action = self._actor_call(obs_t)
		if noise_std > 0.0:
			action = action + torch.randn_like(action) * float(noise_std)
		return action.clamp(-self.max_action, self.max_action)

	def train(self, replay_buffer: Any, agent_n: list) -> None:
		self.train_on_batch(to_torch_batch(replay_buffer.sample(), self.device), agent_n)

	@abstractmethod
	def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
		raise NotImplementedError


class MADDPG(BaseAgent):
	def __init__(self, cfg: object, agent_id: int) -> None:
		self._init_shared_modules(cfg, agent_id)

	def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
		obs = batch["obs"]
		action = batch["action"]
		reward = batch["reward"]
		next_obs = batch["next_obs"]
		terminated = batch["terminated"]
		bootstrap_discount = self._bootstrap_discount_for_agent(batch, terminated)
		with torch.no_grad():
			next_action_raw = None if shared_ctx is None else shared_ctx.get("target_actor_actions_raw")
			if next_action_raw is None:
				next_action_raw = torch.stack([agent._actor_target_call(next_obs) for agent in agent_n], dim=1)
			next_action = self._map_and_guard_joint_actions(next_obs, next_action_raw)
			target_q = reward[:, self.agent_id] + bootstrap_discount * self._critic_target_call(next_obs, next_action)
		current_q = self._critic_call(obs, action)
		critic_loss = F.mse_loss(current_q.float(), target_q.float())
		self._optimizer_step(self.critic_optimizer, critic_loss, self.critic.parameters(), role="critic", module=self.critic)
		if not self._actor_update_allowed(shared_ctx):
			self._record_actor_update_skipped()
			self._soft_update()
			return
		new_action = action.clone()
		new_action[:, self.agent_id] = self._map_and_guard_agent_action(obs, self._actor_call(obs))
		actor_loss = -self._critic_call(obs, new_action).float().mean()
		self._optimizer_step(self.actor_optimizer, actor_loss, self.actor.parameters(), role="actor", module=self.actor)
		self._soft_update()


class MATD3(BaseAgent):
	def __init__(self, cfg: object, agent_id: int) -> None:
		self._init_shared_modules(cfg, agent_id)
		self.policy_noise = float(cfg.algo.policy_noise)
		self.noise_clip = float(cfg.algo.noise_clip)
		self.policy_update_freq = int(cfg.algo.policy_update_freq)
		self.actor_pointer = 0

	def _smoothed_target_actions(self, clean_next_action: torch.Tensor) -> torch.Tensor:
		noise = (torch.randn_like(clean_next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
		return (clean_next_action + noise).clamp(-self.max_action, self.max_action)

	def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
		self.actor_pointer += 1
		obs = batch["obs"]
		action = batch["action"]
		reward = batch["reward"]
		next_obs = batch["next_obs"]
		terminated = batch["terminated"]
		bootstrap_discount = self._bootstrap_discount_for_agent(batch, terminated)
		with torch.no_grad():
			clean_next_action_raw = None if shared_ctx is None else shared_ctx.get("target_actor_actions_raw")
			if clean_next_action_raw is None:
				clean_next_action_raw = torch.stack([agent._actor_target_call(next_obs) for agent in agent_n], dim=1)
			next_action_raw = self._smoothed_target_actions(clean_next_action_raw)
			next_action = self._map_and_guard_joint_actions(next_obs, next_action_raw)
			q1_next, q2_next = self._critic_target_call(next_obs, next_action)
			target_q = reward[:, self.agent_id] + bootstrap_discount * torch.min(q1_next, q2_next)
		current_q1, current_q2 = self._critic_call(obs, action)
		target_q_fp32 = target_q.float()
		critic_loss = F.mse_loss(current_q1.float(), target_q_fp32) + F.mse_loss(current_q2.float(), target_q_fp32)
		self._optimizer_step(self.critic_optimizer, critic_loss, self.critic.parameters(), role="critic", module=self.critic)
		if not self._actor_update_allowed(shared_ctx):
			self._record_actor_update_skipped()
			self._soft_update()
			return
		if self.actor_pointer % self.policy_update_freq != 0:
			return
		new_action = action.clone()
		new_action[:, self.agent_id] = self._map_and_guard_agent_action(obs, self._actor_call(obs))
		q1_policy, _ = self._critic_call(obs, new_action)
		actor_loss = -q1_policy.float().mean()
		self._optimizer_step(self.actor_optimizer, actor_loss, self.actor.parameters(), role="actor", module=self.actor)
		self._soft_update()


class MATD3SafePOC(MATD3):
	def __init__(self, cfg: object, agent_id: int) -> None:
		super().__init__(cfg, agent_id)
		self.safety_projector = JointGridSafetyProjector.from_cfg(cfg, device=self.device)
		self.safety_enabled = True

	def _record_projection_event(self, shared_ctx: dict | None, *, stage: str, batch_size: int, elapsed_s: float) -> None:
		if shared_ctx is None:
			return
		recorder = shared_ctx.get("projection_event_recorder")
		if recorder is not None:
			recorder(stage=stage, batch_size=batch_size, elapsed_s=elapsed_s)

	def _residual_guard(self, safety_local: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		guarded, _ = enforce_local_action_feasibility_torch(
			safety_local,
			actions,
			**self._feasibility_kwargs(),
		)
		return guarded

	def _get_projected_target_actions(self, next_obs: dict, agent_n: list, shared_ctx: dict | None) -> torch.Tensor:
		if shared_ctx is not None and shared_ctx.get("projected_target_actions") is not None:
			return shared_ctx["projected_target_actions"]
		with torch.no_grad():
			clean_next_action_raw = None if shared_ctx is None else shared_ctx.get("target_actor_actions_raw")
			if clean_next_action_raw is None:
				clean_next_action_raw = torch.stack([agent._actor_target_call(next_obs) for agent in agent_n], dim=1)
			next_action_raw = self._smoothed_target_actions(clean_next_action_raw)
			mapped_next_action = self._map_and_guard_joint_actions(next_obs, next_action_raw)
			started = time.perf_counter()
			projected = self.safety_projector.project_actions_from_safety_local(next_obs["safety_local"], mapped_next_action)
			projected = self._residual_guard(next_obs["safety_local"], projected)
		self._record_projection_event(shared_ctx, stage="target", batch_size=int(next_action_raw.shape[0]), elapsed_s=time.perf_counter() - started)
		if shared_ctx is not None:
			shared_ctx["projected_target_actions"] = projected
		return projected

	def _get_projected_policy_actions(self, obs: dict, action: torch.Tensor, agent_n: list, shared_ctx: dict | None) -> torch.Tensor:
		if shared_ctx is not None and shared_ctx.get("projected_policy_actions_all") is not None:
			return shared_ctx["projected_policy_actions_all"]
		batch_size = int(action.shape[0])
		n_agents = int(len(agent_n))
		candidate_joint_actions = action.unsqueeze(0).expand(n_agents, -1, -1, -1).clone()
		for agent_id, agent in enumerate(agent_n):
			candidate_joint_actions[agent_id, :, agent_id] = agent._map_and_guard_agent_action(obs, agent._actor_call(obs))
		flat_candidate_actions = candidate_joint_actions.reshape(n_agents * batch_size, self.num_agents, self.action_dim)
		safety_local = obs["safety_local"].unsqueeze(0).expand(n_agents, -1, -1, -1).reshape(n_agents * batch_size, self.num_agents, obs["safety_local"].shape[-1])
		started = time.perf_counter()
		projected = self.safety_projector.project_actions_from_safety_local(safety_local, flat_candidate_actions)
		projected = self._residual_guard(safety_local, projected)
		self._record_projection_event(shared_ctx, stage="actor", batch_size=int(flat_candidate_actions.shape[0]), elapsed_s=time.perf_counter() - started)
		projected_policy_actions = projected.reshape(n_agents, batch_size, self.num_agents, self.action_dim)
		if shared_ctx is not None:
			shared_ctx["projected_policy_actions_all"] = projected_policy_actions
		return projected_policy_actions

	def _shared_actor_backward_and_step(self, actor_loss: torch.Tensor, agent_n: list, shared_ctx: dict) -> None:
		self._require_finite_tensor(actor_loss, role="actor", name="loss", backward_agent_id=self.agent_id)
		shared_ctx.setdefault("actor_losses", {})[self.agent_id] = float(actor_loss.detach().float().item())
		self.actor_optimizer.zero_grad(set_to_none=True)
		shared_ctx["actor_backward_count"] = int(shared_ctx.get("actor_backward_count", 0)) + 1
		retain_graph = shared_ctx["actor_backward_count"] < len(agent_n)
		actor_loss.backward(retain_graph=retain_graph)
		for agent in agent_n:
			agent._gradient_stats(list(agent.actor.parameters()), role="actor", backward_agent_id=self.agent_id)
		if shared_ctx["actor_backward_count"] < len(agent_n):
			return
		actor_losses = shared_ctx.get("actor_losses", {})
		for agent in agent_n:
			agent._step_optimizer_from_existing_grad(
				agent.actor_optimizer,
				float(actor_losses.get(agent.agent_id, math.nan)),
				agent.actor.parameters(),
				role="actor",
				module=agent.actor,
				backward_agent_id=self.agent_id,
			)
			agent._soft_update()

	def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
		self.actor_pointer += 1
		obs = batch["obs"]
		action = batch["action"]
		reward = batch["reward"]
		next_obs = batch["next_obs"]
		terminated = batch["terminated"]
		bootstrap_discount = self._bootstrap_discount_for_agent(batch, terminated)
		with torch.no_grad():
			projected_next_action = self._get_projected_target_actions(next_obs, agent_n, shared_ctx)
			q1_next, q2_next = self._critic_target_call(next_obs, projected_next_action)
			target_q = reward[:, self.agent_id] + bootstrap_discount * torch.min(q1_next, q2_next)
		current_q1, current_q2 = self._critic_call(obs, action)
		target_q_fp32 = target_q.float()
		critic_loss = F.mse_loss(current_q1.float(), target_q_fp32) + F.mse_loss(current_q2.float(), target_q_fp32)
		self._optimizer_step(self.critic_optimizer, critic_loss, self.critic.parameters(), role="critic", module=self.critic)
		if not self._actor_update_allowed(shared_ctx):
			self._record_actor_update_skipped()
			self._soft_update()
			return
		if self.actor_pointer % self.policy_update_freq != 0:
			return
		projected_policy_actions = self._get_projected_policy_actions(obs, action, agent_n, shared_ctx)
		q1_policy, _ = self._critic_call(obs, projected_policy_actions[self.agent_id])
		actor_loss = -q1_policy.float().mean()
		if shared_ctx is None:
			self._optimizer_step(self.actor_optimizer, actor_loss, self.actor.parameters(), role="actor", module=self.actor)
			self._soft_update()
			return
		self._shared_actor_backward_and_step(actor_loss, agent_n, shared_ctx)


def get_agent_cls(name: str) -> type[BaseAgent]:
	normalized = str(name)
	if normalized == "MADDPG":
		return MADDPG
	if normalized == "MATD3":
		return MATD3
	if normalized == "MATD3_SAFE_POC":
		return MATD3SafePOC
	raise ValueError(f"Unknown algorithm '{normalized}', available: ['MADDPG', 'MATD3', 'MATD3_SAFE_POC']")
