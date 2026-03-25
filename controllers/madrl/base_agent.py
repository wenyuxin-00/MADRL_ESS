"""Base utilities shared by MADRL agents."""

from __future__ import annotations

import importlib.util
import os
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from scripts.utils.nested import add_batch_dim, to_torch_nested

_EMITTED_RUNTIME_WARNINGS: set[str] = set()


class BaseAgent(ABC):
    """Common actor-critic helpers shared by MADDPG and MATD3."""

    def _warn_runtime_once(self, message: str) -> None:
        if message in _EMITTED_RUNTIME_WARNINGS:
            return
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _EMITTED_RUNTIME_WARNINGS.add(message)

    def _runtime_flag(self, name: str, default):
        runtime_cfg = getattr(self.cfg, "runtime", None)
        if runtime_cfg is None or not hasattr(runtime_cfg, name):
            return default
        value = getattr(runtime_cfg, name)
        return default if value is None else value

    def _configure_runtime_acceleration(self) -> None:
        self._actor_forward = self.actor
        self._critic_forward = self.critic
        self._actor_target_forward = self.actor_target
        self._critic_target_forward = self.critic_target
        self._autocast_dtype = None
        self.performance_summary = {
            "amp_enabled": False,
            "amp_dtype": None,
            "amp_fallback_reason": None,
            "compile_enabled": False,
            "compile_mode": None,
            "compile_fallback_reason": None,
            "compiled_modules": [],
        }

        if self.device.type != "cuda":
            self.performance_summary["amp_fallback_reason"] = "CUDA device not in use."
            self.performance_summary["compile_fallback_reason"] = "CUDA device not in use."
            return

        if str(self._runtime_flag("execution_mode", "performance")) != "performance":
            self.performance_summary["amp_fallback_reason"] = "Runtime mode is not performance."
            self.performance_summary["compile_fallback_reason"] = "Runtime mode is not performance."
            return

        if bool(self._runtime_flag("enable_amp", True)):
            amp_dtype_name = str(self._runtime_flag("amp_dtype", "bfloat16")).lower()
            try:
                dtype = getattr(torch, amp_dtype_name)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    pass
                self._autocast_dtype = dtype
                self.performance_summary["amp_enabled"] = True
                self.performance_summary["amp_dtype"] = amp_dtype_name
            except Exception as exc:
                self.performance_summary["amp_fallback_reason"] = str(exc)
                self._warn_runtime_once(
                    f"{type(self).__name__} agent {self.agent_id} disabled CUDA autocast "
                    f"and fell back to eager FP32: {exc}"
                )
        else:
            self.performance_summary["amp_fallback_reason"] = "CUDA autocast disabled by runtime config."

        if not bool(self._runtime_flag("enable_compile", True)):
            self.performance_summary["compile_fallback_reason"] = "torch.compile disabled by runtime config."
            return

        if not hasattr(torch, "compile"):
            self.performance_summary["compile_fallback_reason"] = "torch.compile is unavailable in this PyTorch build."
            return
        if importlib.util.find_spec("triton") is None:
            self.performance_summary["compile_fallback_reason"] = "Triton is unavailable in the current environment."
            return

        compile_kwargs = {
            "mode": str(self._runtime_flag("compile_mode", "reduce-overhead")),
            "fullgraph": bool(self._runtime_flag("compile_fullgraph", False)),
            "dynamic": bool(self._runtime_flag("compile_dynamic", False)),
        }
        try:
            self._actor_forward = torch.compile(self.actor, **compile_kwargs)
            self._critic_forward = torch.compile(self.critic, **compile_kwargs)
            self._actor_target_forward = torch.compile(self.actor_target, **compile_kwargs)
            self._critic_target_forward = torch.compile(self.critic_target, **compile_kwargs)
            self.performance_summary["compile_enabled"] = True
            self.performance_summary["compile_mode"] = compile_kwargs["mode"]
            self.performance_summary["compiled_modules"] = [
                "actor",
                "critic",
                "actor_target",
                "critic_target",
            ]
        except Exception as exc:
            self._actor_forward = self.actor
            self._critic_forward = self.critic
            self._actor_target_forward = self.actor_target
            self._critic_target_forward = self.critic_target
            self.performance_summary["compile_fallback_reason"] = str(exc)
            self._warn_runtime_once(
                f"{type(self).__name__} agent {self.agent_id} disabled torch.compile "
                f"and fell back to eager execution: {exc}"
            )

    def _autocast_context(self):
        if self._autocast_dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    def _actor_call(self, obs_t: dict) -> torch.Tensor:
        with self._autocast_context():
            return self._actor_forward(obs_t)

    def _actor_target_call(self, obs_t: dict) -> torch.Tensor:
        with self._autocast_context():
            return self._actor_target_forward(obs_t)

    def _critic_call(self, obs_t: dict, action_t: torch.Tensor):
        with self._autocast_context():
            return self._critic_forward(obs_t, action_t)

    def _critic_target_call(self, obs_t: dict, action_t: torch.Tensor):
        with self._autocast_context():
            return self._critic_target_forward(obs_t, action_t)

    def _prepare_obs(self, obs: dict) -> tuple[dict, bool]:
        has_batch_dim = obs["local"].ndim == 3
        if not has_batch_dim:
            obs = add_batch_dim(obs)
        return to_torch_nested(obs, self.device), has_batch_dim

    def choose_action(self, obs: dict, noise_std: float) -> np.ndarray:
        obs_t, has_batch_dim = self._prepare_obs(obs)
        with torch.inference_mode():
            action = (
                self.act_from_torch_obs(obs_t, noise_std=noise_std)
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
        if not has_batch_dim:
            action = action[0]
        return action.astype(np.float32)

    def _soft_update(self) -> None:
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

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

    @abstractmethod
    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        """Select actions from already-device-placed torch observations."""

    @abstractmethod
    def train(self, replay_buffer: Any, agent_n: list) -> None:
        """Sample from a replay buffer and execute one update step."""

    @abstractmethod
    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        """Update the agent parameters from one sampled batch."""
