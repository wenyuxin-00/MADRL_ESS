"""High-level multi-agent controller wrapper for MADRL policies."""

from __future__ import annotations

import numpy as np
import torch

from controllers.action_feasibility import (
    action_info_to_numpy,
    compute_action_gap_metrics_torch,
    enforce_local_action_feasibility_torch,
)
from controllers.base import BaseController
from scripts.utils.nested import add_batch_dim, to_torch_nested


def _override_soc_penalty_metrics(
    action_info: dict[str, torch.Tensor] | None,
    penalty_source_info: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if action_info is None:
        return penalty_source_info
    if penalty_source_info is None:
        return action_info
    merged = dict(action_info)
    for key in ("soc_penalty_unweighted", "action_penalty_unweighted"):
        if key in penalty_source_info:
            merged[key] = penalty_source_info[key]
    return merged


class MADRLController(BaseController):
    """Coordinate one action per agent for evaluation or deployment."""

    def __init__(self, agent_n: list, noise_std: float = 0.0, projector=None) -> None:
        self.agent_n = list(agent_n)
        self.noise_std = float(noise_std)
        default_projector = getattr(self.agent_n[0], "safety_projector", None) if self.agent_n else None
        self.projector = projector if projector is not None else default_projector
        self.device = getattr(self.agent_n[0], "device", torch.device("cpu")) if self.agent_n else torch.device("cpu")
        self.apply_action_penalty = True
        self.last_action_info: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        """Feed-forward policies do not keep episode state."""
        self.last_action_info = None

    @staticmethod
    def _format_action(action: object) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 0:
            return action_array.reshape(1)
        return action_array

    def _postprocess_joint_actions(
        self,
        obs: dict,
        actions: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray] | None]:
        if not actions:
            return actions, None

        has_batch_dim = np.asarray(obs["local"]).ndim == 3
        obs_batch = obs if has_batch_dim else add_batch_dim(obs)
        if has_batch_dim:
            action_batch = np.stack(actions, axis=1).astype(np.float32)
        else:
            action_batch = np.stack(actions, axis=0).astype(np.float32)[None, ...]

        obs_t = to_torch_nested(obs_batch, self.device)
        action_t = torch.as_tensor(action_batch, device=self.device, dtype=torch.float32)
        if "safety_local" not in obs_t:
            if has_batch_dim:
                return [action_batch[:, agent_id].copy() for agent_id in range(action_batch.shape[1])], None
            return [action_batch[0, agent_id].copy() for agent_id in range(action_batch.shape[1])], None

        with torch.inference_mode():
            if self.projector is not None:
                projected_t = self.projector.project_actions_from_safety_local(
                    obs_t["safety_local"],
                    action_t,
                )
                executed_t, projector_residual_info = enforce_local_action_feasibility_torch(
                    obs_t["safety_local"],
                    projected_t,
                    efficiency=float(getattr(self.agent_n[0].cfg.env, "efficiency", 1.0)),
                    dt_hours=float(getattr(self.agent_n[0].cfg.env, "dt", 1.0)),
                    soc_min=float(getattr(self.agent_n[0].cfg.env, "soc_min", 0.0)),
                    soc_max=float(getattr(self.agent_n[0].cfg.env, "soc_max", 1.0)),
                )
            else:
                executed_t, _ = enforce_local_action_feasibility_torch(
                    obs_t["safety_local"],
                    action_t,
                    efficiency=float(getattr(self.agent_n[0].cfg.env, "efficiency", 1.0)),
                    dt_hours=float(getattr(self.agent_n[0].cfg.env, "dt", 1.0)),
                    soc_min=float(getattr(self.agent_n[0].cfg.env, "soc_min", 0.0)),
                    soc_max=float(getattr(self.agent_n[0].cfg.env, "soc_max", 1.0)),
                )
                projector_residual_info = None
            action_info = compute_action_gap_metrics_torch(obs_t["safety_local"], action_t, executed_t)
            action_info = _override_soc_penalty_metrics(action_info, projector_residual_info)
        executed_np = executed_t.to(dtype=torch.float32).cpu().numpy()
        action_info_np = action_info_to_numpy(action_info)

        if has_batch_dim:
            return [executed_np[:, agent_id].copy() for agent_id in range(executed_np.shape[1])], action_info_np
        single_env_info = None
        if action_info_np is not None:
            single_env_info = {
                key: np.asarray(value[0], dtype=np.float32)
                for key, value in action_info_np.items()
            }
        return [executed_np[0, agent_id].copy() for agent_id in range(executed_np.shape[1])], single_env_info

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        noise_std = 0.0 if deterministic else self.noise_std
        raw_actions = [
            self._format_action(agent.choose_action(obs, noise_std=noise_std))
            for agent in self.agent_n
        ]
        actions, action_info = self._postprocess_joint_actions(obs, raw_actions)
        self.last_action_info = action_info
        return actions
