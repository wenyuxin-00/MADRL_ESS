"""Isolated MATD3 safe-layer PoC implementation."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from controllers.madrl.matd3 import MATD3
from controllers.madrl.safety_projector import JointGridSafetyProjector


class MATD3SafePOC(MATD3):
    """MATD3 branch that projects joint actions before critic/actor updates."""

    def __init__(self, cfg: object, agent_id: int) -> None:
        super().__init__(cfg, agent_id)
        self.safety_projector = JointGridSafetyProjector.from_cfg(cfg, device=self.device)
        self.safety_enabled = True

    def _record_projection_event(
        self,
        shared_ctx: dict | None,
        *,
        stage: str,
        batch_size: int,
        elapsed_s: float,
    ) -> None:
        if shared_ctx is None:
            return
        recorder = shared_ctx.get("projection_event_recorder")
        if recorder is None:
            return
        recorder(stage=stage, batch_size=batch_size, elapsed_s=elapsed_s)

    def _smoothed_target_actions(
        self,
        clean_next_action: torch.Tensor,
    ) -> torch.Tensor:
        noise = (torch.randn_like(clean_next_action) * self.policy_noise).clamp(
            -self.noise_clip,
            self.noise_clip,
        )
        return (clean_next_action + noise).clamp(-self.max_action, self.max_action)

    def _get_projected_target_actions(
        self,
        next_obs: dict,
        agent_n: list,
        shared_ctx: dict | None,
    ) -> torch.Tensor:
        if shared_ctx is not None:
            cached = shared_ctx.get("projected_target_actions")
            if cached is not None:
                return cached

        with torch.no_grad():
            clean_next_action = None if shared_ctx is None else shared_ctx.get("target_actor_actions_clean")
            if clean_next_action is None:
                clean_next_action = torch.stack([agent._actor_target_call(next_obs) for agent in agent_n], dim=1)
            next_action = self._smoothed_target_actions(clean_next_action)
            projection_started = time.perf_counter()
            projected_next_action = self.safety_projector.project_actions_from_safety_local(
                next_obs["safety_local"],
                next_action,
            )
            projection_elapsed = time.perf_counter() - projection_started

        self._record_projection_event(
            shared_ctx,
            stage="target",
            batch_size=int(next_action.shape[0]),
            elapsed_s=projection_elapsed,
        )
        if shared_ctx is not None:
            shared_ctx["projected_target_actions"] = projected_next_action
        return projected_next_action

    def _get_projected_policy_actions(
        self,
        obs: dict,
        action: torch.Tensor,
        agent_n: list,
        shared_ctx: dict | None,
    ) -> torch.Tensor:
        if shared_ctx is not None:
            cached = shared_ctx.get("projected_policy_actions_all")
            if cached is not None:
                return cached

        batch_size = int(action.shape[0])
        n_agents = int(len(agent_n))
        candidate_joint_actions = action.unsqueeze(0).expand(n_agents, -1, -1, -1).clone()
        for agent_id, agent in enumerate(agent_n):
            candidate_joint_actions[agent_id, :, agent_id] = agent._actor_call(obs)

        flat_candidate_actions = candidate_joint_actions.reshape(
            n_agents * batch_size,
            self.num_agents,
            self.action_dim,
        )
        safety_local = (
            obs["safety_local"]
            .unsqueeze(0)
            .expand(n_agents, -1, -1, -1)
            .reshape(n_agents * batch_size, self.num_agents, obs["safety_local"].shape[-1])
        )
        projection_started = time.perf_counter()
        flat_projected_actions = self.safety_projector.project_actions_from_safety_local(
            safety_local,
            flat_candidate_actions,
        )
        projection_elapsed = time.perf_counter() - projection_started
        self._record_projection_event(
            shared_ctx,
            stage="actor",
            batch_size=int(flat_candidate_actions.shape[0]),
            elapsed_s=projection_elapsed,
        )

        projected_policy_actions = flat_projected_actions.reshape(
            n_agents,
            batch_size,
            self.num_agents,
            self.action_dim,
        )
        if shared_ctx is not None:
            shared_ctx["projected_policy_actions_all"] = projected_policy_actions
        return projected_policy_actions

    def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
        self.actor_pointer += 1

        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            projected_next_action = self._get_projected_target_actions(next_obs, agent_n, shared_ctx)
            q1_next, q2_next = self._critic_target_call(next_obs, projected_next_action)
            target_q = reward[:, self.agent_id] + self.gamma * (1 - done[:, self.agent_id]) * torch.min(
                q1_next,
                q2_next,
            )

        current_q1, current_q2 = self._critic_call(obs, action)
        target_q_fp32 = target_q.float()
        critic_loss = F.mse_loss(current_q1.float(), target_q_fp32) + F.mse_loss(
            current_q2.float(),
            target_q_fp32,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        if self.actor_pointer % self.policy_update_freq != 0:
            return

        projected_policy_actions = self._get_projected_policy_actions(obs, action, agent_n, shared_ctx)
        q1_policy, _ = self._critic_call(obs, projected_policy_actions[self.agent_id])
        actor_loss = -q1_policy.float().mean()

        self.actor_optimizer.zero_grad()
        if shared_ctx is None:
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            self._soft_update()
            return

        shared_ctx["actor_backward_count"] = int(shared_ctx.get("actor_backward_count", 0)) + 1
        retain_graph = self.agent_id < (len(agent_n) - 1)
        actor_loss.backward(retain_graph=retain_graph)
        if shared_ctx["actor_backward_count"] < len(agent_n):
            return

        for agent in agent_n:
            if getattr(agent, "use_grad_clip", False):
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.grad_clip_norm)
            agent.actor_optimizer.step()
            agent._soft_update()


__all__ = ["MATD3SafePOC"]
