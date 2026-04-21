from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from controllers.madrl.base_agent import BaseAgent
from models.assembly import build_actor_network, build_critic_network
from scripts.utils.replay_buffer import to_torch_batch
class MATD3(BaseAgent):

    def __init__(self, cfg: object, agent_id: int) -> None:
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
        self.policy_noise = float(cfg.algo.policy_noise)
        self.noise_clip = float(cfg.algo.noise_clip)
        self.policy_update_freq = int(cfg.algo.policy_update_freq)
        self.actor_pointer = 0
        self.actor = build_actor_network(cfg, self.agent_id).to(self.device)
        self.critic = build_critic_network(cfg).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self._configure_runtime_acceleration()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))

    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        action = self._actor_call(obs_t)
        if noise_std > 0.0:
            action = action + torch.randn_like(action) * float(noise_std)
        return action.clamp(-self.max_action, self.max_action)

    def train(self, replay_buffer: object, agent_n: list) -> None:
        batch = to_torch_batch(replay_buffer.sample(), self.device)
        self.train_on_batch(batch, agent_n)

    def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
        self.actor_pointer += 1
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        with torch.no_grad():
            clean_next_action = None if shared_ctx is None else shared_ctx.get("target_actor_actions_clean")
            if clean_next_action is None:
                clean_next_action = torch.stack([agent._actor_target_call(next_obs) for agent in agent_n], dim=1)
            next_action_list = []
            for agent_id, _agent in enumerate(agent_n):
                next_action = clean_next_action[:, agent_id]
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                    -self.noise_clip,
                    self.noise_clip,
                )
                next_action_list.append((next_action + noise).clamp(-self.max_action, self.max_action))
            next_action = torch.stack(next_action_list, dim=1)
            q1_next, q2_next = self._critic_target_call(next_obs, next_action)
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

        new_action = action.clone()
        new_action[:, self.agent_id] = self._actor_call(obs)
        q1_policy, _ = self._critic_call(obs, new_action)
        actor_loss = -q1_policy.float().mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        self._soft_update()
