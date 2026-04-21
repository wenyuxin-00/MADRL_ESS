from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from controllers.madrl.base_agent import BaseAgent
from models.assembly import build_actor_network, build_critic_network
from scripts.utils.replay_buffer import to_torch_batch
class MADDPG(BaseAgent):

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
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        with torch.no_grad():
            next_action = None if shared_ctx is None else shared_ctx.get("target_actor_actions_clean")
            if next_action is None:
                next_action = torch.stack([agent._actor_target_call(next_obs) for agent in agent_n], dim=1)
            target_q = reward[:, self.agent_id] + self.gamma * (1 - done[:, self.agent_id]) * self._critic_target_call(
                next_obs,
                next_action,
            )

        current_q = self._critic_call(obs, action)
        critic_loss = F.mse_loss(current_q.float(), target_q.float())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        new_action = action.clone()
        new_action[:, self.agent_id] = self._actor_call(obs)
        actor_loss = -self._critic_call(obs, new_action).float().mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        self._soft_update()
