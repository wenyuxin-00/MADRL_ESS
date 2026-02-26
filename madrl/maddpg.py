import torch
import torch.nn.functional as F
import numpy as np
import copy
from common.networks import Actor, Critic_MADDPG
import os


class MADDPG(object):
    def __init__(self, args, agent_id):
        self.device = args.device
        self.N = args.N # 获取智能体的总数量
        self.agent_id = agent_id # 当前智能体的 ID
        self.max_action = args.max_action # 动作的最大值
        self.action_dim = args.action_dim_n[agent_id] # 当前智能体的动作维度
        self.lr_a = args.lr_a # Actor 网络的学习率
        self.lr_c = args.lr_c # Critic 网络的学习率
        self.gamma = args.gamma # 折扣因子
        self.tau = args.tau # 目标网络软更新的系数
        self.use_grad_clip = args.use_grad_clip  # 是否使用梯度裁剪
        
        # --- MODIFIED: Move networks to the specified device ---
        self.actor = Actor(args, agent_id).to(self.device)
        self.critic = Critic_MADDPG(args).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, obs, noise_std):
        # --- MODIFIED: Move observation to device and action back to cpu ---
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(obs).cpu().data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_n):
        # 从经验回放缓冲区中采样一批数据
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # --- Move sampled tensors to device ---
        batch_obs_n = [obs.to(self.device, non_blocking=True) for obs in batch_obs_n]
        batch_a_n = [a.to(self.device, non_blocking=True) for a in batch_a_n]
        batch_r_n = [r.to(self.device, non_blocking=True) for r in batch_r_n]
        batch_obs_next_n = [obs_next.to(self.device, non_blocking=True) for obs_next in batch_obs_next_n]
        batch_done_n = [done.to(self.device, non_blocking=True) for done in batch_done_n]
        
        # --- 更新 Critic 网络 ---
        with torch.no_grad():
            batch_a_next_n = [agent.actor_target(batch_obs_next) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
            Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next

        current_Q = self.critic(batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(target_Q, current_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # --- 更新 Actor 网络 ---
        # Create a mutable copy for modification
        batch_a_n_new = batch_a_n[:]
        batch_a_n_new[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
        actor_loss = -self.critic(batch_obs_n, batch_a_n_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # --- 软更新目标网络 ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, model_dir, episode):
        """
        保存 Actor 和 Critic 网络的参数。
        """
        # 确保目录存在，如果不存在则创建
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # 定义文件路径并保存网络
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
    def load_model(self, model_dir, episode):
        """
        加载 Actor 和 Critic 网络的参数。
        """
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")

        # 加载 state_dict 到对应的网络
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

        # 加载模型后，必须同步更新目标网络，以确保评估和后续训练的稳定性
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        print(f"Agent {self.agent_id}: Models loaded successfully from episode {episode}.")




