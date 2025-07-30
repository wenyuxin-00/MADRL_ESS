import torch
import torch.nn.functional as F
import numpy as np
import copy
from networks import Actor, Critic_MADDPG
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

        # --- MODIFIED: Move all batch data to the specified device ---
        # Note: The replay buffer returns numpy arrays. Convert them to tensors and move to device.
        batch_obs_n = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in batch_obs_n]
        batch_a_n = [torch.tensor(a, dtype=torch.float32).to(self.device) for a in batch_a_n]
        batch_r_n = [torch.tensor(r, dtype=torch.float32).to(self.device) for r in batch_r_n]
        batch_obs_next_n = [torch.tensor(obs_next, dtype=torch.float32).to(self.device) for obs_next in batch_obs_next_n]
        batch_done_n = [torch.tensor(done, dtype=torch.float32).to(self.device) for done in batch_done_n]
        
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

# #========================传统方法，支持CPU训练========================
# class MADDPG(object):
#     def __init__(self, args, agent_id):
#         self.device = args.device
#         self.N = args.N # 获取智能体的总数量
#         self.agent_id = agent_id # 当前智能体的 ID
#         self.max_action = args.max_action # 动作的最大值
#         self.action_dim = args.action_dim_n[agent_id] # 当前智能体的动作维度
#         self.lr_a = args.lr_a # Actor 网络的学习率
#         self.lr_c = args.lr_c # Critic 网络的学习率
#         self.gamma = args.gamma # 折扣因子
#         self.tau = args.tau # 目标网络软更新的系数
#         self.use_grad_clip = args.use_grad_clip  # 是否使用梯度裁剪
#         # 根据 'agent_id' 为每个智能体创建独立的 actor 和 critic 网络
#         self.actor = Actor(args, agent_id) # 创建当前智能体的 Actor 网络
#         self.critic = Critic_MADDPG(args) # 创建当前智能体的 Critic 网络
#         self.actor_target = copy.deepcopy(self.actor) # 创建 Actor 的目标网络
#         self.critic_target = copy.deepcopy(self.critic) # 创建 Critic 的目标网络

#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a) # Actor 网络的优化器
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)  # Critic 网络的优化器

#     def choose_action(self, obs, noise_std): # 每个智能体根据自己的局部观测选择动作（为探索加入噪声）
#         obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0) # 将观测转换为 Tensor，并增加一个批次维度
#         a = self.actor(obs).data.numpy().flatten() # 通过 Actor 网络得到确定性动作，并转换为 numpy 数组
#         a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action) # 给动作加入高斯噪声以进行探索，并使用 clip 函数将动作限制在有效范围内
#         return a

#     def train(self, replay_buffer, agent_n): # 定义训练函数
#         # 从经验回放缓冲区中采样一批数据
#         batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()
        
#         # --- 更新 Critic 网络 ---
#         # 计算目标 Q 值 (target_Q)
#         with torch.no_grad():  # 在这个代码块中不计算梯度，以提高效率
#             # 根据所有智能体的目标 Actor 网络选择下一时刻的动作
#             batch_a_next_n = [agent.actor_target(batch_obs_next) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
#             # 使用目标 Critic 网络计算下一时刻的 Q 值 (Q_next)
#             Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
#             # 根据贝尔曼方程计算目标 Q 值
#             target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next  # shape:(batch_size,1)

#         # 计算当前 Q 值
#         current_Q = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
#         # 计算 Critic 网络的损失，使用均方误差损失函数
#         critic_loss = F.mse_loss(target_Q, current_Q) 
#         # print(f"Agent {self.agent_id} - Critic Loss: {critic_loss.item()}") # 打印 Critic 损失

#         # 优化 Critic 网络
#         self.critic_optimizer.zero_grad() # 清空过往梯度
#         critic_loss.backward() # 反向传播计算梯度
#         if self.use_grad_clip: # 如果启用梯度裁剪
#             critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0) # 对 Critic 网络的梯度进行裁剪，防止梯度爆炸
#             # print(f"Agent {self.agent_id} - Critic Grad Norm: {critic_grad_norm}")
#         self.critic_optimizer.step() # 更新 Critic 网络的参数

#         # --- 更新 Actor 网络 ---
#         # 重新计算当前智能体的动作，其他智能体的动作保持不变
#         batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
#         # Actor 的损失是 Critic 对新动作组合评估的 Q 值的负均值（目标是最大化 Q 值）
#         actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()
#         # print(f"Agent {self.agent_id} - Actor Loss: {actor_loss.item()}") # 打印 Actor 损失

#         # 优化 Actor 网络
#         self.actor_optimizer.zero_grad() # 清空过往梯度
#         actor_loss.backward() # 反向传播计算梯度
#         if self.use_grad_clip: # 如果启用梯度裁剪
#             actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0) # 对 Actor 网络的梯度进行裁剪
#             # print(f"Agent {self.agent_id} - Actor Grad Norm: {actor_grad_norm}")
#         self.actor_optimizer.step() # 更新 Actor 网络的参数

#         # --- 软更新目标网络 ---
#         # 软更新 Critic 的目标网络
#         for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#         # 软更新 Actor 的目标网络
#         for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)











#================================优先更新重要的样本========================================
    # def train(self, replay_buffer, agent_n): # 优先回放
    #     # 从优先经验回放缓冲区中采样一批数据
    #     tree_indices, (batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n), ISWeights = replay_buffer.sample()

    #     # --- 更新 Critic 网络 ---
    #     # 计算目标 Q 值 (target_Q)
    #     with torch.no_grad():  # 在这个代码块中不计算梯度，以提高效率
    #         # 根据所有智能体的目标 Actor 网络选择下一时刻的动作
    #         batch_a_next_n = [agent.actor_target(batch_obs_next) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
    #         # 使用目标 Critic 网络计算下一时刻的 Q 值 (Q_next)
    #         Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
    #         # 根据贝尔曼方程计算目标 Q 值
    #         target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next  # shape:(batch_size,1)

    #     # 计算当前 Q 值
    #     current_Q = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        
    #     # 计算 TD 误差用于更新优先级
    #     abs_errors = (target_Q - current_Q).abs().detach().numpy()
    #      # 计算带重要性采样的 Critic 损失
    #     critic_loss = (ISWeights * F.mse_loss(target_Q, current_Q, reduction='none')).mean()

    #     # 优化 Critic 网络
    #     self.critic_optimizer.zero_grad() # 清空过往梯度
    #     critic_loss.backward() # 反向传播计算梯度
    #     if self.use_grad_clip: # 如果启用梯度裁剪
    #         critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0) # 对 Critic 网络的梯度进行裁剪，防止梯度爆炸
    #         # print(f"Agent {self.agent_id} - Critic Grad Norm: {critic_grad_norm}")
    #     self.critic_optimizer.step() # 更新 Critic 网络的参数

    #     # --- 更新 Actor 网络 ---
    #     # 重新计算当前智能体的动作，其他智能体的动作保持不变
    #     batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
    #     # Actor 的损失是 Critic 对新动作组合评估的 Q 值的负均值（目标是最大化 Q 值）
    #     actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()
    #     # print(f"Agent {self.agent_id} - Actor Loss: {actor_loss.item()}") # 打印 Actor 损失

    #     # 优化 Actor 网络
    #     self.actor_optimizer.zero_grad() # 清空过往梯度
    #     actor_loss.backward() # 反向传播计算梯度
    #     if self.use_grad_clip: # 如果启用梯度裁剪
    #         actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0) # 对 Actor 网络的梯度进行裁剪
    #         # print(f"Agent {self.agent_id} - Actor Grad Norm: {actor_grad_norm}")
    #     self.actor_optimizer.step() # 更新 Actor 网络的参数
    #     # --- 更新经验优先级 ---
    #     replay_buffer.batch_update(tree_indices, abs_errors)
    #     # --- 软更新目标网络 ---
    #     # 软更新 Critic 的目标网络
    #     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #     # 软更新 Actor 的目标网络
    #     for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # def save_model(self, env_name, algorithm, number, total_steps, agent_id): # 将当前智能体 Actor 网络的 state_dict (包含所有可学习参数) 保存到指定路径
    #     torch.save(self.actor.state_dict(), "./model/{}/{}_actor_number_{}_step_{}k_agent_{}.pth".format(env_name, algorithm, number, int(total_steps / 1000), agent_id))
