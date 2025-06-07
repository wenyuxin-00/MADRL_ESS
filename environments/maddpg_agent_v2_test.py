import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
import copy
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, action_dim, batch_size):
        self.capacity = capacity
        self.obs_cap = np.empty((capacity, obs_dim))
        self.next_obs_cap = np.empty((capacity, obs_dim))
        self.state_cap = np.empty((capacity, state_dim))
        self.next_state_cap = np.empty((capacity, state_dim))
        self.action_cap = np.empty((capacity, action_dim))
        self.reward_cap = np.empty((capacity, 1))
        self.done_cap = np.empty((capacity, 1), dtype=bool)
        self.batch_size = batch_size
        self.current = 0 

    def add_memo(self, obs, next_obs, state, next_state, action, reward, done):
        self.obs_cap[self.current] = obs
        self.next_obs_cap[self.current] = next_obs
        self.state_cap[self.current] = state
        self.next_state_cap[self.current] = next_state
        self.action_cap[self.current] = action
        self.reward_cap[self.current] = reward
        self.done_cap[self.current] = done
        self.current = (self.current + 1) % self.capacity 

    def sample(self,idxes):
        obs = self.obs_cap[idxes]
        next_obs = self.next_obs_cap[idxes]
        state = self.state_cap[idxes]
        next_state = self.next_state_cap[idxes]
        action = self.action_cap[idxes]
        reward = self.reward_cap[idxes]
        done = self.done_cap[idxes]

        return obs, next_obs, state, next_state, action, reward, done
    
class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_agent, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims + n_agent * action_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))      




class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # print("fc1输出范围:", x.min(), x.max())  # 检查是否合理
        x = F.relu(self.fc2(x))
        mu = torch.sigmoid(self.pi(x)) #softmax
        # mu = torch.sigmoid(self.pi(x)) + 1e-6 * torch.randn_like(self.pi(x))
        # mu = torch.softmax(self.pi(x), dim=1)
        return mu
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))        


class Agent:
    def __init__(self, memo_size, obs_dim, state_dim, n_agent, action_dim, alpha, beta, fc1_dims, fc2_dims, gamma, tau, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, action_dim=action_dim).to(device)
        self.critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)

        self.target_actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, action_dim=action_dim).to(device)
        self.target_critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)
        
        self.replay_buffer = ReplayBuffer(capacity=memo_size, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim, batch_size=batch_size)
    
    def get_action(self, obs):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2
        single_action = torch.clamp(input=single_action + noise, min=0.0, max=1.0)

        return single_action.detach().cpu().numpy()[0]
    # def get_action(self, obs):
    #     with torch.no_grad():
    #         # 1. Min-Max标准化（按特征维度分别处理）
    #         obs = np.asarray(obs, dtype=np.float32)
            
    #         # 定义各维度理论范围（根据您的场景调整）
    #         feat_ranges = np.array([
    #             [0, 96],     # 时段（假设96个时间步） 
    #             [0, 16.5],  # 电价（根据历史数据调整）
                
    #             [0.0, 1.7],   # 负荷KW
    #             [0.0, 1.0],  # SOC（电池状态）
    #         ])
            
    #         # Min-Max标准化公式：(x - min) / (max - min)
    #         obs_normalized = (obs - feat_ranges[:, 0]) / (feat_ranges[:, 1] - feat_ranges[:, 0] + 1e-8)
            
    #         # 2. 硬截断保证数值安全
    #         obs_normalized = np.clip(obs_normalized, -1.0, 2.0)  # 允许略微超出[0,1]范围
            
    #         # 3. 转换为张量
    #         obs_tensor = torch.as_tensor(obs_normalized, dtype=torch.float32, device=device)
            
    #         # 4. NaN防护（双重检查）
    #         if torch.isnan(obs_tensor).any():
    #             print(f"警告：标准化后观测值含NaN，原始值: {obs}")
    #             obs_tensor = torch.zeros_like(obs_tensor)
            
    #         # 5. 动作生成与保护
    #         action = self.actor(obs_tensor.unsqueeze(0))
    #         if torch.isnan(action).any():
    #             print(f"NaN动作！标准化后obs: {obs_normalized}")
    #             action = torch.rand_like(action) * 0.6 + 0.2  # 生成0.2~0.8的安全动作
            
    #         # 6. 添加受限噪声
    #         noise = torch.randn_like(action) * 0.08  # 减小噪声幅度
    #         noise = torch.clamp(noise, -0.15, 0.15) # 硬截断噪声
            
    #         # 7. 最终动作裁剪（避开边界）
    #         return (action + noise).clamp(0.1, 0.9).squeeze(0).cpu().numpy()

    def save_mode(self, filename):
        self.actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)
        self.target_actor.save_checkpoint(filename)
        self.target_critic.save_checkpoint(filename)

    def load_mode(self, filename):
        self.actor.load_checkpoint(filename)
        self.critic.load_checkpoint(filename)
        self.target_actor.load_checkpoint(filename)
        self.target_critic.load_checkpoint(filename)


