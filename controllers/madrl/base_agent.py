"""MADRL 单智能体基类。

定义单个智能体的 Actor-Critic 网络结构与更新逻辑，
供 MADDPG 和 MATD3 等具体算法继承。

主要类:
    BaseAgent -- 单智能体抽象基类
"""

from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from scripts.utils.nested import add_batch_dim, to_torch_nested


class BaseAgent(ABC):
    """多智能体强化学习智能体的抽象基类。

    每个智能体拥有一个 Actor 网络和一个（或双份）Critic 网络。
    该基类负责：
    - 动作选择（观测预处理 + 前向推理）
    - 目标网络软更新（Polyak 平均）
    - 模型检查点的保存与加载

    子类需要设置的属性:
        agent_id (int): 该智能体在多智能体系统中的索引。
        device (torch.device): 计算设备（CPU 或 CUDA）。
        actor: Actor（策略）网络。
        critic: Critic（价值）网络。
        actor_target: Actor 目标网络。
        critic_target: Critic 目标网络。
        tau (float): 目标网络软更新系数。
    """

    def _prepare_obs(self, obs: dict) -> tuple[dict, bool]:
        """将观测字典预处理为带 batch 维度的 PyTorch 张量。

        参数:
            obs: 结构化观测字典，可能有或没有 batch 维度。

        返回:
            tuple: (转换后的 torch 观测字典, 原始数据是否已有 batch 维度)。
        """
        # 通过 local 观测的维度数判断是否已有 batch 维度（3维=有batch）
        has_batch_dim = obs["local"].ndim == 3
        if not has_batch_dim:
            # 单步观测需要手动添加 batch 维度以统一后续处理
            obs = add_batch_dim(obs)
        # 将嵌套字典中的 numpy 数组递归转换为 torch 张量并移至目标设备
        obs_t = to_torch_nested(obs, self.device)
        return obs_t, has_batch_dim

    def choose_action(self, obs: dict, noise_std: float) -> np.ndarray:
        """根据结构化观测选择动作（支持单步和批量输入）。

        参数:
            obs: 结构化观测字典。可以是单步格式
                ``{"local": (n_agents, local_dim), ...}``
                或批量格式 ``{"local": (batch, n_agents, local_dim), ...}``。
            noise_std: 探索噪声标准差（0 表示确定性策略）。

        返回:
            np.ndarray: 单步输入时形状为 ``(action_dim,)``，
            批量输入时形状为 ``(batch, action_dim)``。取值范围 ``[-1, 1]``。
        """
        obs_t, has_batch_dim = self._prepare_obs(obs)
        # 推理阶段不需要计算梯度，节省显存和计算
        with torch.no_grad():
            action = self.act_from_torch_obs(obs_t, noise_std=noise_std).cpu().numpy()
        if not has_batch_dim:
            # 若原始输入无 batch 维度，去掉输出中添加的 batch 维度
            action = action[0]
        return action.astype(np.float32)

    def _soft_update(self) -> None:
        """对目标网络执行 Polyak 软更新。

        使用公式: target_param = tau * param + (1 - tau) * target_param
        通过缓慢跟踪在线网络参数，使训练目标更稳定。
        """
        # 更新 Critic 目标网络参数
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        # 更新 Actor 目标网络参数
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save_model(self, model_dir: str, episode: int) -> None:
        """将 Actor 和 Critic 的参数保存到磁盘。

        参数:
            model_dir: 检查点文件保存目录。
            episode: 用作检查点标签的 episode 编号。
        """
        os.makedirs(model_dir, exist_ok=True)
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, model_dir: str, episode: int) -> None:
        """从磁盘加载 Actor 和 Critic 的参数。

        加载后会重新构建目标网络（深拷贝），确保目标网络
        与在线网络参数完全一致。

        参数:
            model_dir: 包含检查点文件的目录。
            episode: 要加载的 episode 标签。
        """
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        # 加载后重建目标网络，使其与在线网络参数完全同步
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    @abstractmethod
    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        """从已转换的 PyTorch 观测批次中选择动作。

        参数:
            obs_t: 已在目标设备上的 torch 观测字典。
            noise_std: 探索噪声标准差。

        返回:
            torch.Tensor: 形状为 ``(batch, action_dim)`` 的动作张量。
        """
        ...

    @abstractmethod
    def train(self, replay_buffer: Any, agent_n: list) -> None:
        """从经验回放缓冲区采样一批数据并更新网络参数。

        参数:
            replay_buffer: 共享的经验回放缓冲区。
            agent_n: 所有智能体列表（用于集中式 Critic 训练）。
        """
        ...

    @abstractmethod
    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        """使用一个预采样的批次数据更新网络参数。

        参数:
            batch: 包含键 ``"obs"``、``"action"``、``"reward"``、
                ``"next_obs"``、``"done"`` 的字典，所有值均为
                已在目标设备上的 torch 张量。
            agent_n: 所有智能体列表。
        """
        ...
