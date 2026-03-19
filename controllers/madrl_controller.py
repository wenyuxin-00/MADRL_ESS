"""多智能体深度强化学习（MADRL）控制器。

封装 MADRL 算法（MADDPG / MATD3）的高层控制器接口，
负责多智能体联合动作选择与训练调度。

主要类:
    MADRLController -- MADRL 控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class MADRLController(BaseController):
    """多智能体深度强化学习（MADRL）高层控制器包装。

    将训练好的多个 MADRL 智能体（如 MADDPG / MATD3）封装为统一的
    控制器接口，负责在评估或部署阶段协调所有智能体的联合动作选择。

    采用分布式执行（DE）模式：每个智能体仅基于自身的局部观测
    独立输出动作，不依赖其他智能体的信息。

    属性:
        agent_n: 训练好的智能体列表，每个智能体实现 ``choose_action`` 方法。
        noise_std: 非确定性模式下的探索噪声标准差。
    """

    def __init__(self, agent_n: list, noise_std: float = 0.0) -> None:
        """初始化 MADRL 控制器。

        参数:
            agent_n: 训练好的智能体列表（BaseAgent 子类实例）。
            noise_std: 探索噪声标准差，确定性模式下不使用。
        """
        self.agent_n = list(agent_n)
        self.noise_std = float(noise_std)

    def reset(self) -> None:
        """当前前馈策略无内部状态，无需重置。"""

    @staticmethod
    def _format_action(action: object) -> np.ndarray:
        """将智能体输出的动作统一转换为一维 numpy 数组。

        参数:
            action: 智能体返回的动作，可能是标量、数组或张量。

        返回:
            np.ndarray: 形状为 ``(action_dim,)`` 的 float32 数组。
        """
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 0:
            # 标量动作需要扩展为一维数组以满足环境接口要求
            return action_array.reshape(1)
        return action_array

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """运行所有智能体，返回环境可执行的动作列表。

        参数:
            obs: 环境观测字典，将被传递给每个智能体的 ``choose_action``。
            deterministic: 若为 True，使用零噪声（确定性策略）。

        返回:
            list[np.ndarray]: 每个智能体一个动作数组。
        """
        # 确定性模式下强制将噪声设为 0
        noise_std = 0.0 if deterministic else self.noise_std
        return [
            self._format_action(agent.choose_action(obs, noise_std=noise_std))
            for agent in self.agent_n
        ]
