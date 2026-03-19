"""向量化环境封装（DummyVecEnv）。

在单进程中串行运行多个环境副本，提供统一的批量接口。

主要类:
    DummyVecEnv -- 单进程向量化环境
"""

from __future__ import annotations

import numpy as np

from scripts.utils.nested import stack_nested


def split_batched_actions(actions_n_batched, env_idx: int, num_agents: int):
    """从按智能体维度组织的批量动作中提取单个环境的动作列表。

    参数:
        actions_n_batched: 按 [agent_id][env_idx] 组织的批量动作
        env_idx: 要提取的环境索引
        num_agents: 智能体数量

    返回:
        list[np.ndarray]: 该环境中每个智能体的动作列表，每个元素 shape 由动作空间决定
    """
    return [
        # 从批量动作中按 agent_id 和 env_idx 两个维度索引，取出单个智能体的动作
        np.asarray(actions_n_batched[agent_id][env_idx], dtype=np.float32)
        for agent_id in range(int(num_agents))
    ]


def stack_step_outputs(obs_list, reward_list, done_list, info_list):
    """将各环境的 step 输出堆叠为批量数组。

    参数:
        obs_list: 各环境的观测列表（支持嵌套字典结构）
        reward_list: 各环境的奖励数组列表
        done_list: 各环境的完成标志数组列表
        info_list: 各环境的附加信息字典列表

    返回:
        tuple: (batched_obs, batched_reward, batched_done, info_list)
            - batched_obs: 嵌套字典，叶节点在 axis=0 上堆叠
            - batched_reward: shape (num_envs, num_agents, 1)
            - batched_done: shape (num_envs, num_agents, 1)
            - info_list: 原样返回的附加信息列表
    """
    # 使用 stack_nested 处理可能含嵌套字典的观测结构
    batched_obs = stack_nested(obs_list)
    batched_reward = np.stack(reward_list, axis=0)
    batched_done = np.stack(done_list, axis=0)
    return batched_obs, batched_reward, batched_done, info_list


class DummyVecEnv:
    """单进程向量化环境，在同一进程中串行运行多个环境副本。

    通过统一的批量接口（reset / step / close）管理多个环境实例，
    适用于轻量级环境或调试场景。

    属性:
        num_envs: 并行环境数量
        envs: 环境实例列表
        num_agents: 每个环境中的智能体数量
    """

    def __init__(self, num_envs, env_fn_or_cls, cfg=None, mode: str = "train"):
        """初始化向量化环境。

        参数:
            num_envs: 需要创建的环境数量
            env_fn_or_cls: 环境工厂函数或类（无参调用时用于无配置场景）
            cfg: 实验配置对象，为 None 时使用无参构造
            mode: 运行模式，"train" 或 "eval"
        """
        self.num_envs = int(num_envs)

        if cfg is None:
            # 无配置模式：直接无参调用工厂函数
            self.envs = [env_fn_or_cls() for _ in range(self.num_envs)]
            self.num_agents = self.envs[0].n
        else:
            # 有配置模式：传入配置和运行模式
            self.envs = [env_fn_or_cls(cfg, mode=mode) for _ in range(self.num_envs)]
            self.num_agents = cfg.env.num_agents

    def reset(self):
        """重置所有环境并返回堆叠后的初始观测。

        返回:
            dict[str, np.ndarray]: 嵌套字典，各字段在 axis=0 上按环境索引堆叠
        """
        obs_list = [env.reset() for env in self.envs]
        return stack_nested(obs_list)

    def step(self, actions_n_batched):
        """所有环境执行一步，自动处理 episode 结束时的重置。

        参数:
            actions_n_batched: 按 [agent_id][env_idx] 组织的批量动作

        返回:
            tuple: (batched_obs, batched_reward, batched_done, info_list)
        """
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for env_idx, env in enumerate(self.envs):
            # 从批量动作中提取当前环境的各智能体动作
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            obs, reward, done, info = env.step(action_n)

            if all(done):
                # episode 结束，自动重置并标记
                obs = env.reset()
                info["episode_done"] = True
            else:
                info["episode_done"] = False

            obs_list.append(obs)
            # 将奖励和完成标志重塑为 (num_agents, 1) 以便后续堆叠
            reward_list.append(np.asarray(reward, dtype=np.float32).reshape(self.num_agents, 1))
            done_list.append(np.asarray(done, dtype=np.float32).reshape(self.num_agents, 1))
            info_list.append(info)

        return stack_step_outputs(obs_list, reward_list, done_list, info_list)

    def close(self):
        """关闭所有环境实例，释放资源。"""
        for env in self.envs:
            env.close()
