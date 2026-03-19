"""多进程向量化环境封装（SubprocVecEnv）。

使用多进程并行运行多个环境副本，适合 CPU 密集的环境
（如 pandapower 潮流计算）。

主要类:
    SubprocVecEnv -- 多进程向量化环境
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Client, Listener

import numpy as np
import torch

from scripts.utils.nested import stack_nested
from scripts.utils.torch_runtime import configure_torch_runtime
from envs.vec_env import split_batched_actions, stack_step_outputs


def _make_worker_env(cfg, mode: str, *, worker_rank: int, seed: int | None):
    """在子进程里构建环境，并同步 runtime / seed。

    每个子进程需要独立的配置副本，以避免进程间共享状态冲突。
    子进程强制使用 CPU 设备（GPU 操作仅在主进程中进行）。

    参数:
        cfg: 实验配置对象
        mode: 运行模式，"train" 或 "eval"
        worker_rank: 当前 worker 的序号，用于确定性随机种子偏移
        seed: 基础随机种子，为 None 时不设置

    返回:
        构建好的环境实例
    """
    from copy import deepcopy

    from scripts.builder import build_env

    # 深拷贝配置，避免子进程间互相污染
    worker_cfg = deepcopy(cfg)
    # 子进程强制使用 CPU
    worker_cfg.runtime.device = torch.device("cpu")
    configure_torch_runtime(
        worker_cfg,
        device="cpu",
        seed=seed,
        worker_rank=worker_rank,
    )
    return build_env(worker_cfg, mode=mode)


def _subproc_worker(address, authkey: bytes, cfg, mode: str, worker_rank: int, seed: int | None) -> None:
    """运行单个环境子进程的主循环。

    通过 TCP 连接（AF_INET）与主进程通信，接收指令并返回结果。
    支持的指令: "reset"、"step"、"get_num_agents"、"close"。

    参数:
        address: 主进程 Listener 的地址 (host, port)
        authkey: 连接认证密钥
        cfg: 实验配置对象
        mode: 运行模式
        worker_rank: 当前 worker 序号
        seed: 随机种子
    """
    # 建立与主进程的 TCP 连接
    remote = Client(address, family="AF_INET", authkey=authkey)
    env = _make_worker_env(cfg, mode=mode, worker_rank=worker_rank, seed=seed)

    try:
        while True:
            # 接收主进程的指令和载荷
            cmd, payload = remote.recv()

            if cmd == "reset":
                obs = env.reset(episode_idx=payload)
                remote.send(obs)
                continue

            if cmd == "step":
                obs, reward, done, info = env.step(payload)
                if all(done):
                    # episode 结束时自动重置
                    obs = env.reset()
                    info["episode_done"] = True
                else:
                    info["episode_done"] = False

                # 将奖励和完成标志重塑为 (n_agents, 1) 以统一格式
                remote.send(
                    (
                        obs,
                        np.asarray(reward, dtype=np.float32).reshape(env.n, 1),
                        np.asarray(done, dtype=np.float32).reshape(env.n, 1),
                        info,
                    )
                )
                continue

            if cmd == "get_num_agents":
                # 返回智能体数量，用于主进程初始化
                remote.send(int(env.n))
                continue

            if cmd == "close":
                remote.close()
                break

            raise ValueError(f"Unknown worker command '{cmd}'.")
    except EOFError:
        # 主进程意外断开连接时静默退出
        pass
    finally:
        env.close()


class SubprocVecEnv:
    """多进程向量化环境，为每个环境启动独立子进程实现真正的并行采集。

    使用 TCP 套接字（AF_INET）进行进程间通信，避免 Windows 上
    multiprocessing.Pipe 的兼容性问题。适合 CPU 密集型环境
    （如包含 pandapower 潮流计算的 GridEnv）。

    属性:
        num_envs: 并行环境（子进程）数量
        num_agents: 每个环境中的智能体数量
        closed: 是否已关闭
        remotes: 与各子进程的连接列表
        processes: 子进程对象列表
    """

    def __init__(self, num_envs: int, cfg, mode: str = "train", seed: int | None = None):
        """启动子进程并建立通信连接。

        参数:
            num_envs: 需要创建的并行环境数量
            cfg: 实验配置对象
            mode: 运行模式，"train" 或 "eval"
            seed: 随机种子，为 None 时不设置
        """
        self.num_envs = int(num_envs)
        self.closed = False
        self.authkey = b"madrl_subproc_vec_env"

        # 使用 "spawn" 上下文以确保跨平台兼容性（尤其是 Windows）
        ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []

        for worker_rank in range(self.num_envs):
            # 在随机端口上创建 TCP 监听器，等待子进程连接
            listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=self.authkey)
            process = ctx.Process(
                target=_subproc_worker,
                args=(listener.address, self.authkey, cfg, mode, worker_rank, seed),
                daemon=True,
            )
            process.start()
            # 等待子进程连接，获取双向通信管道
            remote = listener.accept()
            listener.close()
            self.remotes.append(remote)
            self.processes.append(process)

        # 从第一个 worker 查询智能体数量
        self.remotes[0].send(("get_num_agents", None))
        self.num_agents = int(self.remotes[0].recv())

    def reset(self):
        """向所有子进程发送重置指令并收集堆叠后的初始观测。

        返回:
            dict[str, np.ndarray]: 嵌套字典，各字段在 axis=0 上按环境索引堆叠
        """
        for remote in self.remotes:
            remote.send(("reset", None))
        obs_list = [remote.recv() for remote in self.remotes]
        return stack_nested(obs_list)

    def step(self, actions_n_batched):
        """向所有子进程发送动作并收集结果（真正并行执行）。

        参数:
            actions_n_batched: 按 [agent_id][env_idx] 组织的批量动作

        返回:
            tuple: (batched_obs, batched_reward, batched_done, info_list)
        """
        # 先向所有子进程发送动作（非阻塞），实现并行执行
        for env_idx, remote in enumerate(self.remotes):
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            remote.send(("step", action_n))

        # 收集所有子进程的执行结果
        results = [remote.recv() for remote in self.remotes]
        obs_list, reward_list, done_list, info_list = zip(*results)
        return stack_step_outputs(list(obs_list), list(reward_list), list(done_list), list(info_list))

    def close(self):
        """优雅关闭所有子进程，先发送关闭指令再等待退出。"""
        if self.closed:
            return

        # 向所有子进程发送关闭指令，忽略已断开的连接
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

        # 等待子进程退出，超时后强制终止
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        self.closed = True
