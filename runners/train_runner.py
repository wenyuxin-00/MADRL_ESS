"""
runners/train_runner.py
职责：封装多智能体强化学习的训练循环。

TrainRunner 负责：
  - 接收预构建的并行训练环境（DummyVecEnv）和评估环境
  - 通过 algorithms.registry 按算法名实例化 agent
  - 管理 ReplayBuffer、TensorBoard writer、episode history
  - 执行主训练循环（run），含动作选取、env 交互、buffer 存储、网络更新
  - save_model / load_model：保存/加载 actor+critic 权重

第二步重构：env 由 core/builder.py 装配后注入，runner 不再自己构建 env。
"""

import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from algorithms.registry import get_agent_cls
from common.replay_buffer import ReplayBuffer
from runners.checkpoints import build_checkpoint_manifest, write_checkpoint_manifest


class TrainRunner:
    """多智能体训练 Runner。

    Parameters
    ----------
    args : Config
        全局超参数对象（来自 configs/default_config.py）。
        须已包含 N, obs_dim_n, action_dim_n（由 builder 设置）。
    train_env : DummyVecEnv
        预构建的并行训练环境。
    eval_env : EnergyStorageEnv
        预构建的评估环境（单实例）。
    env_name : str
        环境名称，用于日志命名，默认 "EnergyStorageEnv"。
    number : int
        实验编号，用于区分不同 run 的日志目录。
    seed : int
        随机种子，用于 numpy 和 torch。
    """

    def __init__(self, args, train_env, eval_env, env_name="EnergyStorageEnv", number=1, seed=0):
        self.args = args
        self.env_name = env_name

        self.env = train_env
        self.env_evaluate = eval_env

        np.random.seed(seed)
        torch.manual_seed(seed)

        # 通过注册表选择算法
        agent_cls = get_agent_cls(self.args.algorithm)
        self.agent_n = [agent_cls(args, i) for i in range(self.args.N)]

        self.replay_buffer = ReplayBuffer(self.args)
        self.writer = SummaryWriter(log_dir=f"runs/{self.args.algorithm}_{env_name}_{number}_seed_{seed}")

        self.history = []
        self.episode_rewards = []
        self.total_steps = 0
        self.episodes_completed = 0
        self.noise_std = self.args.noise_std_init

    @staticmethod
    def _to_scalar(x) -> float:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])

    def _init_env_history(self):
        """每个并行环境一个 episode 账本（分量键由 reward_fn.component_meta 动态生成）"""
        hist = {
            "price":      [],
            "e_bat_req":  [[] for _ in range(self.args.N)],
            "e_bat_exec": [[] for _ in range(self.args.N)],
            "soc":        [[float(self.args.init_soc)] for _ in range(self.args.N)],
            "r_total_sum": [],
        }
        for meta in self.env_evaluate.reward_fn.component_meta:
            hist[f"{meta.key}_sum"] = []
        return hist

    def save_model(self, model_dir: str, episode: int):
        """保存所有 agent 权重，并写入 latest checkpoint manifest。

        在并行环境下，实际完成的 episode 数可能与名义上的
        ``Config.train_episodes`` 不完全相等，因此 manifest 额外记录
        ``episodes_completed`` 作为后续评估/加载的可信来源。
        """
        algo_dir = os.path.join(model_dir, self.args.algorithm)
        os.makedirs(algo_dir, exist_ok=True)
        for agent in self.agent_n:
            agent.save_model(algo_dir, episode)
        manifest = build_checkpoint_manifest(
            algorithm=self.args.algorithm,
            saved_episode_tag=episode,
            episodes_completed=self.episodes_completed,
            total_steps=self.total_steps,
            num_envs=self.args.num_envs,
            episode_limit=self.args.episode_limit,
            save_dir=algo_dir,
        )
        write_checkpoint_manifest(algo_dir, manifest)
        print(f"Models saved to: {algo_dir}")

    def load_model(self, model_dir: str, episode: int):
        """加载所有 agent 的 actor/critic 权重（含 target 网络同步）。"""
        algo_dir = os.path.join(model_dir, self.args.algorithm)
        for agent in self.agent_n:
            agent.load_model(algo_dir, episode)
        print(f"Models loaded from: {algo_dir}")

    def run(self):
        """主训练循环。

        Returns
        -------
        int
            本次训练完成的 episode 总数。
        """
        target_interactions = self.args.max_train_steps // self.args.num_envs
        interaction_step = 0
        episodes_completed = 0

        reward_metas = self.env_evaluate.reward_fn.component_meta

        active_histories = [self._init_env_history() for _ in range(self.args.num_envs)]
        active_ep_rewards = np.zeros(self.args.num_envs, dtype=np.float32)

        pbar = tqdm(total=target_interactions, desc="Training (Parallel)", unit="iters")

        try:
            obs_n = self.env.reset()

            while interaction_step < target_interactions:
                # 1) 动作
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]

                # 2) 交互
                obs_next_n, r_n, done_n, info_list = self.env.step(a_n)

                # 3) 记录
                for env_idx in range(self.args.num_envs):
                    info = info_list[env_idx]
                    hist = active_histories[env_idx]

                    hist["price"].append(float(info.get("price", 0.0)))

                    e_req = np.asarray(info["e_bat_req"], dtype=np.float32).reshape(self.args.N)
                    e_exec = np.asarray(info["e_bat"], dtype=np.float32).reshape(self.args.N)
                    soc_next = np.asarray(info["soc_next"], dtype=np.float32).reshape(self.args.N)

                    for i in range(self.args.N):
                        hist["e_bat_req"][i].append(float(e_req[i]))
                        hist["e_bat_exec"][i].append(float(e_exec[i]))
                        hist["soc"][i].append(float(soc_next[i]))

                    step_total = sum(self._to_scalar(r_n[a][env_idx]) for a in range(self.args.N))
                    hist["r_total_sum"].append(step_total)
                    active_ep_rewards[env_idx] += step_total

                    for meta in reward_metas:
                        raw = float(np.sum(np.asarray(info[meta.key], dtype=np.float32)))
                        hist[f"{meta.key}_sum"].append(meta.sign * raw)

                # 4) 存 buffer
                self.replay_buffer.store_transitions_batched(obs_n, a_n, r_n, obs_next_n, done_n)

                obs_n = obs_next_n
                interaction_step += 1
                self.total_steps += self.args.num_envs

                # 5) episode 结束
                for env_idx in range(self.args.num_envs):
                    if bool(info_list[env_idx].get("episode_done", False)):
                        if len(self.history) >= 2000:
                            self.history.pop(0)
                        self.history.append(active_histories[env_idx])

                        ep_reward = float(active_ep_rewards[env_idx])
                        self.episode_rewards.append(ep_reward)
                        self.writer.add_scalar("train_episode_total_reward", ep_reward, global_step=self.total_steps)

                        active_histories[env_idx] = self._init_env_history()
                        active_ep_rewards[env_idx] = 0.0
                        episodes_completed += 1
                        self.episodes_completed = episodes_completed

                # 6) 噪声衰减
                if self.args.use_noise_decay:
                    self.noise_std = max(self.noise_std - self.args.noise_std_decay, self.args.noise_std_min)

                # 7) 更新
                if self.replay_buffer.current_size > self.args.batch_size and interaction_step % self.args.update_interval == 0:
                    for _ in range(self.args.updates_per_step):
                        for agent_id in range(self.args.N):
                            self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                pbar.update(1)
                if len(self.episode_rewards) > 0:
                    avg_reward = float(np.mean(self.episode_rewards[-50:]))
                    pbar.set_postfix({"avg_reward": f"{avg_reward:.2f}", "steps": self.total_steps})

        finally:
            pbar.close()
            self.env.close()
            self.env_evaluate.close()
            self.writer.close()

        self.episodes_completed = episodes_completed
        return episodes_completed
