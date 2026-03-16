# common/vec_env.py
import numpy as np


class DummyVecEnv:
    """
    单进程向量化环境。

    支持两种 API：
      - 新 API（推荐）：DummyVecEnv(num_envs, env_fn)
          env_fn 是无参可调用对象，每次调用返回一个新的 env 实例。
          适合通过依赖注入组装的 env（来自 builder）。

      - 旧 API（向后兼容）：DummyVecEnv(num_envs, env_class, args, mode='train')
          直接传入 env 类和 args，保持与旧代码的兼容性。
    """

    def __init__(self, num_envs, env_fn_or_cls, args=None, mode='train'):
        self.num_envs = num_envs

        if args is None:
            # 新 API：env_fn_or_cls 是工厂函数
            self.envs = [env_fn_or_cls() for _ in range(num_envs)]
            self.num_agents = self.envs[0].n
        else:
            # 旧 API：env_fn_or_cls 是 env 类，args 提供配置
            self.envs = [env_fn_or_cls(args, mode=mode) for _ in range(num_envs)]
            self.num_agents = args.num_agents

    def reset(self):
        obs_n_list = [env.reset() for env in self.envs]
        return self._stack_obs(obs_n_list)

    def step(self, actions_n_batched):
        obs_n_list, r_n_list, done_n_list, info_list = [], [], [], []

        for env_idx, env in enumerate(self.envs):
            action_n = [actions_n_batched[agent_id][env_idx] for agent_id in range(self.num_agents)]
            obs_n, r_n, done_n, info = env.step(action_n)

            if all(done_n):
                obs_n = env.reset()
                info['episode_done'] = True
            else:
                info['episode_done'] = False

            obs_n_list.append(obs_n)
            r_n_list.append(r_n)
            done_n_list.append(done_n)
            info_list.append(info)

        batched_obs_n = self._stack_obs(obs_n_list)
        batched_r_n = [np.array([r_n[i] for r_n in r_n_list]).reshape(-1, 1) for i in range(self.num_agents)]
        batched_done_n = [np.array([done_n[i] for done_n in done_n_list]).reshape(-1, 1) for i in range(self.num_agents)]

        return batched_obs_n, batched_r_n, batched_done_n, info_list

    def _stack_obs(self, obs_n_list):
        batched_obs_n = []
        for agent_id in range(self.num_agents):
            agent_obs = np.stack([obs_n[agent_id] for obs_n in obs_n_list])
            batched_obs_n.append(agent_obs)
        return batched_obs_n

    def close(self):
        for env in self.envs:
            env.close()
