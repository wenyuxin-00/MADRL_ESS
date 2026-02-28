# common/vec_env.py
import multiprocessing as mp
import numpy as np

def worker(remote, parent_remote, env_class, args, mode):
    """
    独立进程的工作函数。
    在子进程中实例化环境，避免 multiprocessing 序列化整个环境实例导致的错误。
    """
    parent_remote.close()
    
    # 实例化当前进程专属的电力系统环境
    env = env_class(args, mode=mode)
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                # data 是当前环境接收到的 action list
                obs_n, r_n, done_n, info = env.step(data)
                
                # 经过 96 个时段跑完一天（done），立刻自动重置，无缝衔接下一天
                if all(done_n):
                    obs_n = env.reset()
                    # 可以在 info 中打个标记，告诉主循环这个 episode 结束了
                    info['episode_done'] = True 
                else:
                    info['episode_done'] = False
                    
                remote.send((obs_n, r_n, done_n, info))
                
            elif cmd == 'reset':
                obs_n = env.reset()
                remote.send(obs_n)
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not implemented.")
    except KeyboardInterrupt:
        print("Worker Process Interrupted.")
    finally:
        env.close()

class DummyVecEnv:
    """
    单进程向量化环境（Windows/Jupyter 救星）。
    避免了所有多进程的序列化、管道通信问题，
    同时保留了给 GPU 提供 Batch 数据的能力。
    """
    def __init__(self, num_envs, env_class, args, mode='train'):
        self.num_envs = num_envs
        self.num_agents = args.num_agents
        # 在主进程中直接实例化 32 个环境
        self.envs = [env_class(args, mode=mode) for _ in range(num_envs)]

    def reset(self):
        obs_n_list = [env.reset() for env in self.envs]
        return self._stack_obs(obs_n_list)

    def step(self, actions_n_batched):
        obs_n_list, r_n_list, done_n_list, info_list = [], [], [], []
        
        # 串行遍历 32 个环境进行推演（由于物理计算极快，这里的耗时可以接受）
        for env_idx, env in enumerate(self.envs):
            # 提取当前环境对应的动作
            action_n = [actions_n_batched[agent_id][env_idx] for agent_id in range(self.num_agents)]
            obs_n, r_n, done_n, info = env.step(action_n)
            
            # ★ 同样实现 Auto-Reset 机制
            if all(done_n):
                obs_n = env.reset()
                info['episode_done'] = True
            else:
                info['episode_done'] = False
                
            obs_n_list.append(obs_n)
            r_n_list.append(r_n)
            done_n_list.append(done_n)
            info_list.append(info)

        # 重新打包为以 agent 为主键的 Batch 数据，完美对接你的 MADDPG
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