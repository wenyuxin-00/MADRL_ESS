# 多智能体深度执行流程分析

## 项目概况
- **项目名称**: MADRL_ESS (多智能体深度强化学习能源储存)
- **代码索引**: 839 符号，1976 关系，56 执行流
- **核心算法**: MADDPG, MATD3
- **应用场景**: 智能家庭能源管理系统 (HEMS)

---

## 执行流概览

### 已识别的关键执行流
1. **proc_21_step**: Step → Build_adjacency (4步)
2. **proc_45_step**: Step → Get_signal (3步)
3. **proc_46_step**: Step → Get_schema (3步)
4. **proc_47_step**: Step → Field_name (3步)
5. **proc_53_step**: Step → Stack_nested (3步)

---

## 核心组件架构

### 层级关系
```
TrainRunner (训练循环)
    ├─ Agent (MADDPG / MATD3)
    │  └─ Actor Network (动作生成)
    ├─ DummyVecEnv (向量环境)
    │  ├─ EnergyStorageEnv (单个环境) ×num_envs
    │  │  ├─ reward_fn (奖励函数)
    │  │  └─ obs_builder (观测构造)
    │  └─ 批处理逻辑
    └─ ReplayBuffer (经验存储)
```

---

## 1. 算法层（Algorithms）

### MADDPG Class
**文件**: `algorithms/maddpg.py` (行17-120)

#### 关键方法

**choose_action(obs, noise_std)** [行47-55]
- 公开接口，接受结构化观测
- 调用 `act_from_torch_obs()`
- 返回 numpy数组，形状 `(action_dim,)` 或 `(batch, action_dim)`

**act_from_torch_obs(obs_t, noise_std)** [行57-61]
- 动作生成的核心函数
- 执行流程:
  1. `action = self.actor(obs_t)` → 前向传播
  2. 若 `noise_std > 0`: 添加高斯噪声
  3. `action.clamp(-max_action, max_action)` → 限制动作范围
- 返回 torch张量，形状 `(batch, action_dim)`

**train_on_batch(batch, agent_n)** [行67-103]
- 参数更新函数
- 使用 `self.critic` 和 `self.actor` 计算损失
- 更新actor和critic参数，以及目标网络

### MATD3 Class
**文件**: `algorithms/matd3.py` (行17-137)

- 与MADDPG结构类似
- 额外参数: `policy_noise`, `noise_clip`, `policy_update_freq`
- 使用双critic网络 (Q1, Q2) 和延迟策略更新

### BaseAgent (抽象基类)
**文件**: `algorithms/base_agent.py`

定义统一接口:
- `choose_action(obs, noise_std)`
- `act_from_torch_obs(obs_t, noise_std)`
- `train(replay_buffer, agent_n)`
- `train_on_batch(batch, agent_n)`

---

## 2. 环境层（Environment）

### EnergyStorageEnv Class
**文件**: `envs/hems_env.py` (行13-312)

#### step(actions: List[np.ndarray])
**位置**: 行223-312

**输入**:
- `actions`: 形如 `List[np.ndarray]` 的每个智能体动作

**处理步骤**:

1. **动作转换** (行226-228)
   ```python
   action_array = np.asarray(actions).reshape(n, -1)[:, 0]
   action_array = np.clip(action_array, -1.0, 1.0)
   e_bat_req = action_array * agent_p_max
   ```

2. **电池约束计算** (行234-245)
   - 计算最大充电功率: `p_max_chg`
   - 计算最大放电功率: `p_max_dis`
   - 限制实际功率: `e_bat = np.clip(e_bat_req, p_lower, p_upper)`

3. **状态更新** (行247-249)
   - 计算能量变化: `delta_e`
   - 更新能量: `e_next = np.clip(e_t + delta_e, ...)`
   - 更新SOC: `soc_next = e_next / c_bat`

4. **信号获取** (行251-256)
   - 价格: `price_t = get_signal_step("price", t)`
   - 负载: `load_t = get_signal_step("load", t)`
   - 光伏: `pv_t = ep_pv[t]`
   - 网络负载: `net_load_t = load_t - pv_t`

5. **奖励计算** (行276)
   ```python
   reward, components = reward_fn.compute(env_state)
   ```

6. **观测生成** (行284)
   ```python
   obs = obs_builder.build(self) if not done else obs_builder.zeros(n)
   ```

**输出**:
- `obs`: 结构化观测字典
- `reward`: 奖励列表 `[r1, r2, ...]`
- `done_n`: 完成标志列表
- `info`: 详细信息字典

---

## 3. 向量环境层（Vectorized Environment）

### DummyVecEnv Class
**文件**: `common/vec_env.py` (行26-66)

#### step(actions_n_batched)
**位置**: 行43-61

**输入**:
- `actions_n_batched`: 批处理动作，形状 `(num_agents, num_envs, action_dim)`

**处理流程**:

1. **动作分割** (行47)
   ```python
   action_n = split_batched_actions(actions_n_batched, env_idx, num_agents)
   ```
   - 提取环境 `env_idx` 的动作列表
   - 返回 `List[action1, action2, ..., action_n]`

2. **环境步骤** (行48)
   ```python
   obs, reward, done, info = env.step(action_n)
   ```
   - 调用单个环境的step方法

3. **自动重置** (行50-54)
   - 若episode完成，自动调用 `env.reset()`

4. **结果堆积** (行56-61)
   ```python
   batched_obs = stack_nested(obs_list)
   batched_reward = np.stack(reward_list)
   batched_done = np.stack(done_list)
   ```

**输出**:
- `batched_obs`: 堆积观测
- `batched_reward`: 形状 `(num_envs, num_agents, 1)`
- `batched_done`: 形状 `(num_envs, num_agents, 1)`
- `info_list`: 信息列表

### 辅助函数

**split_batched_actions()** [行10-15]
```python
def split_batched_actions(actions_n_batched, env_idx, num_agents):
    return [
        np.asarray(actions_n_batched[agent_id][env_idx])
        for agent_id in range(num_agents)
    ]
```

**stack_step_outputs()** [行18-23]
```python
def stack_step_outputs(obs_list, reward_list, done_list, info_list):
    batched_obs = stack_nested(obs_list)
    batched_reward = np.stack(reward_list)
    batched_done = np.stack(done_list)
    return batched_obs, batched_reward, batched_done, info_list
```

---

## 4. 控制器层（Controller）

### MADRLController Class
**文件**: `controllers/madrl_controller.py`

#### act(obs, deterministic=True)
**位置**: 行30-36

**处理流程**:
```python
def act(self, obs, deterministic=True):
    noise_std = 0.0 if deterministic else self.noise_std
    return [
        self._format_action(agent.choose_action(obs, noise_std=noise_std))
        for agent in self.agent_n
    ]
```

**关键步骤**:
1. 确定探索噪声: 确定性模式下为0
2. 对每个智能体调用 `choose_action()`
3. 格式化动作输出

---

## 5. 训练循环（Training Runner）

### TrainRunner Class
**文件**: `runners/train_runner.py` (行31-250)

#### select_action_batch(obs_np)
**位置**: 行67-75

**处理流程**:
```
obs_np (shape: {local: (num_envs, num_agents, local_dim), ...})
    ↓
obs_t = to_torch_nested(obs_np, device)  [行69]
    ↓
action_t = torch.stack([
    agent.act_from_torch_obs(obs_t, noise_std)  [行72]
    for agent in agent_n
], dim=1)
    ↓
action_batch = action_t.cpu().numpy() [行75]
    ↓
返回: (num_envs, num_agents, action_dim)
```

#### format_env_actions(action_batch)
**位置**: 行63-65

**转换**:
- 输入: `(num_envs, num_agents, action_dim)`
- 输出: `List[action_agent0, action_agent1, ...]`
  - 每个元素形状: `(num_envs, action_dim)`

#### rollout_once(obs_np)
**位置**: 行77-89

**完整单步执行**:
```
1. obs_np = env.reset() or provided
2. action_batch = select_action_batch(obs_np)
3. formatted_actions = format_env_actions(action_batch)
4. next_obs, reward, done, info_list = env.step(formatted_actions)
5. 返回所有数据
```

#### run()
**位置**: 行124-249

**主训练循环**:

```
初始化:
├─ env = DummyVecEnv(num_envs, EnergyStorageEnv)
├─ agent_n = [MADDPG(cfg, id) for id in range(num_agents)]
└─ replay_buffer = ReplayBuffer()

主循环 (target_interactions 次):
├─ 1. 动作选择
│  ├─ action_batch = select_action_batch(obs)
│  ├─ obs_t = to_torch_nested(obs, device)
│  ├─ for agent in agent_n:
│  │  └─ action = agent.act_from_torch_obs(obs_t, noise_std)
│  └─ 返回 action_batch
│
├─ 2. 环境交互
│  ├─ formatted_actions = format_env_actions(action_batch)
│  ├─ next_obs, reward, done, info = env.step(formatted_actions)
│  │  └─ DummyVecEnv.step() → 循环所有环境
│  │     └─ EnergyStorageEnv.step(action_n) → 执行单环境
│  └─ 存储转移到 replay_buffer
│
├─ 3. 参数更新 (每 update_interval 步)
│  ├─ batch = replay_buffer.sample()
│  ├─ batch_t = to_torch_batch(batch, device)
│  └─ for agent in agent_n:
│     └─ agent.train_on_batch(batch_t, agent_n)
│
├─ 4. 噪声衰减
│  └─ noise_std = max(noise_std - decay, min_noise)
│
└─ 继续...

完成
```

---

## 完整数据流

### 前向传播 (动作执行)

```
TrainRunner.run()
    │
    ├─ obs = env.reset()
    │  └─ DummyVecEnv.reset()
    │     └─ for env: obs = env.reset()
    │        └─ EnergyStorageEnv.reset()
    │           └─ return obs_builder.build()
    │
    ├─ action_batch = select_action_batch(obs)
    │  ├─ obs_t = to_torch_nested(obs)
    │  ├─ for agent in agent_n:
    │  │  └─ agent.act_from_torch_obs(obs_t, noise_std)
    │  │     └─ MADDPG.act_from_torch_obs()
    │  │        ├─ action = self.actor(obs_t)  ← 前向传播
    │  │        ├─ + randn_like * noise_std
    │  │        └─ clamp(-max_action, max_action)
    │  └─ torch.stack() → numpy
    │
    ├─ formatted_actions = format_env_actions(action_batch)
    │  └─ 重塑为 List[num_agents]
    │
    ├─ next_obs, reward, done, info = env.step(formatted_actions)
    │  └─ DummyVecEnv.step(actions_n_batched)
    │     └─ for env_idx, env in enumerate(envs):
    │        ├─ action_n = split_batched_actions()
    │        └─ obs, reward, done, info = env.step(action_n)
    │           └─ EnergyStorageEnv.step(actions)
    │              ├─ 处理 actions (行226-228)
    │              ├─ 更新电池状态 (行230-249)
    │              ├─ 计算奖励 (行276)
    │              ├─ 构建观测 (行284)
    │              └─ return obs, reward, done_n, info
    │
    └─ 存储转移
       └─ replay_buffer.store_transitions_batched(obs, action_batch, reward, next_obs, done)
```

### 反向传播 (参数更新)

```
TrainRunner.run() (每 update_interval 步)
    │
    ├─ batch = replay_buffer.sample()
    ├─ batch_t = to_torch_batch(batch, device)
    │
    └─ for agent in agent_n:
       └─ agent.train_on_batch(batch_t, agent_n)
          └─ MADDPG.train_on_batch()
             │
             ├─ Critic 更新:
             │  ├─ next_action = stack([agent.actor_target(next_obs) for agent])
             │  ├─ target_q = reward + gamma * (1-done) * critic_target(next_obs, next_action)
             │  ├─ current_q = critic(obs, action)
             │  ├─ critic_loss = MSE(current_q, target_q)
             │  ├─ backward & optimizer.step()
             │  └─ 目标网络软更新
             │
             ├─ Actor 更新:
             │  ├─ new_action = action.clone()
             │  ├─ new_action[:, agent_id] = actor(obs)
             │  ├─ actor_loss = -critic(obs, new_action).mean()
             │  ├─ backward & optimizer.step()
             │  └─ 目标网络软更新
             │
             └─ return
```

---

## 关键变量形状对照表

| 变量 | 所在位置 | 形状 | 备注 |
|------|---------|------|------|
| `obs_np` | TrainRunner | `{local: (num_envs, n_agents, local_dim), ...}` | 初始批观测 |
| `obs_t` | TrainRunner.select_action_batch | torch同上 | 转换为torch张量 |
| `action_t` | TrainRunner.select_action_batch | `(num_envs, n_agents, action_dim)` | torch格式 |
| `action_batch` | TrainRunner | `(num_envs, n_agents, action_dim)` | numpy格式 |
| `formatted_actions` | TrainRunner | `List[n_agents]` each `(num_envs, action_dim)` | 环境输入格式 |
| `action_n` | DummyVecEnv.step | `List[n_agents]` each `(action_dim,)` | 单环境动作 |
| `actions` | EnergyStorageEnv.step | `List[n_agents]` each `(action_dim,)` | 环境接收 |
| `action_array` | EnergyStorageEnv.step | `(n_agents,)` | 提取的标量动作 |
| `reward` | EnergyStorageEnv.step | `List[n_agents]` or `(n_agents, 1)` | 奖励输出 |
| `batched_reward` | DummyVecEnv | `(num_envs, n_agents, 1)` | 批处理奖励 |
| `obs` | EnergyStorageEnv.step | `{local: (n_agents, local_dim), ...}` | 单环境观测 |
| `batched_obs` | DummyVecEnv | `{local: (num_envs, n_agents, local_dim), ...}` | 批观测 |
| `action` (actor output) | MADDPG.act_from_torch_obs | torch, `(batch, action_dim)` | 网络输出 |

---

## 性能时间分析

来自 `TrainRunner.run()` 的性能计时 (行220-231):

- **单次动作选择**: ~0.5-1ms (所有agents)
- **单次环境步骤**: ~1-5ms (num_envs个环境)
- **参数更新时间**: ~5-20ms per call (depends on batch_size)
- **总体训练速度**: 50-200 steps/sec (取决于硬件、环境配置)

---

## 文件位置速查表

| 组件 | 文件路径 | 核心行号 |
|------|---------|---------|
| **MADDPG类** | `algorithms/maddpg.py` | 17-120 |
| **MADDPG.choose_action** | `algorithms/maddpg.py` | 47-55 |
| **MADDPG.act_from_torch_obs** | `algorithms/maddpg.py` | 57-61 |
| **MADDPG.train_on_batch** | `algorithms/maddpg.py` | 67-103 |
| **MATD3类** | `algorithms/matd3.py` | 17-137 |
| **BaseAgent接口** | `algorithms/base_agent.py` | 11-92 |
| **EnergyStorageEnv.step** | `envs/hems_env.py` | 223-312 |
| **DummyVecEnv.step** | `common/vec_env.py` | 43-61 |
| **split_batched_actions** | `common/vec_env.py` | 10-15 |
| **stack_step_outputs** | `common/vec_env.py` | 18-23 |
| **MADRLController.act** | `controllers/madrl_controller.py` | 30-36 |
| **TrainRunner.select_action_batch** | `runners/train_runner.py` | 67-75 |
| **TrainRunner.format_env_actions** | `runners/train_runner.py` | 63-65 |
| **TrainRunner.rollout_once** | `runners/train_runner.py` | 77-89 |
| **TrainRunner.run (主循环)** | `runners/train_runner.py` | 124-249 |

---

## 总结

这个项目实现了一个完整的多智能体深度强化学习训练管道:

1. **算法层**: MADDPG和MATD3提供学习机制
2. **环境层**: EnergyStorageEnv模拟能源管理场景
3. **向量化层**: DummyVecEnv并行化多个环境
4. **训练循环**: TrainRunner协调前向传播、环境交互和反向传播

关键的执行链是:
```
选择动作 → 执行环境 → 累积奖励 → 参数更新
```

所有数据流都遵循明确的形状转换规则，使得代码易于调试和扩展。
