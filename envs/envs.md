# envs 架构说明

`envs` 是项目里的环境层，负责把数据集、观测、奖励、物理电网和并行采样组织成一个可被控制器训练或评估的 Gymnasium 环境。

最核心的主线是：

```text
scripts.builder.build_env(...)
    -> data.loaders.registry.build_dataset(...)
    -> envs.grid.deployments.build_agent_deployments(...)
    -> envs.grid.grid_core.GridCore(...)
    -> envs.rewards.NormalReward(...)
    -> envs.observation.default_builder.DefaultObservationBuilder(...)
    -> envs.grid_env.GridEnv(...)
```

训练时如果需要多个环境并行，还会再包一层：

```text
scripts.builder._build_train_vec_env(...)
    -> envs.subproc_vec_env.DummyVecEnv 或 SubprocVecEnv
    -> 多个 GridEnv
```

## 目录结构

```text
envs/
    grid_env.py
    subproc_vec_env.py
    grid/
        deployments.py
        grid_core.py
        net_builder.py
    observation/
        default_builder.py
        normalization.py
    rewards/
        NormalReward.py
```

## 各文件作用

### grid_env.py

`GridEnv` 是所有控制器真正交互的环境入口。它从 dataset 取一个 episode，维护 SoC，接收动作，调用电网物理层，计算 reward，并返回下一步 observation。

它主要负责：

- `reset(...)`：选择 episode，初始化 SoC，返回初始观测。
- `step(actions)`：执行 `[battery_action, pv_action]`，更新储能状态，调用潮流，计算奖励，返回下一步结果。
- `get_signal(...)` / `get_signal_step(...)`：给观测构造器读取当前 episode 的 load、PV、电价等信号。

### subproc_vec_env.py

这个文件提供并行环境包装，用于训练时同时跑多个 `GridEnv`。

它主要负责：

- `DummyVecEnv`：单进程内创建多个环境，调试和小规模训练更直观。
- `SubprocVecEnv`：多进程创建多个环境，训练吞吐更高。
- `ParallelEpisodeSampler`：给多个环境分配 episode，避免并行环境重复采同一个 active episode。

### grid/deployments.py

这个文件把配置里的 agent bus、battery capacity、charge rate 等参数解析成每个 agent 的部署信息。

它主要负责：

- `AgentDeployment`：描述一个 agent 接在哪个 bus 上，以及它的电池容量、功率、SoC 边界。
- `resolve_fixed_battery_spec(...)`：把标量或列表形式的电池配置统一成逐 agent 参数。
- `build_agent_deployments(cfg)`：从 `cfg` 生成 `GridCore` 需要的部署列表。

### grid/net_builder.py

这个文件负责创建和清理 pandapower/simbench 电网对象。

它主要负责：

- `build_simbench_net(sb_code)`：根据 simbench code 创建电网。
- `zero_static_power_elements(net)`：把原始网络里的静态负荷/发电归零，让 prosumer 数据成为唯一运行信号来源。

### grid/grid_core.py

这个文件是电网物理层，封装 pandapower 潮流计算。

它主要负责：

- `GridCore.reset(...)`：恢复 agent bus 的基准状态。
- `GridCore.step(...)`：把 agent 的电池功率和净负荷注入电网，运行潮流。
- `GridStepResult`：返回电压、线路负载、变压器负载、约束违规等物理结果。

### observation/default_builder.py

这个文件把环境内部状态转换成控制器看到的 observation。

它主要负责：

- 构造 local features，比如 SoC、当前负荷、当前 PV、当前价格。
- 构造 sequence features，比如未来电价序列、负荷序列、PV 序列。
- 构造 `safety_local`，供安全投影器判断动作是否可行。
- 支持 raw observation 和 normalized observation 两种口径。

### observation/normalization.py

这个文件负责拟合和应用 observation 归一化。

它主要负责：

- 从训练 dataset 的 active 样本里提取 load、PV、电价。
- 根据配置拟合 robust/tanh/capacity 等归一化参数。
- `ObservationNormalizer` 在构造 observation 时把 raw 值转换成模型输入口径。

### rewards/NormalReward.py

这个文件定义当前主线 MADRL reward。

它主要负责：

- 校验旧 reward 配置不能静默进入主线。
- 根据电价、储能功率、SoC、潮流违规等环境状态计算每个 agent 的 reward。
- 输出 reward component，方便训练日志和结果分析拆分每个奖励项。

## 运行时数据流

一次 `GridEnv.step(...)` 大致是：

```text
controller 输出 actions
    -> GridEnv 拆成 battery_action 和 pv_action
    -> GridEnv 校验动作是否在本地 SoC 约束内
    -> GridEnv 更新储能状态
    -> GridEnv 读取当前 load、PV、电价
    -> GridCore 运行潮流
    -> NormalReward.compute(...) 计算 reward
    -> DefaultObservationBuilder.build(...) 生成下一步 observation
```

其中 dataset 不在 `envs` 里实现。dataset 来自 `data/loaders`，负责读取 CSV、对齐时间、切 episode；`envs` 负责把一个 episode 变成可交互的控制环境。

## 建议阅读顺序

1. 先读 `scripts/builder.py` 的 `build_env(...)`，理解环境是如何被装配出来的。
2. 再读 `data/loaders/registry.py` 和 `data/loaders/prosumer.py`，理解 dataset 如何提供 episode。
3. 然后读 `envs/grid_env.py`，这是环境交互主入口。
4. 接着读 `envs/observation/default_builder.py`，理解控制器实际看到什么。
5. 再读 `envs/observation/normalization.py`，理解 observation 的数值口径。
6. 再读 `envs/rewards/NormalReward.py`，理解 reward 由哪些 component 构成。
7. 然后读 `envs/grid/deployments.py` 和 `envs/grid/grid_core.py`，理解电网物理层。
8. 最后读 `envs/subproc_vec_env.py`，理解训练时多个环境如何并行采样。

## 快速测试命令

只验证 `envs` 主线：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_grid_env.py tests/test_observation_schema.py tests/test_normal_reward.py tests/test_grid_core.py tests/test_builder_vec_env.py tests/test_parallel_episode_sampling.py
```

连同 dataset 装配一起验证：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_registries.py tests/test_prosumer_integration.py tests/test_prosumer_dataset.py tests/test_grid_env.py tests/test_observation_schema.py
```

## 本次清理结果

这次清理删除的是没有主线引用的内容：

- `envs/**/__pycache__`：Python 编译缓存。
- `envs/grid/analysis`：只剩缓存，没有源码引用。
- `envs/grid/topology`：只剩空的 `__init__.py` 占位文件，没有源码引用。

保留的源码文件都有明确主线引用或测试覆盖。
