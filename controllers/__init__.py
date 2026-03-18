"""控制器统一接口，用于评估与对比。

所有控制器实现 ``BaseController``，提供两个核心方法:
    - reset()  -- episode 开始时调用
    - act(obs) -- 根据观测返回动作

已实现控制器:
    ZeroController       -- 零动作基线
    MADRLController      -- 多智能体 RL 控制器（封装 MADDPG/MATD3）
    MPCController        -- 模型预测控制基线
    ClassicDRLController -- 单智能体 DRL 基线（PPO/DQN/SAC）

子模块:
    madrl/ -- MADRL 算法实现（MADDPG, MATD3）
    mpc/   -- MPC 控制器
    drl/   -- 经典单智能体 DRL 控制器
"""

from controllers.base import BaseController
from controllers.drl.classic_drl_controller import ClassicDRLController
from controllers.madrl_controller import MADRLController
from controllers.mpc.mpc_controller import MPCController
from controllers.zero_controller import ZeroController

__all__ = [
    "BaseController",
    "ClassicDRLController",
    "MADRLController",
    "MPCController",
    "ZeroController",
]
