"""MADRL 算法实现（MADDPG / MATD3）。

提供多智能体深度强化学习算法的具体实现。

已实现算法:
    MADDPG -- Multi-Agent Deep Deterministic Policy Gradient
    MATD3  -- Multi-Agent Twin Delayed DDPG (TD3 扩展)

如何添加新算法:
    1. 创建 ``controllers/madrl/your_algo.py``，继承 ``BaseAgent``
    2. 在 ``controllers/madrl/registry.py`` 中注册::

           from controllers.madrl.your_algo import YourAlgo
           AGENT_REGISTRY["YourAlgo"] = YourAlgo

    3. 在此 ``__init__.py`` 中导入
    4. 在配置中使用: ``cfg.algo.name = "YourAlgo"``

参见 ``controllers/madrl/base_agent.py`` 了解完整接口约定。
"""

from controllers.madrl.base_agent import BaseAgent
from controllers.madrl.maddpg import MADDPG
from controllers.madrl.matd3 import MATD3
from controllers.madrl.registry import AGENT_REGISTRY, get_agent_cls

__all__ = [
    "AGENT_REGISTRY",
    "BaseAgent",
    "MADDPG",
    "MATD3",
    "get_agent_cls",
]
