"""Multi-Agent Deep Reinforcement Learning algorithm implementations.
多智能体深度强化学习算法实现。

Current algorithms / 已实现算法:
    - MADDPG  -- Multi-Agent Deep Deterministic Policy Gradient
    - MATD3   -- Multi-Agent Twin Delayed DDPG (TD3 extension)

How to add a new algorithm / 如何添加新算法:
    1. Create ``algorithms/your_algo.py``, implementing ``BaseAgent``
    2. Register in ``algorithms/registry.py``::

           from algorithms.your_algo import YourAlgo
           AGENT_REGISTRY["YourAlgo"] = YourAlgo

    3. Import in this ``__init__.py``
    4. Use in config: ``cfg.algo.name = "YourAlgo"``

See ``algorithms/base_agent.py`` for the full interface contract.
"""

from algorithms.base_agent import BaseAgent
from algorithms.maddpg import MADDPG
from algorithms.matd3 import MATD3
from algorithms.registry import AGENT_REGISTRY, get_agent_cls

__all__ = [
    "AGENT_REGISTRY",
    "BaseAgent",
    "MADDPG",
    "MATD3",
    "get_agent_cls",
]
