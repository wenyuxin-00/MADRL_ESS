"""Algorithm registry.
算法注册表。

How to add a new MADRL algorithm (e.g., MAPPO) / 如何添加新算法:
    1. Create ``algorithms/mappo.py`` implementing ``BaseAgent``
    2. Register here::

           from algorithms.mappo import MAPPO
           register_agent("MAPPO", MAPPO)

    3. Import in ``algorithms/__init__.py``
    4. Use in config: ``cfg.algo.name = "MAPPO"``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms.maddpg import MADDPG
from algorithms.matd3 import MATD3

if TYPE_CHECKING:
    from algorithms.base_agent import BaseAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "MATD3": MATD3,
}


def register_agent(name: str, agent_cls: type[BaseAgent]) -> None:
    """Register a new MADRL algorithm class.
    注册一个新的 MADRL 算法类。
    """
    AGENT_REGISTRY[name] = agent_cls


def get_agent_cls(name: str) -> type[BaseAgent]:
    """Return the agent class registered under the given algorithm name.
    按名称返回已注册的算法类。
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}', available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]
