"""MADRL 算法注册表。

根据算法名称（如 "MADDPG"、"MATD3"）返回对应的智能体类。

主要函数:
    get_agent_cls -- 根据名称获取算法类
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from controllers.madrl.maddpg import MADDPG
from controllers.madrl.matd3 import MATD3

if TYPE_CHECKING:
    from controllers.madrl.base_agent import BaseAgent

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
