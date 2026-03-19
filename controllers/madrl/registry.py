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

# 算法名称 -> 智能体类的映射注册表
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "MATD3": MATD3,
}


def register_agent(name: str, agent_cls: type[BaseAgent]) -> None:
    """注册一个新的 MADRL 算法类到全局注册表。

    参数:
        name: 算法名称（如 ``"MADDPG"``、``"MATD3"``），作为注册表的键。
        agent_cls: 对应的智能体类，必须是 BaseAgent 的子类。

    注意:
        若名称已存在，会覆盖原有注册。
    """
    AGENT_REGISTRY[name] = agent_cls


def get_agent_cls(name: str) -> type[BaseAgent]:
    """按算法名称从注册表中获取对应的智能体类。

    参数:
        name: 算法名称（如 ``"MADDPG"``、``"MATD3"``）。

    返回:
        type[BaseAgent]: 已注册的智能体类。

    异常:
        ValueError: 当指定名称未在注册表中找到时抛出。
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}', available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]
