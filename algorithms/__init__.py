"""Formal algorithm entrypoints for the canonical MADRL stack."""

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
