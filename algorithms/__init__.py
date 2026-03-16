"""Formal algorithm entrypoints.

This package becomes the platform-level home for MADRL algorithms in the
third refactor round. Legacy imports can stay under ``madrl/`` as thin
compatibility shims, but new code should depend on ``algorithms``.
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
