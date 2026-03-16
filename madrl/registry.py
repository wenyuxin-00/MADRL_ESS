"""Compatibility layer for legacy algorithm registry imports."""

from algorithms.registry import AGENT_REGISTRY, get_agent_cls

__all__ = ["AGENT_REGISTRY", "get_agent_cls"]
