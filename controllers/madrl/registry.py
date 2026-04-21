from __future__ import annotations
from typing import TYPE_CHECKING
from controllers.madrl.maddpg import MADDPG
from controllers.madrl.matd3 import MATD3
from controllers.madrl.matd3_safe_poc import MATD3SafePOC
if TYPE_CHECKING:
    from controllers.madrl.base_agent import BaseAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "MATD3": MATD3,
    "MATD3_SAFE_POC": MATD3SafePOC,
}
def register_agent(name: str, agent_cls: type[BaseAgent]) -> None:
    AGENT_REGISTRY[name] = agent_cls

def get_agent_cls(name: str) -> type[BaseAgent]:
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}', available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]
