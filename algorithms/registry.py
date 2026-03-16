"""Algorithm registry for the platform-level algorithm package."""

from algorithms.maddpg import MADDPG
from algorithms.matd3 import MATD3

AGENT_REGISTRY: dict = {
    "MADDPG": MADDPG,
    "MATD3": MATD3,
}


def get_agent_cls(name: str):
    """Return the agent class registered under the given algorithm name."""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}', available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]
