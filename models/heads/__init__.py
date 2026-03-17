"""Head implementations."""

from models.heads.actor_head import DeterministicContinuousActorHead
from models.heads.critic_head import SingleQHead, TwinQHead

__all__ = ["DeterministicContinuousActorHead", "SingleQHead", "TwinQHead"]
