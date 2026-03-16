"""Critic model implementations."""

from models.critics.maddpg_mlp_critic import MADDPGMLPCritic
from models.critics.matd3_mlp_critic import MATD3MLPCritic

__all__ = ["MADDPGMLPCritic", "MATD3MLPCritic"]
