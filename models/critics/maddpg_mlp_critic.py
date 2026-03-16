"""Default MLP critic for MADDPG."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from models.base import CriticBase
from models.registry import register_critic
from models.utils import orthogonal_init

logger = logging.getLogger(__name__)


@register_critic("maddpg_mlp")
class MADDPGMLPCritic(CriticBase):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            logger.debug("use_orthogonal_init")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q = F.relu(self.fc1(s_a))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q
