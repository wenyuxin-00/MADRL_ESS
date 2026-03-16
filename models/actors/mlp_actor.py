"""Default MLP actor implementation."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from models.base import ActorBase
from models.registry import register_actor
from models.utils import orthogonal_init

logger = logging.getLogger(__name__)


@register_actor("mlp")
class MLPActor(ActorBase):
    def __init__(self, args, agent_id):
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim_n[agent_id])
        if args.use_orthogonal_init:
            logger.debug("use_orthogonal_init")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))
        return a
