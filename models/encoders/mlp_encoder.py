from __future__ import annotations
import torch.nn.functional as F
from torch import nn
from models.utils import orthogonal_init
class MLPEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, use_orthogonal_init: bool):
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # 第一层: 线性变换 + ReLU
        return F.relu(self.fc2(x)) # 第二层: 线性变换 + ReLU
