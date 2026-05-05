from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from configs.cfg import Cfg


def _init(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight); nn.init.zeros_(layer.bias)


def to_torch_obs(obs: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: (value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32, device=device)) for key, value in obs.items()}


def add_batch_obs(obs: dict[str, object]) -> dict[str, object]:
    return {key: (value.unsqueeze(0) if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)) for key, value in obs.items()}


def actor_features(obs: dict[str, torch.Tensor], agent_id: int) -> torch.Tensor:
    return torch.cat([
        obs["madrl_local"][:, int(agent_id)],
        obs["wholesale_price_relative_seq"].reshape(obs["madrl_local"].shape[0], -1),
        obs["wholesale_price_spread_seq"].reshape(obs["madrl_local"].shape[0], -1),
        obs["load_seq"][:, int(agent_id)].reshape(obs["madrl_local"].shape[0], -1),
        obs["pv_seq"][:, int(agent_id)].reshape(obs["madrl_local"].shape[0], -1),
    ], dim=-1)


def critic_features(obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    batch = int(action.shape[0])
    return torch.cat([
        obs["madrl_local"].reshape(batch, -1),
        obs["wholesale_price_relative_seq"].reshape(batch, -1),
        obs["wholesale_price_spread_seq"].reshape(batch, -1),
        obs["load_seq"].reshape(batch, -1),
        obs["pv_seq"].reshape(batch, -1),
        action.reshape(batch, -1),
    ], dim=-1)


class Actor(nn.Module):
    def __init__(self, cfg: Cfg, agent_id: int) -> None:
        super().__init__(); self.agent_id = int(agent_id)
        dim = 5 + int(cfg.obs.sequence_length) * 4
        self.fc1 = nn.Linear(dim, int(cfg.model.hidden_dim)); self.fc2 = nn.Linear(int(cfg.model.hidden_dim), int(cfg.model.hidden_dim)); self.out = nn.Linear(int(cfg.model.hidden_dim), int(cfg.model.action_dim))
        self.apply(_init)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = actor_features(obs, self.agent_id)
        return torch.tanh(self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))) 


class Critic(nn.Module):
    def __init__(self, cfg: Cfg, twin: bool) -> None:
        super().__init__(); n, s, h = int(cfg.env.num_agents), int(cfg.obs.sequence_length), int(cfg.model.hidden_dim)
        dim = n * 5 + 2 * s + 2 * n * s + n * int(cfg.model.action_dim)
        self.twin = bool(twin)
        self.fc1 = nn.Linear(dim, h); self.fc2 = nn.Linear(h, h); self.q1 = nn.Linear(h, 1)
        self.fc1b = nn.Linear(dim, h) if self.twin else None; self.fc2b = nn.Linear(h, h) if self.twin else None; self.q2 = nn.Linear(h, 1) if self.twin else None
        self.apply(_init)

    def _q(self, x: torch.Tensor, fc1: nn.Linear, fc2: nn.Linear, head: nn.Linear) -> torch.Tensor:
        return head(F.relu(fc2(F.relu(fc1(x))))).squeeze(-1)

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = critic_features(obs, action)
        q1 = self._q(x, self.fc1, self.fc2, self.q1)
        return (q1, self._q(x, self.fc1b, self.fc2b, self.q2)) if self.twin else q1


def build_actors(cfg: Cfg) -> list[Actor]:
    return [Actor(cfg, agent_id) for agent_id in range(int(cfg.env.num_agents))]


def build_critics(cfg: Cfg) -> list[Critic]:
    twin = str(cfg.algo.name) in {"MATD3", "MATD3_SAFE_POC"}
    return [Critic(cfg, twin=twin) for _ in range(int(cfg.env.num_agents))]
