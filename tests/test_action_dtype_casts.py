from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from controllers.madrl.base_agent import BaseAgent
from scripts.train import TrainRunner


class _DummyActionAgent(BaseAgent):
    def __init__(self, device: torch.device) -> None:
        self.cfg = SimpleNamespace(runtime=SimpleNamespace(device=device))
        self.device = device

    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        del noise_std
        batch = int(obs_t["local"].shape[0])
        return torch.ones((batch, 1), dtype=torch.bfloat16, device=self.device)

    def train(self, replay_buffer, agent_n: list) -> None:
        raise NotImplementedError

    def train_on_batch(self, batch: dict, agent_n: list, shared_ctx: dict | None = None) -> None:
        raise NotImplementedError


def test_base_agent_choose_action_casts_bfloat16_before_numpy():
    agent = _DummyActionAgent(torch.device("cpu"))
    obs = {"local": np.zeros((2, 3), dtype=np.float32)}

    action = agent.choose_action(obs, noise_std=0.0)

    assert action.dtype == np.float32
    assert action.shape == (1,)


def test_train_runner_select_action_batch_casts_bfloat16_before_numpy():
    runner = TrainRunner.__new__(TrainRunner)
    runner.cfg = SimpleNamespace(
        runtime=SimpleNamespace(device=torch.device("cpu")),
        env=SimpleNamespace(num_agents=2),
    )
    runner.noise_std = 0.0
    runner.agent_n = [
        _DummyActionAgent(torch.device("cpu")),
        _DummyActionAgent(torch.device("cpu")),
    ]

    obs = {"local": np.zeros((4, 2, 3), dtype=np.float32)}
    action_batch = runner.select_action_batch(obs)

    assert action_batch.dtype == np.float32
    assert action_batch.shape == (4, 2, 1)


def test_bfloat16_predictions_use_float32_loss_for_backward():
    prediction = torch.randn((8, 1), dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn((8, 1), dtype=torch.float32)

    loss = F.mse_loss(prediction.float(), target.float())
    loss.backward()

    assert loss.dtype == torch.float32
    assert prediction.grad is not None
