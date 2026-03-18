"""Classic single-agent DRL baseline controller -- implementation template.
经典单智能体 DRL baseline 控制器 -- 实现模板。

This is an extension point for single-agent RL baselines (PPO, DQN, SAC, etc.)
that treat the multi-agent system as one joint agent.
这里预留 compare notebook 入口，后续可接 PPO / DQN / SAC 等基线。

How to implement / 如何实现:
    1. Train a single-agent policy (e.g., using stable-baselines3)
    2. In ``__init__``, load the trained model
    3. In ``reset()``, reset any RNN hidden state if applicable
    4. In ``act(obs)``:
       - Flatten the structured observation into a single vector
       - Run inference through your model
       - Split the joint action into per-agent actions
       - Return ``[np.array([a_i]) for i in range(n_agents)]``

See ``controllers/madrl_controller.py`` for a similar agent-wrapping pattern.
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class ClassicDRLController(BaseController):
    """Placeholder for single-agent DRL baselines (PPO/DQN/SAC).
    单智能体 DRL baseline 占位控制器。
    """

    def reset(self) -> None:
        """No internal state yet. Override when implementing."""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        raise NotImplementedError(
            "ClassicDRLController is a placeholder. See the module docstring for implementation guidance."
        )
