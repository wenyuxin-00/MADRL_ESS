"""MPC (Model Predictive Control) controller -- implementation template.
MPC（模型预测控制）控制器 -- 实现模板。

This is an extension point for optimization-based baselines. The comparison
notebook already wires this controller in; you only need to fill in the logic.
本轮重构先把 compare 入口和接口留好，具体 MPC 逻辑后续再接入。

How to implement / 如何实现:
    1. Add your MPC solver dependency (e.g., cvxpy, casadi) to requirements.txt
    2. In ``__init__``, configure the prediction horizon and solver parameters
    3. In ``reset()``, initialize any episode-level state
    4. In ``act(obs)``:
       - Extract price forecast from ``obs["price_seq"]`` (shape: ``(n_agents, horizon, 1)``)
       - Extract current SoC from ``obs["local"][:, soc_index]``
       - Formulate and solve the optimization problem over the horizon
       - Return the first-step actions as ``[np.array([a_i]) for i in range(n_agents)]``

See ``controllers/zero_controller.py`` for the simplest working example.
See ``controllers/madrl_controller.py`` for an agent-wrapping example.
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class MPCController(BaseController):
    """Placeholder for Model Predictive Control baseline.
    MPC baseline 占位控制器。
    """

    def reset(self) -> None:
        """No internal state yet. Override when implementing."""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        raise NotImplementedError(
            "MPCController is a placeholder. See the module docstring for implementation guidance."
        )
