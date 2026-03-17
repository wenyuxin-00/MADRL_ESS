"""Unified controller interface for evaluation and comparison.
控制器统一接口，用于评估与对比。

All controllers implement ``BaseController`` with two methods:
    - reset()  -- called at episode start
    - act(obs) -- returns actions given structured observations

Available controllers / 已实现控制器:
    - ZeroController        -- always outputs zero actions (baseline)
    - MADRLController       -- wraps trained MADRL agents
    - MPCController         -- placeholder for Model Predictive Control
    - ClassicDRLController  -- placeholder for single-agent RL baselines (PPO/DQN/SAC)

See ``controllers/mpc_controller.py`` and ``controllers/classic_drl_controller.py``
for implementation templates.
"""

from controllers.base import BaseController
from controllers.classic_drl_controller import ClassicDRLController
from controllers.madrl_controller import MADRLController
from controllers.mpc_controller import MPCController
from controllers.zero_controller import ZeroController

__all__ = [
    "BaseController",
    "ClassicDRLController",
    "MADRLController",
    "MPCController",
    "ZeroController",
]
