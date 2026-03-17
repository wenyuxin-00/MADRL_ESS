"""Output head implementations -- final projections for actor and critic.
输出头实现 -- Actor 和 Critic 的最终投影层。

Available heads / 已实现输出头:
    - DeterministicContinuousActorHead  -- tanh-bounded continuous actions
    - SingleQHead                       -- single Q-value (for MADDPG)
    - TwinQHead                         -- twin Q-values (for MATD3/TD3)

How to add a new head / 如何添加新输出头:
    1. Create in ``models/heads/your_head.py`` as an ``nn.Module``
       - Input: ``(batch, hidden_dim)`` tensor from encoder
       - Output: action tensor (actor) or Q-value scalar (critic)
    2. Register in ``models/registry.py``::

           ACTOR_HEAD_REGISTRY["your_head"] = YourActorHead
           # or
           CRITIC_HEAD_REGISTRY["your_head"] = YourCriticHead
"""

from models.heads.actor_head import DeterministicContinuousActorHead
from models.heads.critic_head import SingleQHead, TwinQHead

__all__ = ["DeterministicContinuousActorHead", "SingleQHead", "TwinQHead"]
