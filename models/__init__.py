"""Neural network model assembly.
神经网络模型装配。

Architecture: Adapter -> Encoder -> Head
模型架构: 适配器 -> 编码器 -> 输出头

    - **Adapter** (family_adapters.py): converts structured observations to
      model-specific input format (flat vector for MLP, token sequence for
      Transformer, node features for Graph)
    - **Encoder** (encoders/): feature extraction backbone (MLPEncoder,
      TransformerEncoder, GraphEncoder)
    - **Head** (heads/): output projection (actor head -> actions, critic head -> Q-values)

How to add a new model family / 如何添加新模型家族:
    1. Create an encoder in ``models/encoders/``
    2. Create actor/critic adapters in ``models/family_adapters.py``
    3. Register all three in ``models/registry.py``
    4. Use in config: ``cfg.model.family = "your_family"``
"""

from models.assembly import (
    ActorNetwork,
    CriticNetwork,
    build_actor_network,
    build_critic_network,
    validate_and_finalize_model_config,
)

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "build_actor_network",
    "build_critic_network",
    "validate_and_finalize_model_config",
]
