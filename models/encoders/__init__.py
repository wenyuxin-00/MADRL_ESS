"""Encoder implementations -- feature extraction backbones.
编码器实现 -- 特征提取骨干网络。

Available encoders / 已实现编码器:
    - MLPEncoder          -- two-layer MLP (default, fast)
    - TransformerEncoder  -- multi-head self-attention encoder
    - GraphEncoder        -- message-passing GNN encoder

How to add a new encoder / 如何添加新编码器:
    1. Create ``models/encoders/your_encoder.py`` as an ``nn.Module``
       - Input: ``(batch, input_dim)`` tensor
       - Output: ``(batch, hidden_dim)`` tensor
    2. Register in ``models/registry.py``::

           ENCODER_REGISTRY["your_family"] = YourEncoder

    3. Import in this ``__init__.py``
"""

from models.encoders.graph_encoder import GraphEncoder
from models.encoders.mlp_encoder import MLPEncoder
from models.encoders.transformer_encoder import TransformerEncoder

__all__ = ["GraphEncoder", "MLPEncoder", "TransformerEncoder"]
