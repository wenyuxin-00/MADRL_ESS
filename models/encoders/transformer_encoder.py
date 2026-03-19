"""Transformer 编码器模块。

使用标准 Transformer 编码器对 token 序列进行自注意力编码，
最终通过均值池化将变长序列压缩为固定维度的特征向量。

主要类：
    - TransformerEncoder: 基于自注意力的序列编码器，输出均值池化后的特征
"""

from __future__ import annotations

from torch import nn

from models.utils import orthogonal_init


class TransformerEncoder(nn.Module):
    """基于均值池化的 Transformer 序列编码器。

    网络结构: tokens -> 线性投影 -> Transformer 编码器 -> 均值池化 -> output
    先将 token 投影到 hidden_dim 维空间，再通过多层自注意力编码，
    最后在序列维度上取均值得到固定维度的特征向量。

    属性:
        input_proj: 输入投影层 (input_dim -> hidden_dim)
        encoder: PyTorch 标准 TransformerEncoder（多层堆叠）
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, use_orthogonal_init: bool):
        """初始化 Transformer 编码器。

        参数:
            input_dim: 每个 token 的输入维度
            hidden_dim: 隐藏层维度（即 Transformer 的 d_model）
            num_heads: 多头注意力的头数
            num_layers: Transformer 编码器的层数
            use_orthogonal_init: 是否对输入投影层使用正交初始化
        """
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        # 构建单层 Transformer 编码器层（无 dropout，适合强化学习场景）
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            batch_first=True,
            dropout=0.0,
        )
        # 堆叠多层构成完整编码器
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        if use_orthogonal_init:
            orthogonal_init(self.input_proj)

    def forward(self, payload: dict):
        """前向传播：投影 -> 自注意力编码 -> 均值池化。

        参数:
            payload: 适配器输出的字典，payload["tokens"] 形状为 (batch, num_tokens, input_dim)

        返回:
            均值池化后的特征向量，形状为 (batch, hidden_dim)
        """
        x = self.input_proj(payload["tokens"])  # 线性投影到 hidden_dim
        x = self.encoder(x)                     # 多层自注意力编码
        return x.mean(dim=1)                    # 在 token 维度上均值池化
