"""Actor / Critic 网络组装入口。

本模块是模型构建的顶层入口，负责将 Adapter、Encoder、Head 三个组件
按 "流水线" 方式组装成完整的 Actor 或 Critic 网络。

组装流水线::

    结构化观测字典 (structured obs dict)
         |
         v
    Adapter (family_adapters.py)
         |  -- 将结构化观测转换为模型特定的输入格式：
         |     MLP: 扁平向量,  Transformer: token 序列,  Graph: 节点特征
         v
    Encoder (encoders/)
         |  -- 特征提取骨干网络
         v
    Head (heads/)
         |  -- 输出投影：动作 (actor) 或 Q 值 (critic)
         v
    输出 (output)

``family`` 配置项决定使用哪种 Adapter + Encoder 组合：
  - ``mlp``         -- 快速、简单、良好的默认选择
  - ``transformer`` -- 基于自注意力的 token 序列处理（局部 + 序列）
  - ``graph``       -- 基于消息传递的 GNN 图神经网络

主要类：
    - ActorNetwork: 组装后的 Actor 网络
    - CriticNetwork: 组装后的 Critic 网络

主要函数：
    - build_actor_network: 构建单个 Actor 网络
    - build_critic_network: 构建单个 Critic 网络
    - validate_and_finalize_model_config: 校验并补全模型配置
"""

from __future__ import annotations

from torch import nn

from models.registry import (
    get_actor_head_cls,
    get_adapter_cls,
    get_critic_head_cls,
    get_encoder_cls,
)

_TWIN_Q_ALGORITHMS = {"MATD3", "MATD3_SAFE_POC"}


class ActorNetwork(nn.Module):
    """组装后的 Actor 网络：adapter -> encoder -> actor head。

    将适配器、编码器和输出头串联为完整的前向推理流水线。
    输入为结构化观测字典，输出为确定性连续动作。

    属性:
        adapter: 观测适配器，将结构化观测转换为模型输入格式
        encoder: 特征编码器，提取高维特征表示
        head: Actor 输出头，将特征映射为动作
    """

    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        """初始化 ActorNetwork。

        参数:
            adapter: 观测适配器模块
            encoder: 特征编码器模块
            head: Actor 输出头模块
        """
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.head = head

    def forward(self, obs):
        """前向传播：观测 -> 适配 -> 编码 -> 动作。

        参数:
            obs: 结构化观测字典，包含 "local" 等键

        返回:
            动作张量，形状取决于 action_dim 和 max_action
        """
        features = self.adapter(obs)          # 将结构化观测转为模型输入格式
        embedding = self.encoder(features)    # 编码为高维特征向量
        return self.head(embedding)           # 映射为确定性动作


class CriticNetwork(nn.Module):
    """组装后的 Critic 网络：adapter -> encoder -> critic head。

    与 ActorNetwork 类似，但额外接受联合动作作为输入。
    Critic 用于评估状态-动作对的 Q 值。

    属性:
        adapter: 观测适配器，将联合观测和动作转换为模型输入格式
        encoder: 特征编码器，提取高维特征表示
        head: Critic 输出头，将特征映射为 Q 值
    """

    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        """初始化 CriticNetwork。

        参数:
            adapter: 观测适配器模块（接受 obs 和 action）
            encoder: 特征编码器模块
            head: Critic 输出头模块（SingleQ 或 TwinQ）
        """
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.head = head

    def _encode(self, obs, action):
        """内部方法：将观测和动作编码为特征向量。

        参数:
            obs: 结构化观测字典
            action: 联合动作张量

        返回:
            编码后的特征向量
        """
        features = self.adapter(obs, action)  # 将联合观测和动作转为模型输入
        return self.encoder(features)         # 编码为高维特征向量

    def forward(self, obs, action):
        """前向传播：返回所有 Q 头的输出。

        参数:
            obs: 结构化观测字典
            action: 联合动作张量

        返回:
            SingleQHead 返回单个 Q 值；TwinQHead 返回 (Q1, Q2) 元组
        """
        return self.head(self._encode(obs, action))

    def Q1(self, obs, action):
        """仅返回第一个 Q 值（用于 MATD3 策略更新时只需 Q1）。

        参数:
            obs: 结构化观测字典
            action: 联合动作张量

        返回:
            第一个 Q 网络的输出值

        注意:
            仅 TwinQHead 支持此方法；SingleQHead 会抛出 AttributeError。
        """
        if not hasattr(self.head, "Q1"):
            raise AttributeError("The configured critic head does not expose Q1.")
        return self.head.Q1(self._encode(obs, action))


def validate_and_finalize_model_config(cfg):
    """校验模型配置的合法性，并自动补齐 critic_head_type 默认值。

    校验规则：
        - algo.name 必须是 MADDPG 或 MATD3
        - model.family 必须是 mlp / transformer / graph
        - MADDPG 要求 single_q，MATD3 要求 twin_q

    参数:
        cfg: 全局配置对象，包含 algo、model 等子配置

    返回:
        补全默认值后的配置对象

    注意:
        此函数会就地修改 cfg.model.critic_head_type（如果为 None）。
    """
    algorithm = cfg.algo.name
    # 校验算法名称
    if algorithm not in {"MADDPG", *_TWIN_Q_ALGORITHMS}:
        raise ValueError(
            f"Unknown algo.name '{algorithm}', available: ['MADDPG', 'MATD3', 'MATD3_SAFE_POC']"
        )

    # 校验模型家族
    if cfg.model.family not in {"mlp", "transformer", "graph"}:
        raise ValueError("model.family must be one of ['mlp', 'transformer', 'graph']")

    # 如果未指定 critic_head_type，根据算法自动设置默认值
    if cfg.model.critic_head_type is None:
        cfg.model.critic_head_type = "single_q" if algorithm == "MADDPG" else "twin_q"

    # 校验 critic_head_type 与算法的一致性
    expected_critic_head = "single_q" if algorithm == "MADDPG" else "twin_q"
    if cfg.model.critic_head_type != expected_critic_head:
        raise ValueError(
            f"algo.name='{algorithm}' requires model.critic_head_type='{expected_critic_head}', "
            f"got '{cfg.model.critic_head_type}'."
        )
    return cfg


def _build_encoder(cfg, input_dim: int):
    """根据 model.family 构建对应的编码器实例（内部辅助函数）。

    根据不同的 family 类型，传入各自特有的超参数：
        - mlp: 仅需通用参数
        - transformer: 额外需要 num_heads 和 num_layers
        - graph: 额外需要 num_layers

    参数:
        cfg: 全局配置对象
        input_dim: 编码器输入维度（由适配器的 output_dim 决定）

    返回:
        构建好的编码器模块实例
    """
    family = cfg.model.family
    encoder_cls = get_encoder_cls(family)
    # 所有编码器共享的参数
    common_kwargs = {
        "input_dim": int(input_dim),
        "hidden_dim": int(cfg.model.hidden_dim),
        "use_orthogonal_init": bool(cfg.model.use_orthogonal_init),
    }
    if family == "mlp":
        return encoder_cls(**common_kwargs)
    if family == "transformer":
        # Transformer 编码器额外需要注意力头数和层数
        return encoder_cls(
            **common_kwargs,
            num_heads=int(cfg.model.transformer_num_heads),
            num_layers=int(cfg.model.transformer_num_layers),
        )
    if family == "graph":
        # 图编码器额外需要消息传递层数
        return encoder_cls(
            **common_kwargs,
            num_layers=int(cfg.model.graph_num_layers),
        )
    raise ValueError(f"Unsupported model.family '{family}'.")


def build_actor_network(cfg, agent_id: int) -> ActorNetwork:
    """按当前 family 组装一个完整的 Actor 网络。

    组装流程: 适配器 -> 编码器 -> Actor 输出头。
    每个智能体拥有独立的 Actor 网络（通过 agent_id 区分）。

    参数:
        cfg: 全局配置对象，需包含 model、algo、runtime 等子配置
        agent_id: 智能体编号，用于从联合观测中提取该智能体的局部观测

    返回:
        组装完成的 ActorNetwork 实例
    """
    validate_and_finalize_model_config(cfg)
    # 从注册表获取当前 family 对应的适配器类和输出头类
    adapter_cls = get_adapter_cls(cfg.model.family, "actor")
    actor_head_cls = get_actor_head_cls(cfg.model.actor_head_type)

    # 依次构建适配器、编码器、输出头
    adapter = adapter_cls(cfg, agent_id)
    encoder = _build_encoder(cfg, adapter.output_dim)  # 编码器输入维度 = 适配器输出维度
    head = actor_head_cls(
        hidden_dim=int(cfg.model.hidden_dim),
        action_dim=int(cfg.runtime.action_dim),
        max_action=float(cfg.model.max_action),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )
    return ActorNetwork(adapter=adapter, encoder=encoder, head=head)


def build_critic_network(cfg) -> CriticNetwork:
    """按当前 family 组装一个完整的 Critic 网络。

    组装流程: 适配器 -> 编码器 -> Critic 输出头。
    Critic 网络接收所有智能体的联合观测和联合动作（中心化训练）。

    参数:
        cfg: 全局配置对象，需包含 model、algo、runtime 等子配置

    返回:
        组装完成的 CriticNetwork 实例

    注意:
        Critic 不需要 agent_id，因为它处理的是全局联合信息。
    """
    validate_and_finalize_model_config(cfg)
    # 从注册表获取当前 family 对应的适配器类和输出头类
    adapter_cls = get_adapter_cls(cfg.model.family, "critic")
    critic_head_cls = get_critic_head_cls(cfg.model.critic_head_type)

    # 依次构建适配器、编码器、输出头
    adapter = adapter_cls(cfg)
    encoder = _build_encoder(cfg, adapter.output_dim)  # 编码器输入维度 = 适配器输出维度
    head = critic_head_cls(
        hidden_dim=int(cfg.model.hidden_dim),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )
    return CriticNetwork(adapter=adapter, encoder=encoder, head=head)
