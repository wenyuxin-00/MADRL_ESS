from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class ComponentMeta:
    """奖励分量的元数据，驱动 history 动态初始化和画图自适应。

    Attributes:
        key:   对应 info dict 的键名，e.g. "r_inc"
        label: 子图标题，e.g. "+ r_inc (incremental cost)"
        color: 子图颜色，e.g. "green"
        sign:  +1 或 -1（该分量在 total 中的贡献符号，用于正确累加历史）
               +1 → total 中是加号；-1 → total 中是减号
    """
    key: str
    label: str
    color: str
    sign: int  # +1 or -1


class RewardFn(ABC):
    """所有奖励函数的抽象基类。

    子类需实现两个接口：
    1. compute()         — 计算奖励，返回 (total, components) 元组
    2. component_meta   — 声明各分量的元数据（驱动动态画图和 history）

    env_state 键说明：
        e_bat_req  (N,)  — 请求功率（可能越限）
        e_bat      (N,)  — 执行功率（可行域投影后）
        soc_t      (N,)  — 本步开始时的 SoC
        soc_next   (N,)  — 本步结束后的 SoC
        e_t        (N,)  — 本步开始时的储能量 = soc_t * c_bat
        e_next     (N,)  — 本步结束后的储能量 = soc_next * c_bat
        price_t    float — 当前电价
        load_t     (N,)  — 当前负荷
        mu_t       float — 未来 K 步平均电价（当前步）
        mu_next    float — 未来 K 步平均电价（下一步）
        gamma      float — 折扣因子（PBRS 使用）
    """

    @abstractmethod
    def compute(self, env_state: dict) -> tuple:
        """
        计算奖励并返回各分量。

        Returns:
            (total_reward, components)
            - total_reward: np.ndarray, shape (N,), dtype float32
            - components:   dict[str, np.ndarray], 每个值 shape (N,) float32
                            键名与 component_meta 中的 key 一一对应
                            存储的是"绝对值"——符号由 ComponentMeta.sign 管理
        """
        ...

    @property
    @abstractmethod
    def component_meta(self) -> list:
        """
        返回各分量的元数据列表（list[ComponentMeta]）。
        顺序与 compute() 返回的 components dict 中的键顺序一致。
        该列表驱动：
          - hems_env.py step() 中 info 的动态构建
          - Runner 中 history dict 的动态初始化
          - Runner 中每步数据的动态存储
          - plot_reward_decomposition 的子图数量、颜色、标题
        """
        ...
