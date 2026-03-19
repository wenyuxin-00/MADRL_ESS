"""电网数据类型定义。

定义电网潮流求解后需要用到的结构化数据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridStepResult:
    """一次 pandapower 潮流求解的结果容器。"""

    converged: bool                 # 潮流是否收敛
    vm_pu: np.ndarray               # 各母线电压幅值（标幺值）
    va_degree: np.ndarray           # 各母线电压相角（度）
    line_loading_pct: np.ndarray    # 各线路负载率（百分比）
    p_mw_from: np.ndarray           # 各线路首端有功功率（MW）
    agent_vm_pu: np.ndarray         # 各智能体所在母线的电压
    v_violation: np.ndarray         # 各智能体的电压越限量
    l_violation: float              # 最严重的线路过载量
    n_buses: int                    # 母线总数
    n_lines: int                    # 线路总数
