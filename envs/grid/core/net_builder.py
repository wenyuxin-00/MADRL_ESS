"""pandapower 电网构建器。

根据 SimBench 编码或自定义拓扑构建 pandapower 网络对象，
并提供母线注入功率设置和时间序列配置等工具函数。

主要函数:
    build_simbench_net      -- 构建并缓存 SimBench 网络
    apply_bus_injections    -- 设置母线净注入功率
    set_timeseries_profiles -- 设置单步的负荷与光伏时序数据
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandapower as pp

# 模块级缓存: SimBench 编码 -> 模板网络对象
# 每个环境获取独立的深拷贝，因此可安全修改
_NET_CACHE: dict[str, "pp.pandapowerNet"] = {}


def build_simbench_net(sb_code: str) -> "pp.pandapowerNet":
    """根据 SimBench 编码构建 pandapower 网络（带缓存）。

    模板网络仅在首次调用时加载，后续调用通过深拷贝获取独立副本，
    避免重复加载 SimBench 数据。

    参数:
        sb_code: SimBench 网络标识符，如 ``"1-LV-rural1--0-sw"``

    返回:
        pandapowerNet: 独立的 pandapower 网络对象，可安全修改
    """
    if sb_code not in _NET_CACHE:
        import simbench as sb

        # 首次加载并缓存模板网络
        _NET_CACHE[sb_code] = sb.get_simbench_net(sb_code)
    # 深拷贝保证每个调用者获得独立副本
    return copy.deepcopy(_NET_CACHE[sb_code])


def apply_bus_injections(
    net: "pp.pandapowerNet",
    bus_id_to_p_kw: dict[int, float],
    bus_id_to_q_kvar: dict[int, float] | None = None,
) -> None:
    """设置一组母线的净注入功率（原地修改网络）。

    对于 bus_id_to_p_kw 中的每个母线，查找该母线上的第一个
    静态发电机（sgen）或负荷元件并调整其有功功率。
    正的 p_kw 表示发电（向电网输出）；负的表示用电（从电网吸收）。
    若母线上不存在 sgen，则新建一个负荷元件。

    参数:
        net: 要修改的 pandapower 网络对象
        bus_id_to_p_kw: 母线索引 -> 净注入功率（kW）的映射。
                        正值 = 发电（如光伏盈余或电池放电），
                        负值 = 用电（如净负荷或电池充电）
        bus_id_to_q_kvar: 可选的无功功率注入（kVAr）映射，
                          省略时所有母线默认为零
    """
    import pandapower as pp

    if bus_id_to_q_kvar is None:
        bus_id_to_q_kvar = {}

    for bus_id, p_kw in bus_id_to_p_kw.items():
        q_kvar = bus_id_to_q_kvar.get(bus_id, 0.0)
        # 单位转换: kW -> MW, kVAr -> MVAr
        p_mw = p_kw / 1000.0
        q_mvar = q_kvar / 1000.0

        # 查找该母线上已有的静态发电机
        sgen_mask = net.sgen["bus"] == bus_id
        if sgen_mask.any():
            idx = net.sgen.index[sgen_mask][0]
            # sgen 功率不能为负，取非负部分
            net.sgen.at[idx, "p_mw"] = max(0.0, p_mw)
            net.sgen.at[idx, "q_mvar"] = q_mvar
            # 剩余的负功率部分作为额外负荷处理
            load_mask = net.load["bus"] == bus_id
            extra_load_mw = max(0.0, -p_mw)
            if load_mask.any():
                # SET（非累加）负荷值，避免每步累积导致潮流发散
                net.load.at[net.load.index[load_mask][0], "p_mw"] = extra_load_mw
            elif extra_load_mw > 0.0:
                # 该母线无负荷元件，创建新的负荷
                pp.create_load(net, bus=bus_id, p_mw=extra_load_mw, q_mvar=0.0)
        else:
            # 无 sgen -- 使用负荷元件（符号反转：正负荷 = 用电）
            load_mask = net.load["bus"] == bus_id
            if load_mask.any():
                idx = net.load.index[load_mask][0]
                net.load.at[idx, "p_mw"] = -p_mw     # 正负荷 = 用电
                net.load.at[idx, "q_mvar"] = -q_mvar
            else:
                # 新建负荷元件（负 p_mw 表示发电）
                pp.create_load(net, bus=bus_id, p_mw=-p_mw, q_mvar=-q_mvar)


def set_timeseries_profiles(
    net: "pp.pandapowerNet",
    bus_load_kw: dict[int, float],
    bus_pv_kw: dict[int, float],
) -> None:
    """将单步的背景负荷和光伏数据写入网络。

    便捷包装函数：在调用 runpp 之前，将时序数据中当前步的
    负荷和光伏值分别设置到 load.p_mw 和 sgen.p_mw 字段。

    参数:
        net: pandapower 网络对象（原地修改）
        bus_load_kw: 母线索引 -> 背景负荷需求（kW，正值 = 用电）
        bus_pv_kw: 母线索引 -> 光伏发电量（kW，正值 = 发电）
    """
    # 设置各母线的负荷功率
    for bus_id, p_kw in bus_load_kw.items():
        mask = net.load["bus"] == bus_id
        if mask.any():
            net.load.at[net.load.index[mask][0], "p_mw"] = p_kw / 1000.0

    # 设置各母线的光伏出力
    for bus_id, p_kw in bus_pv_kw.items():
        mask = net.sgen["bus"] == bus_id
        if mask.any():
            net.sgen.at[net.sgen.index[mask][0], "p_mw"] = p_kw / 1000.0
