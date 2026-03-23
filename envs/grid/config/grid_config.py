"""鐢电綉鎷撴墤涓庣害鏉熼厤缃€?

瀹氫箟鐢电綉 SimBench 缂栫爜銆佹眰瑙ｅ櫒绫诲瀷銆佺數鍘嬬害鏉熺瓑鍙傛暟锛?
浠ュ強鏅鸿兘浣撳湪鐢电綉涓殑閮ㄧ讲浣嶇疆鍜岃澶囧弬鏁般€?

涓昏绫?
    AgentDeployment -- 鍗曚釜鏅鸿兘浣撶殑鐗╃悊閮ㄧ讲鍙傛暟

涓昏鍑芥暟:
    build_agent_deployments -- 浠庡疄楠岄厤缃В鏋愭櫤鑳戒綋閮ㄧ讲鍒楄〃
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class AgentDeployment:
    """鍗曚釜寮哄寲瀛︿範鏅鸿兘浣撶殑鐗╃悊閮ㄧ讲浣嶇疆鍜岃澶囧弬鏁般€?

    灞炴€?
        bus_id: 璇ユ櫤鑳戒綋鐢垫睜杩炴帴鐨?pandapower 姣嶇嚎绱㈠紩
        battery_capacity_kwh: 鍙敤鐢垫睜鑳介噺瀹归噺锛坘Wh锛?
        battery_power_kw: 鏈€澶у厖鏀剧數鍔熺巼锛坘W锛屽绉板紡锛?
        init_soc: 姣忎釜 episode 寮€濮嬫椂鐨勫垵濮嬭嵎鐢电姸鎬侊紙0--1 涔嬮棿鐨勬瘮渚嬪€硷級
        soc_min: 鍏佽鐨勬渶浣?SoC锛堟瘮渚嬪€硷級
        soc_max: 鍏佽鐨勬渶楂?SoC锛堟瘮渚嬪€硷級
        efficiency: 鍏呮斁鐢垫晥鐜囷紙鎸夊崐绋嬭绠楋紝鍗抽潪 sqrt 绾﹀畾锛?
                    涓?GridEnv 淇濇寔涓€鑷达級
    """

    bus_id: int                         # 姣嶇嚎绱㈠紩
    battery_capacity_kwh: float         # 鐢垫睜瀹归噺锛坘Wh锛?
    battery_power_kw: float             # 鏈€澶у厖鏀剧數鍔熺巼锛坘W锛?
    init_soc: float = 0.5              # 鍒濆 SoC
    soc_min: float = 0.05             # 鏈€浣?SoC
    soc_max: float = 0.95             # 鏈€楂?SoC
    efficiency: float = 0.95          # 鍏呮斁鐢垫晥鐜?


def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    """浠庡疄楠岄厤缃В鏋愭櫤鑳戒綋閮ㄧ讲鍒楄〃銆?

    鑻?cfg.grid.agent_bus_ids 宸茶缃紝鍒欎娇鐢ㄦ寚瀹氱殑姣嶇嚎 ID锛?
    璁惧鍙傛暟鍙栬嚜鏁版嵁闆嗗厓鏁版嵁锛堝鍙敤锛夋垨 cfg.env 涓殑榛樿鍊笺€?
    鑻ユ湭閰嶇疆鑷畾涔?ID锛屽垯鍥為€€鍒?"1-LV-rural1--0-sw" 鐨?
    Phase-1 鍥哄畾鎷撴墤棰勮銆?

    鍙傛暟:
        cfg: 瀹屾暣鐨?ExperimentConfig 瀹為獙閰嶇疆瀹炰緥

    杩斿洖:
        list[AgentDeployment]: 姣忎釜鏅鸿兘浣撲竴涓潯鐩紝
                               鎬绘暟涓?cfg.env.num_agents
    """
    from envs.grid.topology.rural1_fixed import RURAL1_AGENT_DEPLOYMENTS

    n = int(cfg.env.num_agents)
    bus_ids: list[int] = list(cfg.grid.agent_bus_ids)

    if not bus_ids:
        # 閰嶇疆涓湭鎸囧畾姣嶇嚎 ID -- 浣跨敤 Phase-1 榛樿閮ㄧ讲
        return RURAL1_AGENT_DEPLOYMENTS[:n]

    # 浠庨厤缃拰鐜榛樿鍊兼瀯寤洪儴缃插垪琛?
    c_bat = float(cfg.env.battery_capacity)
    p_max = float(cfg.env.max_charge_rate)
    init_soc = float(cfg.env.init_soc)
    soc_min = float(cfg.env.soc_min)
    soc_max = float(cfg.env.soc_max)
    eff = float(cfg.env.efficiency)

    deployments = [
        AgentDeployment(
            bus_id=int(bus_ids[i]),
            battery_capacity_kwh=c_bat,
            battery_power_kw=p_max,
            init_soc=init_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            efficiency=eff,
        )
        for i in range(n)
    ]
    return deployments

