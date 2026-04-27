from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.utils.storage_profit import compute_storage_profit_components


# 作用：描述 reward 分项在训练摘要和 notebook 可视化中的展示信息。
@dataclass(frozen=True)
class ComponentMeta:
    key: str
    label: str
    color: str
    sign: int


NON_NEGATIVE_REWARD_FIELDS = (
    "action_boundary_penalty_weight",
    "soc_boundary_regularization_weight",
    "throughput_bonus_eur_per_kwh_max",
    "soc_boundary_epsilon",
    "soc_boundary_margin",
    "w_voltage_pen",
    "w_line_pen",
    "w_trafo_pen",
)
COMPONENT_META = (
    ComponentMeta("madrl_r_inc", "+ madrl_r_inc", "blue", 1),
    ComponentMeta("madrl_r_action_penalty", "- madrl_r_action_penalty", "amber", -1),
    ComponentMeta("madrl_r_soc_regularization", "- madrl_r_soc_regularization", "green", -1),
    ComponentMeta("madrl_r_throughput_bonus", "+ madrl_r_throughput_bonus", "teal", 1),
    ComponentMeta("madrl_r_safe_v", "- madrl_r_safe_v", "red", -1),
    ComponentMeta("madrl_r_safe_line", "- madrl_r_safe_line", "purple", -1),
    ComponentMeta("madrl_r_safe_trafo", "- madrl_r_safe_trafo", "maroon", -1),
    ComponentMeta("madrl_r_safe_total", "- madrl_r_safe_total", "rose", -1),
    ComponentMeta("madrl_r_total_internal", "+ madrl_r_total_internal", "slate", 1),
)


# 作用：计算 MADRL 每一步的 storage profit、边界惩罚、探索 bonus 和电网安全惩罚。
class NormalReward:
    # 作用：读取并校验 reward 配置，旧字段和缺失字段在边界处直接失败。
    def __init__(self, cfg: object) -> None:
        reward_cfg = cfg.reward
        if str(reward_cfg.storage_objective_mode) != "max_storage_profit":
            raise ValueError(f"storage_objective_mode must be 'max_storage_profit', got {reward_cfg.storage_objective_mode!r}.")
        if str(reward_cfg.storage_price_mode) != "real_time_price":
            raise ValueError(f"storage_price_mode must be 'real_time_price', got {reward_cfg.storage_price_mode!r}.")
        if not np.isclose(float(reward_cfg.storage_profit_weight), 1.0):
            raise ValueError(f"storage_profit_weight must be 1.0, got {reward_cfg.storage_profit_weight!r}.")

        for field_name in NON_NEGATIVE_REWARD_FIELDS:
            value = float(getattr(reward_cfg, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative, got {value}.")
            setattr(self, field_name, value)

    # 作用：返回 reward 分项的稳定顺序和展示元数据。
    @property
    def component_meta(self) -> list[ComponentMeta]:
        return list(COMPONENT_META)

    # 作用：按照训练进度把 throughput bonus 从早期最大值线性退火到 0。
    def _throughput_bonus_weight(self, env_state: dict) -> float:
        progress = float(env_state["training_progress"])
        progress = float(np.clip(progress, 0.0, 1.0))
        return self.throughput_bonus_eur_per_kwh_max * float(np.clip((0.80 - progress) / 0.60, 0.0, 1.0))

    # 作用：基于当前环境状态计算每个 agent 的 MADRL reward 和分项明细。
    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        battery_power_t = np.asarray(env_state["battery_power_t"], dtype=np.float32)
        soc_t = np.asarray(env_state["soc_t"], dtype=np.float32)
        v_violation = np.asarray(env_state["v_violation"], dtype=np.float32)
        price_t, dt = float(env_state["storage_price_t"]), float(env_state["dt"])
        soc_min, soc_max = float(env_state["soc_min"]), float(env_state["soc_max"])
        psi_v_raw, psi_line_raw, psi_trafo_raw = float(env_state["psi_v_raw"]), float(env_state["psi_line_raw"]), float(env_state["psi_trafo_raw"])
        throughput_bonus_weight_t = self._throughput_bonus_weight(env_state)

        if battery_power_t.ndim != 1 or soc_t.shape != battery_power_t.shape or v_violation.shape != battery_power_t.shape:
            raise ValueError(f"Reward arrays must be 1D with matching shapes, got battery_power_t={battery_power_t.shape}, soc_t={soc_t.shape}, v_violation={v_violation.shape}.")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if not 0.0 <= soc_min <= soc_max <= 1.0:
            raise ValueError(f"soc bounds must satisfy 0 <= soc_min <= soc_max <= 1, got [{soc_min}, {soc_max}].")

        soc_lower_soft = soc_min + self.soc_boundary_margin
        soc_upper_soft = soc_max - self.soc_boundary_margin
        if soc_lower_soft > soc_upper_soft:
            raise ValueError(f"soc_boundary_margin makes soft bounds invalid: [{soc_lower_soft}, {soc_upper_soft}].")

        storage_components = compute_storage_profit_components(battery_power_kw=battery_power_t, price_eur_per_kwh=price_t, dt_hours=dt)
        madrl_r_inc = storage_components["storage_profit_eur"].astype(np.float32)
        # 电池按实时价格充放电的增量利润：充电是成本，放电是收益。
        pushes_lower = (battery_power_t < 0.0) & (soc_t <= soc_min + self.soc_boundary_epsilon)
        pushes_upper = (battery_power_t > 0.0) & (soc_t >= soc_max - self.soc_boundary_epsilon)
        boundary_push = np.logical_or(pushes_lower, pushes_upper)
        madrl_r_action_penalty = (self.action_boundary_penalty_weight * np.abs(battery_power_t) * boundary_push.astype(np.float32)).astype(np.float32)
        # SOC 已在硬边界附近时，如果动作继续把电池往边界外推，就扣这项。

        below = np.maximum(0.0, soc_lower_soft - soc_t).astype(np.float32)
        above = np.maximum(0.0, soc_t - soc_upper_soft).astype(np.float32)
        madrl_r_soc_regularization = (self.soc_boundary_regularization_weight * (below * below + above * above)).astype(np.float32)
        # SOC 落在软安全带外的二次惩罚；越偏离软边界，惩罚增长越快。

        throughput_kwh = (np.abs(battery_power_t) * np.float32(dt)).astype(np.float32)
        madrl_r_throughput_bonus = (throughput_bonus_weight_t * throughput_kwh).astype(np.float32)
        # 训练早期鼓励电池产生有效吞吐，随 training_progress 线性退火到 0。

        n_agents = int(battery_power_t.shape[0])
        v_sum = float(np.sum(v_violation))
        # 非 agent 节点触发电压违约时，没有局部归因信号，只能按 agent 均摊。
        v_weights = (v_violation / v_sum).astype(np.float32) if v_sum > 0.0 else np.full((n_agents,), 1.0 / max(n_agents, 1), dtype=np.float32) if psi_v_raw > 0.0 else np.zeros((n_agents,), dtype=np.float32)
        madrl_r_safe_v = (n_agents * self.w_voltage_pen * psi_v_raw * v_weights).astype(np.float32)
        # 电压越限惩罚按 agent 本地违约占比分摊；没有本地归因时均摊。
        madrl_r_safe_line = np.full((n_agents,), self.w_line_pen * psi_line_raw, dtype=np.float32)
        # 线路越限是全局安全成本，所有 agent 承担同一份惩罚。
        madrl_r_safe_trafo = np.full((n_agents,), self.w_trafo_pen * psi_trafo_raw, dtype=np.float32)
        # 变压器越限是全局安全成本，所有 agent 承担同一份惩罚。
        madrl_r_safe_total = (madrl_r_safe_v + madrl_r_safe_line + madrl_r_safe_trafo).astype(np.float32)
        # 安全惩罚合计，会从最终 MADRL reward 中扣除。
        madrl_r_total_internal = (madrl_r_inc - madrl_r_action_penalty - madrl_r_soc_regularization + madrl_r_throughput_bonus - madrl_r_safe_total).astype(np.float32)
        # 最终每个 agent 的训练 reward：储能收益 - 边界/安全惩罚 + 早期探索 bonus。

        components = {
            "madrl_r_inc": madrl_r_inc,
            "madrl_r_action_penalty": madrl_r_action_penalty,
            "madrl_r_soc_regularization": madrl_r_soc_regularization,
            "madrl_r_throughput_bonus": madrl_r_throughput_bonus,
            "madrl_r_safe_v": madrl_r_safe_v,
            "madrl_r_safe_line": madrl_r_safe_line,
            "madrl_r_safe_trafo": madrl_r_safe_trafo,
            "madrl_r_safe_total": madrl_r_safe_total,
            "madrl_r_total_internal": madrl_r_total_internal,
        }
        return madrl_r_total_internal, components
