"""全局安全势函数 + 灵敏度近似差分 credit 的 Grid 复合奖励。

奖励分配策略：
- 基础经济项（套利、SoC、PBRS 等）保持 per-agent。
- 全局安全惩罚（电压 / 线路 / 变压器）基于全网 sum-of-squared-excess，所有 agent 共享。
- 灵敏度 credit 是 per-agent 的一阶近似差分奖励，估计每个 agent 对减少全局安全代价的边际贡献。
  注意：这是 first-order approximate difference reward，不是精确 difference reward。
"""

from __future__ import annotations

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn
from envs.rewards.composite import CompositeReward


class GridCompositeReward(RewardFn):
    """全局安全势函数 + 灵敏度 credit 的 Grid 复合奖励函数。

    总奖励 = base_total
           - w_global_safe * w_v_pen   * psi_v_raw       [全 agent 相同]
           - w_global_safe * w_line_pen * psi_line_raw    [全 agent 相同]
           - w_global_safe * w_trafo_pen* psi_trafo_raw   [全 agent 相同]
           + r_sens_credit                                 [per-agent]
    """

    def __init__(self, cfg: object) -> None:
        self._base = CompositeReward(cfg)
        self.w_v_pen = float(cfg.grid.w_v_pen)
        self.w_line_pen = float(cfg.grid.w_line_pen)
        self.w_trafo_pen = float(cfg.grid.w_trafo_pen)
        self.w_global_safe = float(getattr(cfg.reward, "w_global_safe", 1.0))
        self.w_sens_credit = float(getattr(cfg.reward, "w_sens_credit", 0.2))
        self.sens_credit_scale = float(getattr(cfg.reward, "sens_credit_scale", 0.05))

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            *self._base.component_meta,
            ComponentMeta(
                "r_safe_v_global",
                "- r_safe_v (global voltage penalty)",
                "red",
                -1,
            ),
            ComponentMeta(
                "r_safe_line_global",
                "- r_safe_line (global line penalty)",
                "brown",
                -1,
            ),
            ComponentMeta(
                "r_safe_trafo_global",
                "- r_safe_trafo (global trafo penalty)",
                "maroon",
                -1,
            ),
            ComponentMeta(
                "r_sens_credit",
                "+ r_sens_credit (sensitivity credit)",
                "darkgreen",
                +1,
            ),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        base_total, components = self._base.compute(env_state)
        n_agents = int(base_total.shape[0])

        # --- 权重（允许 env_state 覆盖，便于调试） ---
        w_gs = float(env_state.get("w_global_safe", self.w_global_safe))
        w_v = float(env_state.get("w_v_pen", self.w_v_pen))
        w_line = float(env_state.get("w_line_pen", self.w_line_pen))
        w_trafo = float(env_state.get("w_trafo_pen", self.w_trafo_pen))

        # --- 全局安全势函数（sum-of-squared-excess，所有 agent 共享） ---
        psi_v = float(env_state.get("psi_v_raw", 0.0))
        psi_line = float(env_state.get("psi_line_raw", 0.0))
        psi_trafo = float(env_state.get("psi_trafo_raw", 0.0))

        r_safe_v = np.full(n_agents, w_gs * w_v * psi_v, dtype=np.float32)
        r_safe_line = np.full(n_agents, w_gs * w_line * psi_line, dtype=np.float32)
        r_safe_trafo = np.full(n_agents, w_gs * w_trafo * psi_trafo, dtype=np.float32)

        # --- 灵敏度近似差分 credit（per-agent，一阶近似） ---
        r_sens = self._compute_sensitivity_credit(env_state, n_agents, w_v, w_line, w_trafo)

        total = (base_total - r_safe_v - r_safe_line - r_safe_trafo + r_sens).astype(
            np.float32
        )
        components["r_safe_v_global"] = r_safe_v
        components["r_safe_line_global"] = r_safe_line
        components["r_safe_trafo_global"] = r_safe_trafo
        components["r_sens_credit"] = r_sens
        return total, components

    # ------------------------------------------------------------------
    # 内部：灵敏度 credit 计算
    # ------------------------------------------------------------------

    def _compute_sensitivity_credit(
        self,
        env_state: dict,
        n_agents: int,
        w_v: float,
        w_line: float,
        w_trafo: float,
    ) -> np.ndarray:
        """计算灵敏度近似差分 credit。

        基于缓存的有限差分灵敏度快照，估计每个 agent 当前动作对减少
        全局安全势函数的边际贡献。

        注意：这是 first-order approximate difference reward，不是精确值。
        """
        r_sens = np.zeros(n_agents, dtype=np.float32)
        sensitivity = env_state.get("sensitivity_snapshot")
        if sensitivity is None:
            return r_sens

        bus_v_excess = np.asarray(
            env_state.get("bus_v_excess", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        )
        bus_v_signed = np.asarray(
            env_state.get("bus_v_signed_indicator", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        )
        line_excess = np.asarray(
            env_state.get("line_excess", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        )
        trafo_excess = np.asarray(
            env_state.get("trafo_excess", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        )
        e_bat = np.asarray(
            env_state.get("e_bat", np.zeros(n_agents, dtype=np.float32)),
            dtype=np.float32,
        )

        dvm_dp = np.asarray(sensitivity.get("dvm_dp", np.zeros((0, n_agents))), dtype=np.float32)
        dline_dp = np.asarray(
            sensitivity.get("dline_loading_dp", np.zeros((0, n_agents))), dtype=np.float32
        )
        dtrafo_dp = np.asarray(
            sensitivity.get("dtrafo_loading_dp", np.zeros((0, n_agents))), dtype=np.float32
        )

        w_sc = float(env_state.get("w_sens_credit", self.w_sens_credit))
        scale = max(float(env_state.get("sens_credit_scale", self.sens_credit_scale)), 1e-12)

        for i in range(n_agents):
            # dpsi_v / dp_i
            dpsi_v_dp_i = 0.0
            if bus_v_excess.size > 0 and dvm_dp.shape[0] == bus_v_excess.size:
                dpsi_v_dp_i = float(
                    np.sum(2.0 * bus_v_excess * bus_v_signed * dvm_dp[:, i])
                )

            # dpsi_line / dp_i（仅越限线路参与）
            dpsi_line_dp_i = 0.0
            if line_excess.size > 0 and dline_dp.shape[0] == line_excess.size:
                overloaded = (line_excess > 0.0).astype(np.float32)
                dpsi_line_dp_i = float(
                    np.sum(2.0 * line_excess * (dline_dp[:, i] / 100.0) * overloaded)
                )

            # dpsi_trafo / dp_i
            dpsi_trafo_dp_i = 0.0
            if trafo_excess.size > 0 and dtrafo_dp.shape[0] == trafo_excess.size:
                overloaded_t = (trafo_excess > 0.0).astype(np.float32)
                dpsi_trafo_dp_i = float(
                    np.sum(2.0 * trafo_excess * (dtrafo_dp[:, i] / 100.0) * overloaded_t)
                )

            # 加权合成梯度
            dpsi_total_dp_i = (
                w_v * dpsi_v_dp_i + w_line * dpsi_line_dp_i + w_trafo * dpsi_trafo_dp_i
            )
            raw_credit_i = -(dpsi_total_dp_i) * float(e_bat[i])
            r_sens[i] = float(w_sc * np.tanh(raw_credit_i / scale))

        return r_sens.astype(np.float32)
