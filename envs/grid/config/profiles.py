"""Named grid profiles for power-flow-constrained training."""

from __future__ import annotations

from typing import Any

DEFAULT_GRID_PROFILE = "rural1_phase1"
GRID_PROFILES = (DEFAULT_GRID_PROFILE,)


def apply_grid_profile(cfg: Any, profile: str = DEFAULT_GRID_PROFILE) -> Any:
    """Mutate ``cfg`` in place with the selected grid profile."""
    if profile == DEFAULT_GRID_PROFILE:
        _apply_rural1_phase1(cfg)
        return cfg
    raise ValueError(f"Unknown grid profile '{profile}'. Available: {list(GRID_PROFILES)}")



def _apply_rural1_phase1(cfg: Any) -> None:
    """Phase-1 rural feeder defaults used by the grid training mainline."""
    cfg.env.env_type = "grid_pf"
    cfg.env.num_agents = 3
    cfg.env.episode_limit = 96 * 2
    cfg.env.future_horizon = 24
    cfg.env.battery_capacity = 5.0
    cfg.env.max_charge_rate = 2.5
    cfg.env.efficiency = 0.95
    cfg.env.init_soc = 0.5
    cfg.env.soc_min = 0.05
    cfg.env.soc_max = 0.95
    cfg.env.soc_target = 0.5
    cfg.env.dt = 0.25
    cfg.env.storage_power_scale = 12.0
    cfg.env.storage_capacity_scale = 12.0

    cfg.grid.sb_code = "1-LV-rural1--0-sw"
    cfg.grid.pf_solver = "nr"
    cfg.grid.agent_bus_ids = [10, 6, 12]
    cfg.grid.v_min_pu = 0.95
    cfg.grid.v_max_pu = 1.05
    cfg.grid.line_max_loading_pct = 100.0
    cfg.grid.w_v_pen = 10.0
    cfg.grid.w_line_pen = 10.0
    cfg.grid.w_trafo_pen = 10.0
    cfg.grid.sensitivity_delta_kw = 1.0
    cfg.grid.sensitivity_max_staleness_steps = 32
    cfg.grid.train_compact_info = True
    cfg.grid.sensitivity_trigger_action_delta_kw = 1.0
    cfg.grid.sensitivity_trigger_load_delta_kw = 2.0
    cfg.grid.sensitivity_trigger_psi_delta = 0.001
    cfg.grid.sensitivity_trigger_on_pf_recovery = True
    cfg.grid.sensitivity_trigger_on_violation_change = True

    cfg.data.dataset_type = "csv_prosumer"

    cfg.reward.type = "grid_composite"
    cfg.reward.w_pen = 6.0
    cfg.reward.w_soc = 0.30
    cfg.reward.lambda_bonus = 0.001
    cfg.reward.w_global_safe = 1.0
    cfg.reward.w_sens_credit = 0.2
    cfg.reward.sens_credit_scale = 0.05

    cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.obs.adjacency_type = "identity"

    cfg.forecast.type = "perfect"
    cfg.forecast.target_signals = ["price", "load", "pv"]

    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
