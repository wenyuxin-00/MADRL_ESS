"""Grid-specific experiment profiles.

Usage in a notebook::

    from configs import compose_experiment_config
    from configs.grid_profiles import apply_grid_profile

    cfg = compose_experiment_config(profile="debug", algorithm="MADDPG")
    cfg = apply_grid_profile(cfg, "rural1_phase1")
    # Then optionally override individual fields:
    cfg.train.train_episodes = 300

Available profiles
------------------
``rural1_phase1``
    Fixed SimBench ``1-LV-rural1--0-sw`` topology, 3 prosumer agents,
    ``p_batt`` control only, ``grid_composite`` reward.
    Uses ``csv_prosumer`` dataset.  Sets ``num_envs=1`` and
    ``vec_env_type="dummy"`` because multi-process pandapower is not
    validated until Phase 3.
"""

from __future__ import annotations

from typing import Any


def apply_grid_profile(cfg: Any, profile: str = "rural1_phase1") -> Any:
    """Apply a named grid profile to an existing ``ExperimentConfig``.

    Parameters
    ----------
    cfg:
        Any ``ExperimentConfig`` instance, typically built with
        ``compose_experiment_config()`` first.
    profile:
        Name of the grid profile to apply.

    Returns
    -------
    ExperimentConfig
        The same object, mutated in-place and returned for chaining.

    Raises
    ------
    ValueError
        If *profile* is not recognised.
    """
    if profile == "rural1_phase1":
        _apply_rural1_phase1(cfg)
    else:
        raise ValueError(
            f"Unknown grid profile '{profile}'. Available: ['rural1_phase1']"
        )
    return cfg


def _apply_rural1_phase1(cfg: Any) -> None:
    """Phase 1: SimBench 1-LV-rural1--0-sw, 3 prosumer agents, p_batt only."""
    # Environment.
    cfg.env.env_type = "grid_pf"
    cfg.env.num_agents = 3
    cfg.env.episode_limit = 96 * 2          # 2 days × 96 steps/day
    cfg.env.future_horizon = 24             # 6-hour look-ahead (24 × 15 min)
    cfg.env.battery_capacity = 5.0          # kWh
    cfg.env.max_charge_rate = 2.5           # kW
    cfg.env.efficiency = 0.95
    cfg.env.init_soc = 0.5
    cfg.env.soc_min = 0.05
    cfg.env.soc_max = 0.95
    cfg.env.soc_target = 0.5
    cfg.env.dt = 0.25                       # 15-minute intervals

    # Grid.
    cfg.grid.sb_code = "1-LV-rural1--0-sw"
    cfg.grid.pf_solver = "nr"
    cfg.grid.agent_bus_ids = [10, 6, 12]
    cfg.grid.v_min_pu = 0.95
    cfg.grid.v_max_pu = 1.05
    cfg.grid.line_max_loading_pct = 100.0
    cfg.grid.w_v_pen = 10.0
    cfg.grid.w_l_pen = 5.0

    # Dataset.
    cfg.data.dataset_type = "csv_prosumer"

    # Reward.
    cfg.reward.type = "grid_composite"

    # Observation — same as simbench profile.
    cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.obs.adjacency_type = "identity"

    # Forecast — oracle (perfect knowledge) for Phase 1.
    cfg.forecast.type = "perfect"
    cfg.forecast.target_signals = ["price", "load", "pv"]

    # Training — Phase 1 uses a single environment to avoid multi-process
    # pandapower serialisation issues.
    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
