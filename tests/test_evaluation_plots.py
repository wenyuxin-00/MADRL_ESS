from __future__ import annotations

from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from scripts.plots.reward_plots import plot_reward_decomposition
from scripts.plots.rollout_plots import plot_voltage_and_net_load_dashboard
from scripts.utils.grid_notebook_workflow import (
    RolloutResult,
    plot_power_balance_bars,
    plot_rollout_comparison_dashboard,
    plot_rollout_dashboard,
    plot_test_voltage_profile,
)

def test_evaluation_plot_helpers_do_not_emit_glyph_warnings_with_english_titles(tmp_path):
    reward_summary = {
        "episodes": [1],
        "episode_total_reward": [1.0],
        "components": {
            "r_purchase_cost": {
                "label": "- r_purchase_cost (grid purchase cost)",
                "color": "green",
                "sign": -1,
                "values": [0.15],
            },
            "r_export_subsidy": {
                "label": "+ r_export_subsidy (grid export subsidy)",
                "color": "teal",
                "sign": 1,
                "values": [0.03],
            },
        },
        "aggregates": {"grid_safety_penalty": [-0.3]},
    }
    reward_summary_path = tmp_path / "reward_summary.json"
    reward_summary_path.write_text(json.dumps(reward_summary), encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught, patch(
        "scripts.plots.reward_plots.plt.show",
        side_effect=AssertionError("plot_reward_decomposition should not call plt.show()"),
    ):
        warnings.simplefilter("always")
        reward_figure = plot_reward_decomposition(
            reward_summary=reward_summary_path,
            title="Training Reward Decomposition",
            window=1,
        )
        summary_figure = plot_reward_decomposition(
            reward_summary=reward_summary,
            title="External Training Reward Decomposition",
            window=1,
        )

    assert reward_figure is not None
    assert summary_figure is not None
    plt.close(reward_figure)
    plt.close(summary_figure)
    glyph_warnings = [str(item.message) for item in caught if "Glyph" in str(item.message)]
    assert glyph_warnings == []


def test_voltage_plot_helper_renders_agent_and_background_buses():
    rollout = RolloutResult(
        step_df=pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"), "price": [0.1, 0.2], "price_pred": [0.1, 0.2]}),
        agent_df=pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"), "agent_profile": ["A", "A"], "e_bat": [0.0, 0.0], "soc": [0.5, 0.6], "load": [1.0, 1.1], "load_pred": [1.0, 1.1], "pv": [0.2, 0.3], "pv_pred": [0.2, 0.3], "controller": ["DRL", "DRL"], "episode_idx": [0, 0], "step": [0, 1], "agent_id": [0, 0]}),
        grid_df=pd.DataFrame(
            {
                "controller": ["DRL"] * 6,
                "episode_idx": [0] * 6,
                "step": [0, 0, 0, 1, 1, 1],
                "timestamp": list(pd.date_range("2020-01-01", periods=2, freq="15min").repeat(3)),
                "bus_id": [1, 2, 3, 1, 2, 3],
                "vm_pu": [1.00, 0.99, 1.01, 1.02, 0.98, 1.00],
                "is_agent_bus": [False, True, False, False, True, False],
            }
        ),
        summary=pd.DataFrame({"controller": ["DRL"], "agent_profile": ["A"], "purchase_cost": [0.3], "export_subsidy": [0.0], "objective_total": [0.3]}),
        meta={
            "controller": "DRL (oracle_eval)",
            "agent_profiles": ["A"],
            "agent_bus_ids": [2],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
        },
    )

    figure = plot_test_voltage_profile(rollout)
    assert len(figure.axes[0].lines) >= 3
    plt.close(figure)


def test_voltage_and_net_load_dashboard_renders_two_panels_with_limits():
    timestamps = pd.date_range("2020-01-01", periods=3, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "base_net_load_total": [2.5, 2.7, 2.8],
                "base_net_load_effective_total": [2.2, 2.4, 2.5],
                "net_load_total": [2.0, 2.1, 2.3],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(
            {
                "timestamp": list(timestamps.repeat(3)),
                "episode_idx": [0] * 9,
                "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "bus_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "vm_pu": [1.00, 0.99, 1.01, 1.01, 0.98, 1.00, 1.02, 0.97, 0.99],
                "is_agent_bus": [False, True, False, False, True, False, False, True, False],
            }
        ),
        summary=pd.DataFrame(),
        meta={
            "controller": "DRL (forecast_eval)",
            "agent_bus_ids": [2],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
            "trafo_limit_kw": 3.0,
        },
    )

    figure = plot_voltage_and_net_load_dashboard(rollout)

    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines) >= 4
    assert len(figure.axes[1].lines) >= 5
    plt.close(figure)


def test_rollout_dashboard_renders_expected_main_panels_for_three_agents():
    timestamps = pd.date_range("2020-01-01", periods=3, freq="15min")
    step_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [0.10, 0.20, 0.15],
            "price_pred": [0.12, 0.18, 0.16],
            "base_net_load_total": [2.40, 2.55, 2.70],
            "base_net_load_effective_total": [2.00, 2.15, 2.30],
            "net_load_total": [2.10, 2.20, 2.35],
            "pv_raw_total": [1.20, 1.10, 1.00],
            "pv_effective_total": [1.00, 0.90, 0.75],
            "pv_curtail_total": [0.20, 0.20, 0.25],
            "load_total": [3.20, 3.25, 3.30],
            "grid_import_total": [2.10, 2.20, 2.35],
            "grid_export_total": [0.0, 0.0, 0.0],
            "battery_charge_total": [0.3, 0.2, 0.15],
            "battery_discharge_total": [0.0, 0.0, 0.0],
            "purchase_cost_total": [1.0, 1.1, 0.9],
            "export_subsidy_total": [0.0, 0.0, 0.0],
            "objective_total": [1.0, 1.1, 0.9],
            "episode_idx": [0, 0, 0],
            "step": [0, 1, 2],
        }
    )
    agent_rows = []
    for agent_id, profile in enumerate(["A", "B", "C"]):
        for step_idx, timestamp in enumerate(timestamps):
            agent_rows.append(
                {
                    "timestamp": timestamp,
                    "agent_profile": profile,
                    "agent_id": agent_id,
                    "episode_idx": 0,
                    "step": step_idx,
                    "e_bat": (-1) ** agent_id * (0.1 + 0.05 * step_idx),
                    "soc": 0.4 + 0.1 * agent_id + 0.03 * step_idx,
                    "load": 1.0 + 0.1 * agent_id + 0.05 * step_idx,
                    "load_pred": 0.95 + 0.1 * agent_id + 0.04 * step_idx,
                    "pv": 0.2 + 0.03 * agent_id + 0.02 * step_idx,
                    "pv_raw": 0.2 + 0.03 * agent_id + 0.02 * step_idx,
                    "pv_effective": 0.17 + 0.02 * agent_id + 0.015 * step_idx,
                    "pv_curtail": 0.03 + 0.01 * agent_id + 0.005 * step_idx,
                    "pv_utilization": 0.85 - 0.03 * agent_id,
                    "pv_pred": 0.18 + 0.03 * agent_id + 0.01 * step_idx,
                    "base_net_load": 0.8 + 0.07 * agent_id + 0.03 * step_idx,
                    "base_net_load_effective": 0.7 + 0.06 * agent_id + 0.02 * step_idx,
                    "net_load": 0.7 + 0.05 * agent_id + 0.02 * step_idx,
                    "grid_import_kw": 0.7 + 0.05 * agent_id + 0.02 * step_idx,
                    "grid_export_kw": 0.0,
                    "purchase_cost": 0.2 + 0.1 * agent_id,
                    "export_subsidy": 0.0,
                    "objective_total": 0.2 + 0.1 * agent_id,
                    "controller": "DRL",
                }
            )
    agent_df = pd.DataFrame(agent_rows)
    grid_rows = []
    bus_ids = [1, 2, 3, 4, 5]
    for step_idx, timestamp in enumerate(timestamps):
        for bus_id in bus_ids:
            grid_rows.append(
                {
                    "controller": "DRL",
                    "episode_idx": 0,
                    "step": step_idx,
                    "timestamp": timestamp,
                    "bus_id": bus_id,
                    "vm_pu": 0.98 + 0.005 * bus_id - 0.002 * step_idx,
                    "is_agent_bus": bus_id in {2, 3, 4},
                }
            )
    rollout = RolloutResult(
        step_df=step_df,
        agent_df=agent_df,
        grid_df=pd.DataFrame(grid_rows),
        summary=pd.DataFrame(
            {
                "controller": ["DRL"] * 3,
                "agent_profile": ["A", "B", "C"],
                "purchase_cost": [0.5, 0.6, 0.7],
                "export_subsidy": [0.0, 0.0, 0.0],
                "objective_total": [0.5, 0.6, 0.7],
            }
        ),
        meta={
            "controller": "DRL (forecast_eval)",
            "agent_profiles": ["A", "B", "C"],
            "agent_bus_ids": [2, 3, 4],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
        },
    )

    figure = plot_rollout_dashboard(rollout)
    main_axes = getattr(figure, "_dashboard_main_axes")
    assert len(main_axes) == 9
    assert len(main_axes[0].lines) == 2
    assert len(main_axes[1].lines) == 6
    assert len(main_axes[2].lines) == 6
    assert len(main_axes[3].lines) >= 5
    assert len(main_axes[4].lines) == 3
    assert len(main_axes[5].lines) >= 2
    assert len(main_axes[6:]) == 3
    plt.close(figure)


def test_power_balance_plot_renders_expected_stacks():
    timestamps = pd.date_range("2020-01-01", periods=3, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "load_total": [3.0, 3.2, 3.1],
                "battery_charge_total": [0.4, 0.1, 0.0],
                "pv_effective_total": [1.5, 1.2, 1.0],
                "pv_curtail_total": [0.2, 0.1, 0.0],
                "grid_import_total": [1.2, 1.7, 1.9],
                "grid_export_total": [0.0, 0.0, 0.0],
                "battery_discharge_total": [0.0, 0.2, 0.3],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "DRL (forecast_eval)"},
    )

    figure = plot_power_balance_bars(rollout)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].patches) == 21
    plt.close(figure)


def test_rollout_comparison_dashboard_renders_six_panels():
    metrics_df = pd.DataFrame(
        {
            "controller": ["MPC (oracle_eval)", "MPC (forecast_eval)", "DRL (forecast_eval)"],
            "purchase_cost_total": [1.0, 1.2, 0.9],
            "export_subsidy_total": [0.1, 0.1, 0.2],
            "objective_total": [0.9, 1.3, 0.7],
            "voltage_violation_count": [0, 2, 0],
            "trafo_penalty_total": [0.0, 0.3, 0.0],
            "line_penalty_total": [0.0, 0.2, 0.0],
        }
    )

    figure = plot_rollout_comparison_dashboard(metrics_df)
    assert len(figure.axes) == 6
    plt.close(figure)
