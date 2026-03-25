from __future__ import annotations

from types import SimpleNamespace
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plots.plots import plot_last_k_episodes_price_action_soc
from scripts.plots.reward_plots import plot_reward_decomposition
from scripts.utils.grid_notebook_workflow import RolloutResult, plot_test_voltage_profile


def test_evaluation_plot_helpers_do_not_emit_glyph_warnings_with_english_titles():
    history = [
        {
            "price": [0.1, 0.2, 0.3],
            "soc": [[0.5, 0.55, 0.6, 0.65], [0.4, 0.45, 0.5, 0.55]],
            "e_bat_req": [[0.1, -0.1, 0.0], [0.0, 0.1, -0.1]],
            "e_bat_exec": [[0.1, -0.05, 0.0], [0.0, 0.1, -0.05]],
            "r_cost_sum": [0.1, 0.05],
            "r_throughput_sum": [0.02, 0.01],
            "r_action_pen_sum": [-0.05, -0.02],
            "r_safe_v_sum": [-0.2, -0.1],
            "r_safe_trafo_sum": [-0.1, -0.2],
        }
    ]
    reward_fn = SimpleNamespace(
        component_meta=[
            SimpleNamespace(key="r_cost", label="+ r_cost (incremental cost)", color="green", sign=+1),
            SimpleNamespace(key="r_throughput", label="+ r_throughput (throughput bonus)", color="teal", sign=+1),
            SimpleNamespace(key="r_action_pen", label="- r_action_pen (action penalty)", color="orange", sign=-1),
            SimpleNamespace(key="r_safe_v", label="- r_safe_v (voltage penalty)", color="red", sign=-1),
            SimpleNamespace(key="r_safe_trafo", label="- r_safe_trafo (trafo penalty)", color="maroon", sign=-1),
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_reward_decomposition(
            history=history,
            episode_rewards=[1.0],
            reward_fn=reward_fn,
            title="Training Reward Decomposition",
            window=1,
        )
        plot_last_k_episodes_price_action_soc(
            history=history,
            k=1,
            n_agents=2,
            title_prefix="Eval",
        )

    plt.close("all")
    glyph_warnings = [str(item.message) for item in caught if "Glyph" in str(item.message)]
    assert glyph_warnings == []


def test_voltage_plot_helper_renders_agent_and_background_buses():
    rollout = RolloutResult(
        step_df=pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"), "price": [0.1, 0.2], "price_pred": [0.1, 0.2], "operating_cost": [1.0, 1.2]}),
        agent_df=pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"), "agent_profile": ["A", "A"], "e_bat": [0.0, 0.0], "soc": [0.5, 0.6], "load": [1.0, 1.1], "load_pred": [1.0, 1.1], "pv": [0.2, 0.3], "pv_pred": [0.2, 0.3], "operating_cost": [0.1, 0.2], "controller": ["DRL", "DRL"], "episode_idx": [0, 0], "step": [0, 1], "agent_id": [0, 0]}),
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
        summary=pd.DataFrame({"controller": ["DRL"], "agent_profile": ["A"], "operating_cost": [0.3]}),
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
