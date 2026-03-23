from __future__ import annotations

from types import SimpleNamespace
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.plots import plot_last_k_episodes_price_action_soc
from scripts.plots.reward_plots import plot_reward_decomposition


def test_evaluation_plot_helpers_do_not_emit_glyph_warnings_with_english_titles():
    history = [
        {
            "price": [0.1, 0.2, 0.3],
            "soc": [[0.5, 0.55, 0.6, 0.65], [0.4, 0.45, 0.5, 0.55]],
            "e_bat_req": [[0.1, -0.1, 0.0], [0.0, 0.1, -0.1]],
            "e_bat_exec": [[0.1, -0.05, 0.0], [0.0, 0.1, -0.05]],
            "r_inc_sum": [0.1, 0.05],
            "r_pen_sum": [-0.05, -0.02],
            "r_pbrs_sum": [0.02, 0.01],
            "r_soc_sum": [-0.01, -0.005],
            "r_bonus_sum": [0.001, 0.001],
            "r_safe_v_global_sum": [-0.2, -0.1],
            "r_safe_line_global_sum": [-0.4, -0.3],
            "r_safe_trafo_global_sum": [-0.1, -0.2],
            "r_sens_credit_sum": [0.01, 0.02],
        }
    ]
    reward_fn = SimpleNamespace(
        component_meta=[
            SimpleNamespace(key="r_inc", label="+ r_inc (incremental cost)", color="green", sign=+1),
            SimpleNamespace(key="r_pen", label="- r_pen (action penalty)", color="orange", sign=-1),
            SimpleNamespace(key="r_pbrs", label="+ r_pbrs (PBRS)", color="blue", sign=+1),
            SimpleNamespace(key="r_soc", label="- r_soc (SoC regularization)", color="purple", sign=-1),
            SimpleNamespace(key="r_bonus", label="+ r_bonus (throughput bonus)", color="teal", sign=+1),
            SimpleNamespace(key="r_safe_v_global", label="- r_safe_v (global voltage penalty)", color="red", sign=-1),
            SimpleNamespace(key="r_safe_line_global", label="- r_safe_line (global line penalty)", color="brown", sign=-1),
            SimpleNamespace(key="r_safe_trafo_global", label="- r_safe_trafo (global trafo penalty)", color="maroon", sign=-1),
            SimpleNamespace(key="r_sens_credit", label="+ r_sens_credit (sensitivity credit)", color="darkgreen", sign=+1),
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
