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
            "penalty_sum": [0.2, 0.1],
        }
    ]
    reward_fn = SimpleNamespace(
        component_meta=[SimpleNamespace(key="penalty", label="Penalty", color="tab:red")]
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
