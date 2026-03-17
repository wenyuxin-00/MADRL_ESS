"""Evaluation and comparison utilities.
评估与对比入口。

Key functions / 主要功能:
    - evaluate_controller()           -- run a controller on an env, collect rewards
    - evaluate_controller_suite()     -- compare multiple controllers side-by-side
    - init_episode_record / append_step_record  -- per-step history recording
    - plot_comparison_bar()           -- bar chart comparing mean rewards
    - plot_last_k_episodes_price_action_soc()  -- trajectory visualization
    - plot_reward_decomposition()     -- reward component breakdown over training
"""

from evaluation.comparison import comparison_records_to_rows, evaluate_controller_suite, plot_comparison_bar
from evaluation.episode_recorder import append_step_record, init_episode_record
from evaluation.evaluator import evaluate_controller

# Plotting functions require matplotlib; import them explicitly when needed:
#   from evaluation.plots import plot_last_k_episodes_price_action_soc
#   from evaluation.reward_plots import plot_reward_decomposition

__all__ = [
    "append_step_record",
    "comparison_records_to_rows",
    "evaluate_controller",
    "evaluate_controller_suite",
    "init_episode_record",
    "plot_comparison_bar",
]
