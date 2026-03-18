"""奖励分量分析图。

将复合奖励拆分为各分量并可视化，便于调试奖励设计。

主要函数:
    plot_reward_breakdown -- 绘制奖励分量分解图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_reward_decomposition(history, episode_rewards, reward_fn, title, window=20):
    """自适应画图：Total + 各奖励分量（子图数/颜色/标题由 reward_fn.component_meta 驱动）。

    新增奖励分量只需修改 composite.py 的 component_meta，此函数无需改动。

    Parameters
    ----------
    history : list[dict]
        TrainRunner.history，每个元素为一个 episode 的轨迹字典。
    episode_rewards : list[float]
        TrainRunner.episode_rewards，每个 episode 的 total reward（sum over agents）。
    reward_fn : RewardFn
        env.reward_fn 实例，用于读取 component_meta。
    title : str
        图标题。
    window : int
        移动平均窗口大小，默认 20。
    """
    if len(history) == 0:
        print("No history to plot.")
        return

    metas   = reward_fn.component_meta
    n_plots = 1 + len(metas)

    ep_total = np.array(episode_rewards, dtype=np.float32)
    ep_comps = [
        np.array([np.sum(ep[f"{m.key}_sum"]) for ep in history], dtype=np.float32)
        for m in metas
    ]

    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)
    if n_plots == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=15)

    def _plot(ax, data, color, name):
        ma = pd.Series(data).rolling(window=window, min_periods=1).mean()
        ax.plot(ma, color=color, linewidth=2, label=f"MA({window})")
        ax.set_title(name)
        ax.grid(True, linestyle=":")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="best")

    _plot(axs[0], ep_total, "red", "1) Episode Total Reward (sum over agents)")
    for i, (ep_c, meta) in enumerate(zip(ep_comps, metas)):
        _plot(axs[i + 1], ep_c, meta.color, f"{i+2}) {meta.label}")

    axs[-1].set_xlabel("Episode")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
