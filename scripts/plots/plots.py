"""通用训练曲线绘制。

绘制训练过程中的奖励曲线、损失曲线等可视化图表。

主要函数:
    plot_training_curves -- 绘制训练曲线
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_last_k_episodes_price_action_soc(history, k=2, n_agents=3, title_prefix="Train"):
    """画最后 k 个 episode：子图1电价；子图2..N+1 各agent电池功率(e_bat)柱状 + SOC折线。

    Parameters
    ----------
    history : list[dict]
        TrainRunner.history，每个元素为一个 episode 的轨迹字典。
    k : int
        展示最后 k 个 episode，默认 2。
    n_agents : int
        智能体数量，默认 3。
    title_prefix : str
        图标题前缀，默认 "Train"。
    """
    if len(history) == 0:
        print("No history to plot.")
        return

    k = min(k, len(history))
    start = len(history) - k

    for ep_idx in range(start, len(history)):
        ep = history[ep_idx]
        price = np.asarray(ep["price"], dtype=np.float32)            # (T,)
        T = len(price)
        ts = np.arange(T)

        fig, axs = plt.subplots(1 + n_agents, 1, figsize=(11, 2.4 * (1 + n_agents)), sharex=True)
        fig.suptitle(f"{title_prefix} Episode {ep_idx + 1}: Price + Battery Power & SOC", fontsize=14, y=0.995)

        # 1) price
        ax0 = axs[0]
        ax0.plot(ts, price, color="m", linewidth=2, label="Price")
        ax0.set_title("Price")
        ax0.set_ylabel("Price")
        ax0.grid(True, linestyle=":")
        ax0.legend(loc="best")

        # 2.. N+1) 每个 agent：e_bat 柱状 + soc 折线
        for i in range(n_agents):
            ax = axs[i + 1]
            soc = np.asarray(ep["soc"][i], dtype=np.float32)         # (T+1,)
            p_req = np.asarray(ep["e_bat_req"][i], dtype=np.float32)   # (T,)
            p_exec = np.asarray(ep["e_bat_exec"][i], dtype=np.float32) # (T,)

            # 执行功率：实心柱
            ax.bar(ts, np.maximum(0, p_exec), color="red", width=0.85, label="Exec Charge (+)")
            ax.bar(ts, np.minimum(0, p_exec), color="green", width=0.85, label="Exec Discharge (-)")

            # 请求功率：半透明叠加（同颜色但 alpha 更低）
            ax.bar(ts, np.maximum(0, p_req), color="red", width=0.85, alpha=0.25, label="Req Charge (+)")
            ax.bar(ts, np.minimum(0, p_req), color="green", width=0.85, alpha=0.25, label="Req Discharge (-)")
            ax.set_title(f"Agent {i+1}: e_bat (bar) & SOC (line)")
            ax.set_ylabel("e_bat")
            ax.grid(True, axis="y", linestyle=":")
            ax.legend(loc="upper left")

            ax2 = ax.twinx()
            soc_plot = soc[1:] if soc.shape[0] == T + 1 else soc[:T]
            ax2.plot(ts, soc_plot, color="blue", marker=".", markersize=3, linewidth=1.5, label="SOC")
            ax2.set_ylabel("SOC", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")
            ax2.set_ylim(0.0, 1.05)
            ax2.legend(loc="upper right")

        axs[-1].set_xlabel("Time Step")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
