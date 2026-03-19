"""多控制器对比评估。

同时评估多个控制器（如 MADRL vs MPC vs Zero）并生成对比报告。

主要函数:
    compare_controllers -- 对比多个控制器
"""

from __future__ import annotations

from typing import Callable

from scripts.evaluate import evaluate_controller


def evaluate_controller_suite(
    env_factory: Callable[[], object],
    controller_builders: dict[str, Callable[[], object]],
    *,
    n_episodes: int = 1,
    deterministic: bool = True,
) -> list[dict]:
    """逐个评估多个控制器，并优雅处理占位符或错误。

    对每个控制器独立创建环境实例，运行评估后关闭。
    未实现的控制器或运行出错的控制器会被标记状态而非中断流程。

    参数:
        env_factory: 无参可调用对象，每次调用返回一个新环境实例。
        controller_builders: 控制器名称到构建函数的映射字典。
        n_episodes: 每个控制器评估的 episode 数量。
        deterministic: 是否使用确定性策略。

    返回:
        评估记录列表，每个记录包含 controller 名称、状态和评估结果。
    """
    records = []
    for name, build_controller in controller_builders.items():
        env = env_factory()
        try:
            # 构建控制器并执行评估
            controller = build_controller()
            result = evaluate_controller(
                env=env,
                controller=controller,
                n_episodes=n_episodes,
                deterministic=deterministic,
                record_history=True,
            )
            records.append(
                {
                    "controller": name,
                    "status": "ok",
                    "mean_episode_reward": result["mean_episode_reward"],
                    "episode_rewards": result["episode_rewards"],
                    "histories": result.get("histories", []),
                }
            )
        except NotImplementedError as exc:
            # 控制器尚未实现（占位符），记录状态但不中断
            records.append(
                {
                    "controller": name,
                    "status": "not_implemented",
                    "message": str(exc),
                    "mean_episode_reward": None,
                    "episode_rewards": [],
                    "histories": [],
                }
            )
        except Exception as exc:
            # 其他运行时错误，记录错误信息
            records.append(
                {
                    "controller": name,
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "mean_episode_reward": None,
                    "episode_rewards": [],
                    "histories": [],
                }
            )
        finally:
            env.close()
    return records


def comparison_records_to_rows(records: list[dict]) -> list[dict]:
    """把评估记录整理成 notebook 更易展示的简表行。

    参数:
        records: evaluate_controller_suite 返回的评估记录列表。

    返回:
        简化后的行列表，每行包含控制器名、状态、平均奖励和消息。
    """
    rows = []
    for record in records:
        rows.append(
            {
                "controller": record["controller"],
                "status": record["status"],
                "mean_episode_reward": record.get("mean_episode_reward"),
                "message": record.get("message", ""),
            }
        )
    return rows


def plot_comparison_bar(records: list[dict], title: str = "Controller Comparison"):
    """画一张对比各控制器均值回报的柱状图，忽略未实现项。

    参数:
        records: evaluate_controller_suite 返回的评估记录列表。
        title: 图表标题。
    """
    import matplotlib.pyplot as plt

    # 仅保留评估成功的记录
    valid = [record for record in records if record["status"] == "ok"]
    if not valid:
        print("No valid controller results to plot.")
        return

    labels = [record["controller"] for record in valid]
    values = [record["mean_episode_reward"] for record in valid]

    # 为不同控制器分配不同颜色
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#335c67", "#9e2a2b", "#e09f3e", "#540b0e"][: len(valid)])
    ax.set_title(title)
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.show()
