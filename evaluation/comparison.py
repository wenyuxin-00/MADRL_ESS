"""compare notebook 使用的统一对比辅助函数。"""

from __future__ import annotations

from typing import Callable

from evaluation.evaluator import evaluate_controller


def evaluate_controller_suite(
    env_factory: Callable[[], object],
    controller_builders: dict[str, Callable[[], object]],
    *,
    n_episodes: int = 1,
    deterministic: bool = True,
) -> list[dict]:
    """逐个评估 controller，并优雅处理占位符或错误。"""
    records = []
    for name, build_controller in controller_builders.items():
        env = env_factory()
        try:
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
    """把评估记录整理成 notebook 更易展示的简表。"""
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
    """画一张对比均值回报柱状图，忽略未实现项。"""
    import matplotlib.pyplot as plt

    valid = [record for record in records if record["status"] == "ok"]
    if not valid:
        print("No valid controller results to plot.")
        return

    labels = [record["controller"] for record in valid]
    values = [record["mean_episode_reward"] for record in valid]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#335c67", "#9e2a2b", "#e09f3e", "#540b0e"][: len(valid)])
    ax.set_title(title)
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.show()
