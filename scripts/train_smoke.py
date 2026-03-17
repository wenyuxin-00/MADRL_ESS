"""最小训练脚本。

保留脚本入口，但不再把它当成唯一可用入口。
日常实验建议优先使用 `madrl/train_madrl.ipynb`。
"""

from __future__ import annotations

from configs import compose_experiment_config
from core.builder import build_train_runner


def main() -> int:
    """运行一个极小的 debug 训练任务。"""
    cfg = compose_experiment_config(profile="debug", algorithm="MADDPG", model_family="mlp")
    runner = build_train_runner(cfg, seed=0, env_name="SmokeEnv", number=1)
    try:
        return runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    completed = main()
    print(f"Completed episodes: {completed}")