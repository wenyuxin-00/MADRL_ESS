"""快速调试训练入口。

使用 debug profile 快速启动一次小规模训练，
用于验证训练流程是否正常。
"""

from __future__ import annotations

from configs import compose_experiment_config
from scripts.builder import build_train_runner


def main() -> int:
    """Run one small debug experiment from the command line."""
    cfg = compose_experiment_config(profile="debug", algorithm="MADDPG", model_family="mlp")
    runner = build_train_runner(cfg, seed=0, env_name="SmokeEnv", number=1)
    try:
        return runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    completed = main()
    print(f"Completed episodes: {completed}")
