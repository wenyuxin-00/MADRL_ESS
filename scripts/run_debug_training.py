"""Quick debug entry point for the GridEnv training mainline."""

from __future__ import annotations

from configs import compose_experiment_config
from scripts.builder import build_train_runner


def main() -> int:
    cfg = compose_experiment_config(profile="debug", algorithm="MADDPG", model_family="mlp")
    runner = build_train_runner(cfg, seed=0, env_name="GridEnvDebug", number=1)
    try:
        return runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    completed = main()
    print(f"Completed episodes: {completed}")
