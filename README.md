# MADRL-ESS: Multi-Agent Deep Reinforcement Learning for Energy Storage Scheduling

A modular research platform for studying multi-agent reinforcement learning in energy storage scheduling scenarios. Designed for easy experimentation with different algorithms, model architectures, forecasters, and reward designs.

## Architecture Overview

```
ExperimentConfig (configs/)
        |
        v
  build_train_runner (core/builder.py)
        |
        +---> build_dataset       (datasets/registry.py)     -- price & load signals
        +---> build_forecaster    (forecast/registry.py)     -- price prediction
        +---> build_obs_builder   (envs/observation/registry.py)  -- structured obs
        +---> get_reward_fn       (common/rewards/__init__.py)    -- reward function
        +---> get_env_cls         (envs/registry.py)         -- RL environment
        +---> build_actor/critic  (models/assembly.py)       -- neural networks
        +---> get_agent_cls       (algorithms/registry.py)   -- MADRL algorithm
        |
        v
  TrainRunner.run() (runners/train_runner.py)
        |
        v
  evaluate_controller (evaluation/evaluator.py)
        |
        v
  Visualization (evaluation/plots.py, reward_plots.py)
```

## Model Assembly Pipeline

```
Structured Obs Dict  -->  Adapter  -->  Encoder  -->  Head  -->  Output
                          (MLP/       (MLP/        (Actor/
                          Transformer/ Transformer/  Critic
                          Graph)       Graph)        Head)
```

## Directory Structure

```
MADRL_ESS/
  algorithms/        -- MADRL algorithms (MADDPG, MATD3) + base class & registry
  common/            -- Shared utilities: replay buffer, vec envs, nested helpers
    rewards/         -- Reward functions (composite, sparse) + base class & registry
  configs/           -- Dataclass config hierarchy + notebook-friendly profiles
  controllers/       -- Unified evaluation interface (MADRL, Zero, MPC, DRL placeholders)
  core/              -- Builder: single entry point for assembling experiments
  data/              -- CSV datasets (train_prices.csv, test_prices.csv)
  datasets/          -- Dataset loading & episode slicing + registry
  envs/              -- RL environments (EnergyStorageEnv) + registry
    observation/     -- Structured observation building framework
  evaluation/        -- Evaluation, comparison, and plotting utilities
  forecast/          -- Price forecasters (Perfect, Naive, LSTM) + registry
  madrl/             -- Training & comparison notebooks
  models/            -- Neural network assembly (adapters, encoders, heads) + registry
    encoders/        -- MLP, Transformer, Graph encoders
    heads/           -- Actor & critic output heads
  runners/           -- Training loop (TrainRunner) + checkpoint management
  scripts/           -- Standalone scripts (smoke test)
  tests/             -- Comprehensive test suite
```

## Quick Start

```python
from configs import compose_experiment_config, print_experiment_summary
from core import build_train_runner

# 1. Compose a debug config (small scale, fast iteration)
cfg = compose_experiment_config(train_profile="debug")
print_experiment_summary(cfg)

# 2. Build and run training
runner = build_train_runner(cfg)
runner.run()

# 3. Evaluate
from controllers import MADRLController
from evaluation import evaluate_controller
from core import build_env

controller = MADRLController(runner.agents)
eval_env = build_env(cfg, mode="test")
results = evaluate_controller(eval_env, controller, n_episodes=5)
print(f"Mean reward: {results['mean_episode_reward']:.2f}")
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `forecast/forecast.ipynb` | Train & evaluate the LSTM price forecaster, save artifacts |
| `madrl/train_madrl.ipynb` | Configure, train, and visualize MADRL agents |
| `madrl/compare.ipynb` | Compare controllers (MADRL vs. Zero vs. baselines) |

## How to Extend

Each component is pluggable via a **registry pattern**. To add a new component:

| Component | Base class | Registry file | Config field |
|-----------|-----------|---------------|-------------|
| Algorithm | `BaseAgent` | `algorithms/registry.py` | `cfg.algo.name` |
| Environment | `gym.Env` | `envs/registry.py` | `cfg.env.env_type` |
| Dataset | `BaseEpisodeDataset` | `datasets/registry.py` | `cfg.data.dataset_type` |
| Forecaster | `Forecaster` | `forecast/registry.py` | `cfg.forecast.type` |
| Reward function | `RewardFn` | `common/rewards/__init__.py` | `cfg.reward.type` |
| Obs builder | `ObservationBuilder` | `envs/observation/registry.py` | `cfg.obs.builder_type` |
| Model encoder | `nn.Module` | `models/registry.py` | `cfg.model.family` |
| Controller | `BaseController` | (direct use) | N/A |

Each registry file contains a step-by-step "How to add" guide in its docstring.

## Configuration Profiles

Use `compose_experiment_config()` with profile arguments for quick setup:

```python
cfg = compose_experiment_config(
    train_profile="debug",        # "debug" | "fast_train" | "base"
    model_family="mlp",           # "mlp" | "transformer" | "graph"
    algo_name="MADDPG",           # "MADDPG" | "MATD3"
    reward_type="composite",      # "composite" | "sparse"
    forecast_type="perfect",      # "perfect" | "naive" | "lstm"
    obs_profile="default",        # "default" | "minimal" | "local_only"
)
```

## Testing

```bash
pytest                    # run all tests
pytest -x                 # stop on first failure
pytest tests/test_training_smoke.py  # run specific test
```

## Requirements

- Python 3.9+
- PyTorch 2.x
- gym 0.26.x
- See `requirements.txt` for full dependencies
