"""
core/builder.py
职责：提供统一的构建入口函数，将 notebook 中的实例化逻辑集中管理。

第二步重构：完整装配流水线
  dataset → forecaster → obs_builder → reward_fn → env_factory → DummyVecEnv
  → eval_env → 设置 args 维度 → TrainRunner

设计原则：
  - 每个训练环境拥有独立的 forecaster 实例（forecaster 有 episode 级状态）
  - obs_builder 无状态，可在所有训练 env 间共享（但每个 env 各自持有一份即可）
  - eval env 独立使用 test 数据集与独立 forecaster
"""

from common.vec_env import DummyVecEnv
from common.rewards import get_reward_fn
from datasets.registry import build_dataset
from envs.hems_env import EnergyStorageEnv
from envs.observation.registry import build_obs_builder
from forecast.registry import build_forecaster
from models.registry import ACTOR_REGISTRY, CRITIC_REGISTRY
from runners.train_runner import TrainRunner

ALGORITHM_DEFAULT_CRITIC: dict = {
    "MADDPG": "maddpg_mlp",
    "MATD3": "matd3_mlp",
}

ALGORITHM_ALLOWED_CRITICS: dict = {
    "MADDPG": {"maddpg_mlp"},
    "MATD3": {"matd3_mlp"},
}


def resolve_algorithm_model_config(args):
    """Fill default model config and fail early on invalid registry keys.

    The third refactor round moves model ownership to ``models.registry``.
    Doing the validation here keeps notebook usage unchanged while surfacing
    clear configuration errors before training starts.
    """
    # In notebook autoreload flows, an older function object can occasionally
    # survive while module globals are mid-refresh. Keep a local fallback so
    # config resolution still works instead of failing with NameError.
    algorithm_default_critic = globals().get(
        "ALGORITHM_DEFAULT_CRITIC",
        {
            "MADDPG": "maddpg_mlp",
            "MATD3": "matd3_mlp",
        },
    )
    algorithm_allowed_critics = globals().get(
        "ALGORITHM_ALLOWED_CRITICS",
        {
            "MADDPG": {"maddpg_mlp"},
            "MATD3": {"matd3_mlp"},
        },
    )

    algorithm = getattr(args, "algorithm", "MADDPG")
    if algorithm not in algorithm_default_critic:
        raise ValueError(
            f"Unknown algorithm '{algorithm}', available: {list(algorithm_default_critic)}"
        )

    if getattr(args, "actor_type", None) is None:
        args.actor_type = "mlp"

    if getattr(args, "critic_type", None) is None:
        args.critic_type = algorithm_default_critic[algorithm]

    allowed_critics = algorithm_allowed_critics[algorithm]
    if args.critic_type not in allowed_critics:
        allowed = sorted(allowed_critics)
        default = algorithm_default_critic[algorithm]
        raise ValueError(
            f"algorithm '{algorithm}' is incompatible with critic_type '{args.critic_type}'. "
            f"Allowed critics: {allowed}. Default: '{default}'."
        )

    if args.actor_type not in ACTOR_REGISTRY:
        raise ValueError(
            f"Unknown actor_type '{args.actor_type}', available: {list(ACTOR_REGISTRY)}"
        )
    if args.critic_type not in CRITIC_REGISTRY:
        raise ValueError(
            f"Unknown critic_type '{args.critic_type}', available: {list(CRITIC_REGISTRY)}"
        )

    return args


def build_train_runner(args, seed: int = 0, env_name: str = "EnergyStorageEnv", number: int = 1) -> TrainRunner:
    """完整装配并返回 TrainRunner 实例。

    流水线：
      1. 构建 train/test 数据集
      2. 构建共享 reward_fn
      3. 定义 env 工厂函数（每次调用返回一个新 env，含独立 forecaster）
      4. 构建 DummyVecEnv（32 个并行训练环境）
      5. 构建 eval env（单实例，test 数据集）
      6. 从 eval env 推断 obs_dim / action_dim，写入 args
      7. 返回 TrainRunner(args, train_env, eval_env)

    Parameters
    ----------
    args : Config
        全局超参数对象（来自 configs/default_config.py）。
    seed : int
        随机种子，默认 0。
    env_name : str
        环境名称，用于 TensorBoard 日志目录命名，默认 "EnergyStorageEnv"。
    number : int
        实验编号，默认 1。

    Returns
    -------
    TrainRunner
        已完成初始化（env、agent、buffer、writer）的训练 runner。
    """
    resolve_algorithm_model_config(args)

    # 1. 数据集
    train_dataset = build_dataset(args, mode="train")
    eval_dataset  = build_dataset(args, mode="test")

    # 2. 奖励函数（train / eval 共享同一实例）
    reward_type = getattr(args, "reward_type", "composite")
    reward_fn = get_reward_fn(reward_type, args)

    # 3. 训练 env 工厂（每次调用产生独立 forecaster + obs_builder）
    def make_train_env():
        return EnergyStorageEnv(
            args,
            mode="train",
            dataset=train_dataset,
            reward_fn=reward_fn,
            forecaster=build_forecaster(args),
            obs_builder=build_obs_builder(args),
        )

    # 4. 并行训练环境
    train_env = DummyVecEnv(args.num_envs, make_train_env)

    # 5. 评估环境
    eval_env = EnergyStorageEnv(
        args,
        mode="test",
        dataset=eval_dataset,
        reward_fn=reward_fn,
        forecaster=build_forecaster(args),
        obs_builder=build_obs_builder(args),
    )

    # 6. 将维度信息写入 args（供 agent 构建网络用）
    args.N = args.num_agents
    args.obs_dim_n    = [eval_env.observation_space[i].shape[0] for i in range(args.N)]
    args.action_dim_n = [eval_env.action_space[i].shape[0]      for i in range(args.N)]

    # 7. 构建 runner
    return TrainRunner(
        args,
        train_env=train_env,
        eval_env=eval_env,
        env_name=env_name,
        number=number,
        seed=seed,
    )
