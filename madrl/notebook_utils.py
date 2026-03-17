"""MADRL notebook 轻量工具。

这里只保留 notebook 直接需要的公共逻辑：
- 打印实验摘要
- 做 schema / rollout / actor-critic 前向检查
- 评估训练后的 controller
- 生成 compare notebook 的 controller builders
- 从 checkpoint 加载 MADRL controller
- 提供统一的 forecast artifact 路径
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch

from algorithms.registry import get_agent_cls
from common.nested import add_batch_dim, to_torch_nested
from common.project_paths import get_checkpoint_root, project_root
from configs.profiles import print_experiment_summary
from controllers import ClassicDRLController, MADRLController, MPCController, ZeroController
from core.builder import build_env, build_train_runner
from evaluation import evaluate_controller
from forecast.artifacts import get_default_lstm_artifact_paths
from runners.checkpoints import resolve_checkpoint_to_load


def summarize_cfg(cfg) -> dict:
    """打印并返回实验摘要。"""
    return print_experiment_summary(cfg)


def get_lstm_artifact_paths(root=None) -> dict:
    """返回统一的 LSTM artifact 三件套路径。"""
    return get_default_lstm_artifact_paths(root)


def get_madrl_checkpoint_root(root=None) -> Path:
    """Return the canonical checkpoint directory used by training notebooks."""
    return get_checkpoint_root(root)


def build_runner(cfg, seed: int = 0, env_name: str = "NotebookTrain", number: int = 1):
    """为 notebook 创建 runner。"""
    return build_train_runner(cfg, seed=seed, env_name=env_name, number=number)


def inspect_runner_io(runner, cfg) -> dict:
    """检查 runner 的 rollout、actor 和 critic 前向接口。"""
    rollout = runner.rollout_once(runner.env.reset())
    eval_obs = runner.env_evaluate.reset(episode_idx=0)
    eval_obs_t = to_torch_nested(add_batch_dim(eval_obs), cfg.runtime.device)

    with torch.no_grad():
        actor_action = torch.stack([agent.actor(eval_obs_t) for agent in runner.agent_n], dim=1)
        critic_out = runner.agent_n[0].critic(eval_obs_t, actor_action)

    summary = {
        "observation_schema": dict(cfg.runtime.observation_schema or {}),
        "observation_layout": dict(cfg.runtime.observation_layout or {}),
        "action_dim": int(cfg.runtime.action_dim),
        "rollout_obs_shapes": {key: tuple(value.shape) for key, value in rollout["obs"].items()},
        "next_obs_shapes": {key: tuple(value.shape) for key, value in rollout["next_obs"].items()},
        "action_batch_shape": tuple(rollout["action_batch"].shape),
        "reward_shape": tuple(rollout["reward"].shape),
        "done_shape": tuple(rollout["done"].shape),
        "info_keys": sorted(rollout["info_list"][0].keys()),
        "actor_action_shape": tuple(actor_action.shape),
        "critic_shape": tuple(
            critic_out[0].shape if isinstance(critic_out, tuple) else critic_out.shape
        ),
    }
    pprint(summary)
    return summary


def sanity_check_runner(cfg, seed: int = 0) -> dict:
    """创建临时 runner，完成一轮公共前向检查。"""
    sanity_cfg = deepcopy(cfg)
    runner = build_runner(sanity_cfg, seed=seed, env_name="NotebookSanity", number=1)
    try:
        return inspect_runner_io(runner, sanity_cfg)
    finally:
        runner.close()


def evaluate_runner(runner, cfg, n_episodes: int = 1, deterministic: bool = True) -> dict:
    """把训练后的 runner 转成 controller 并执行统一评估。"""
    eval_env = build_env(cfg, mode="test")
    controller = MADRLController(runner.agent_n, noise_std=runner.noise_std)
    try:
        return evaluate_controller(
            env=eval_env,
            controller=controller,
            n_episodes=n_episodes,
            deterministic=deterministic,
            record_history=True,
        )
    finally:
        eval_env.close()


def load_madrl_controller(
    cfg,
    model_root,
    *,
    algorithm: str | None = None,
    episode_tag: int | None = None,
    device=None,
):
    """从 checkpoint 加载 MADRL controller，供 compare notebook 使用。"""
    load_cfg = deepcopy(cfg)
    load_cfg.algo.name = algorithm or load_cfg.algo.name
    if device is not None:
        load_cfg.runtime.device = torch.device(device)

    env = build_env(load_cfg, mode="test")
    try:
        load_cfg.runtime.observation_schema = dict(env.observation_schema)
        load_cfg.runtime.observation_layout = dict(env.observation_layout)
        load_cfg.runtime.action_dim = int(env.action_space[0].shape[0])

        checkpoint_info = resolve_checkpoint_to_load(model_root, load_cfg.algo.name, episode_tag=episode_tag)
        agent_cls = get_agent_cls(load_cfg.algo.name)
        agents = [agent_cls(load_cfg, agent_id=i) for i in range(load_cfg.env.num_agents)]
        for agent in agents:
            agent.load_model(checkpoint_info["algo_dir"], checkpoint_info["saved_episode_tag"])

        controller = MADRLController(agents, noise_std=0.0)
        return {
            "controller": controller,
            "checkpoint_info": checkpoint_info,
            "cfg": load_cfg,
        }
    finally:
        env.close()


def build_compare_controller_builders(
    cfg,
    *,
    controllers_to_compare: list[str],
    model_root: str | Path,
    algorithm: str,
    episode_tag: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """为 compare notebook 生成 controller builders 与附加元信息。"""
    model_root = Path(model_root)
    controller_builders: dict[str, object] = {}
    metadata: dict[str, object] = {}

    if "madrl" in controllers_to_compare:
        checkpoint_info = resolve_checkpoint_to_load(model_root, algorithm, episode_tag=episode_tag)
        metadata["madrl_checkpoint_info"] = checkpoint_info
        saved_episode_tag = checkpoint_info["saved_episode_tag"]

        def build_madrl():
            return load_madrl_controller(
                cfg,
                model_root=model_root,
                algorithm=algorithm,
                episode_tag=saved_episode_tag,
            )["controller"]

        controller_builders["madrl"] = build_madrl

    if "zero" in controllers_to_compare:
        controller_builders["zero"] = lambda: ZeroController(action_dim_n=[1] * cfg.env.num_agents)
    if "mpc" in controllers_to_compare:
        controller_builders["mpc"] = lambda: MPCController()
    if "classic_drl" in controllers_to_compare:
        controller_builders["classic_drl"] = lambda: ClassicDRLController()

    return controller_builders, metadata


__all__ = [
    "build_compare_controller_builders",
    "build_runner",
    "evaluate_runner",
    "get_lstm_artifact_paths",
    "get_madrl_checkpoint_root",
    "inspect_runner_io",
    "load_madrl_controller",
    "project_root",
    "sanity_check_runner",
    "summarize_cfg",
]
