"""Notebook experiment utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch

from configs.profiles import print_experiment_summary
from controllers import ClassicDRLController, MADRLController, MPCController, ZeroController
from controllers.madrl.registry import get_agent_cls
from predictors.artifacts import get_default_lstm_artifact_dir
from scripts.builder import build_env, build_train_runner
from scripts.checkpoints import resolve_checkpoint_to_load
from scripts.evaluate import evaluate_controller
from scripts.utils.nested import add_batch_dim, to_torch_nested
from scripts.utils.project_paths import get_checkpoint_root, get_forecast_artifact_root, project_root
from scripts.utils.torch_runtime import resolve_device


def summarize_cfg(cfg) -> dict:
    """Print and return a compact experiment summary."""
    return print_experiment_summary(cfg)


def get_lstm_artifact_root(root=None) -> Path:
    """Return the default LSTM artifact root for notebooks."""
    if root is None:
        return get_default_lstm_artifact_dir()
    return get_forecast_artifact_root(root) / "lstm"


def get_madrl_checkpoint_root(root=None) -> Path:
    """Return the default checkpoint root used by training notebooks."""
    return get_checkpoint_root(root)


def build_runner(cfg, seed: int = 0, env_name: str = "NotebookTrain", number: int = 1):
    """Build a TrainRunner for notebook use."""
    return build_train_runner(cfg, seed=seed, env_name=env_name, number=number)


def inspect_runner_io(runner, cfg) -> dict:
    """Inspect runner rollout, actor, and critic interfaces."""
    rollout = runner.rollout_once()
    eval_obs, reset_info = runner.env_evaluate.reset(episode_idx=0)
    del reset_info
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
        "terminated_shape": tuple(rollout["terminated"].shape),
        "truncated_shape": tuple(rollout["truncated"].shape),
        "info_keys": sorted(rollout["info_list"][0].keys()),
        "actor_action_shape": tuple(actor_action.shape),
        "critic_shape": tuple(
            critic_out[0].shape if isinstance(critic_out, tuple) else critic_out.shape
        ),
    }
    pprint(summary)
    return summary


def sanity_check_runner(cfg, seed: int = 0) -> dict:
    """Build a temporary runner and run a one-step interface check."""
    sanity_cfg = deepcopy(cfg)
    runner = build_runner(sanity_cfg, seed=seed, env_name="NotebookSanity", number=1)
    try:
        return inspect_runner_io(runner, sanity_cfg)
    finally:
        runner.close()


def evaluate_runner(runner, cfg, n_episodes: int = 1, deterministic: bool = True) -> dict:
    """Turn a trained runner into a controller and evaluate it."""
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
    """Load a MADRL controller from checkpoints."""
    load_cfg = deepcopy(cfg)
    load_cfg.algo.name = algorithm or load_cfg.algo.name
    if device is not None:
        load_cfg.runtime.device = resolve_device(device)

    env = build_env(load_cfg, mode="test")
    try:
        load_cfg.runtime.observation_schema = dict(env.observation_schema)
        load_cfg.runtime.observation_layout = dict(env.observation_layout)
        load_cfg.runtime.action_dim = int(env.action_space[0].shape[0])

        checkpoint_info = resolve_checkpoint_to_load(
            model_root,
            load_cfg.algo.name,
            episode_tag=episode_tag,
        )
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
    """Build controller factories and metadata for comparison notebooks."""
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
    "get_lstm_artifact_root",
    "get_madrl_checkpoint_root",
    "inspect_runner_io",
    "load_madrl_controller",
    "project_root",
    "sanity_check_runner",
    "summarize_cfg",
]
