"""Notebook experiment utilities."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import torch

from configs import print_experiment_summary
from controllers import MADRLController
from controllers.madrl.registry import get_agent_cls
from predictors.artifacts import get_default_lstm_artifact_dir
from scripts.builder import build_env, build_train_runner
from scripts.checkpoints import (
    build_training_run_paths,
    find_latest_training_run,
    resolve_checkpoint_to_load,
)
from scripts.evaluate import evaluate_controller
from scripts.utils.nested import add_batch_dim, to_torch_nested
from scripts.utils.project_paths import (
    get_checkpoint_root,
    get_forecast_artifact_root,
    project_root,
)
from scripts.utils.torch_runtime import resolve_device


def summarize_cfg(cfg) -> dict:
    return print_experiment_summary(cfg)


def get_lstm_artifact_root(root=None) -> Path:
    if root is None:
        return get_default_lstm_artifact_dir()
    return get_forecast_artifact_root(root) / "lstm"


def get_madrl_checkpoint_root(root=None) -> Path:
    return get_checkpoint_root(root)


def prepare_madrl_run_paths(
    *,
    root=None,
    checkpoint_root=None,
    algorithm: str,
    prediction_mode: str,
    experiment_name: str,
    train_episodes: int | None,
    max_train_steps: int | None,
):
    resolved_checkpoint_root = (
        Path(checkpoint_root).resolve()
        if checkpoint_root is not None
        else get_madrl_checkpoint_root(root).resolve()
    )
    return build_training_run_paths(
        resolved_checkpoint_root,
        algorithm=algorithm,
        prediction_mode=prediction_mode,
        experiment_name=experiment_name,
        train_episodes=train_episodes,
        max_train_steps=max_train_steps,
    )


def resolve_madrl_model_root(
    *,
    algorithm: str,
    prediction_mode: str,
    experiment_name: str,
    model_root=None,
    root=None,
    checkpoint_root=None,
) -> Path:
    if model_root is not None:
        candidate = Path(model_root).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Requested model_root does not exist: '{candidate}'")
        if (candidate / algorithm).exists():
            return candidate
        if candidate.name == algorithm and any(candidate.glob("actor_agent_*_ep_*.pth")):
            return candidate.parent
        raise FileNotFoundError(
            "model_root should point to a run directory that contains the algorithm subdirectory "
            f"or directly to that algorithm subdirectory. Got '{candidate}'."
        )

    resolved_checkpoint_root = (
        Path(checkpoint_root).resolve()
        if checkpoint_root is not None
        else get_madrl_checkpoint_root(root).resolve()
    )
    return find_latest_training_run(
        resolved_checkpoint_root,
        algorithm=algorithm,
        prediction_mode=prediction_mode,
        experiment_name=experiment_name,
    )


def write_notebook_run_metadata(
    *,
    meta_dir,
    experiment_controls: dict,
    data_controls: dict,
    train_controls: dict,
    checkpoint_controls: dict,
    result_payload: dict,
) -> dict[str, Path]:
    meta_dir = Path(meta_dir).resolve()
    meta_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "experiment_controls.json": experiment_controls,
        "data_controls.json": data_controls,
        "train_controls.json": train_controls,
        "checkpoint_controls.json": checkpoint_controls,
        "train_result.json": result_payload,
    }
    written_paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        target_path = meta_dir / filename
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written_paths[filename] = target_path
    return written_paths


def build_runner(cfg, seed: int = 0, env_name: str = "NotebookTrain", number: int = 1):
    return build_train_runner(cfg, seed=seed, env_name=env_name, number=number)


def inspect_runner_io(runner, cfg) -> dict:
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
    sanity_cfg = deepcopy(cfg)
    runner = build_runner(sanity_cfg, seed=seed, env_name="NotebookSanity", number=1)
    try:
        return inspect_runner_io(runner, sanity_cfg)
    finally:
        runner.close()


def evaluate_runner(runner, cfg, n_episodes: int = 1, deterministic: bool = True) -> dict:
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
    model_root=None,
    *,
    algorithm: str | None = None,
    episode_tag: int | None = None,
    device=None,
    prediction_mode: str | None = None,
    experiment_name: str = "grid_mainline",
    checkpoint_root=None,
    root=None,
):
    load_cfg = deepcopy(cfg)
    load_cfg.algo.name = algorithm or load_cfg.algo.name
    if device is not None:
        load_cfg.runtime.device = resolve_device(device)

    resolved_prediction_mode = prediction_mode or (
        "perfect" if str(load_cfg.forecast.type).strip().lower() == "perfect" else "normal"
    )
    resolved_model_root = resolve_madrl_model_root(
        algorithm=load_cfg.algo.name,
        prediction_mode=resolved_prediction_mode,
        experiment_name=experiment_name,
        model_root=model_root,
        root=root,
        checkpoint_root=checkpoint_root,
    )

    env = build_env(load_cfg, mode="test")
    try:
        load_cfg.runtime.observation_schema = dict(env.observation_schema)
        load_cfg.runtime.observation_layout = dict(env.observation_layout)
        load_cfg.runtime.action_dim = int(env.action_space[0].shape[0])

        checkpoint_info = resolve_checkpoint_to_load(
            resolved_model_root,
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
            "model_root": str(resolved_model_root),
        }
    finally:
        env.close()


__all__ = [
    "build_runner",
    "evaluate_runner",
    "get_lstm_artifact_root",
    "get_madrl_checkpoint_root",
    "inspect_runner_io",
    "load_madrl_controller",
    "prepare_madrl_run_paths",
    "project_root",
    "resolve_madrl_model_root",
    "sanity_check_runner",
    "summarize_cfg",
    "write_notebook_run_metadata",
]
