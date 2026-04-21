from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from controllers import MADRLController
from controllers.madrl.registry import get_agent_cls
from scripts.builder import build_env
from scripts.checkpoints import (
    find_latest_training_run,
    resolve_checkpoint_to_load,
)
from scripts.utils.project_paths import (
    get_checkpoint_root,
)
from scripts.utils.torch_runtime import resolve_device
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
        else get_checkpoint_root(root).resolve()
    )
    return find_latest_training_run(
        resolved_checkpoint_root,
        algorithm=algorithm,
        prediction_mode=prediction_mode,
        experiment_name=experiment_name,
    )

def _uses_precomputed_shared_data(cfg) -> bool:
    shared_data_dir = getattr(getattr(cfg, "runtime", None), "shared_data_dir", None)
    return shared_data_dir not in (None, "")

def _infer_checkpoint_action_dim(checkpoint_info: dict) -> int | None:
    algo_dir = Path(checkpoint_info["algo_dir"])
    saved_episode_tag = checkpoint_info["saved_episode_tag"]
    actor_path = algo_dir / f"actor_agent_0_ep_{saved_episode_tag}.pth"
    if not actor_path.exists():
        return None
    state_dict = torch.load(actor_path, map_location="cpu")
    bias = state_dict.get("head.fc.bias")
    if isinstance(bias, torch.Tensor) and bias.ndim == 1:
        return int(bias.shape[0])
    weight = state_dict.get("head.fc.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
        return int(weight.shape[0])
    return None

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
    from scripts.utils.grid_notebook_workflow import ensure_forecast_ready
    load_cfg = deepcopy(cfg)
    load_cfg.algo.name = algorithm or load_cfg.algo.name
    if device is not None:
        load_cfg.runtime.device = resolve_device(device)
    if _uses_precomputed_shared_data(load_cfg):
        load_cfg.runtime.forecast_ready = None
    else:
        load_cfg.runtime.forecast_ready = ensure_forecast_ready(load_cfg)

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
        checkpoint_info = resolve_checkpoint_to_load(
            resolved_model_root,
            load_cfg.algo.name,
            episode_tag=episode_tag,
        )
        env_action_dim = int(env.action_space[0].shape[0])
        checkpoint_action_dim = _infer_checkpoint_action_dim(checkpoint_info)
        load_cfg.runtime.action_dim = int(
            checkpoint_action_dim if checkpoint_action_dim is not None else env_action_dim
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
