import torch

from controllers.madrl.registry import get_agent_cls
from models import build_actor_network, build_critic_network, validate_and_finalize_model_config
from scripts.builder import build_env
from scripts.utils.nested import add_batch_dim, to_torch_nested
from tests.support.helpers import make_case_dir, make_smoke_config


def _prepare_runtime(cfg):
    env = build_env(cfg, mode="test")
    try:
        cfg.runtime.observation_schema = dict(env.observation_schema)
        cfg.runtime.observation_layout = dict(env.observation_layout)
        cfg.runtime.action_dim = int(env.action_space[0].shape[0])
        obs, _ = env.reset(episode_idx=0)
        return to_torch_nested(add_batch_dim(obs), cfg.runtime.device)
    finally:
        env.close()


def test_mlp_model_assembly_forward(tmp_path):
    case_dir = make_case_dir(tmp_path, "model_mlp")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    obs_batch = _prepare_runtime(cfg)
    validate_and_finalize_model_config(cfg)

    actor_n = [build_actor_network(cfg, agent_id) for agent_id in range(cfg.env.num_agents)]
    critic = build_critic_network(cfg)

    action = torch.stack([actor(obs_batch) for actor in actor_n], dim=1)
    q = critic(obs_batch, action)

    assert action.shape == (1, cfg.env.num_agents, cfg.runtime.action_dim)
    assert q.shape == (1, 1)


def test_transformer_model_assembly_forward(tmp_path):
    case_dir = make_case_dir(tmp_path, "model_transformer")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.model.family = "transformer"
    obs_batch = _prepare_runtime(cfg)
    validate_and_finalize_model_config(cfg)

    actor_n = [build_actor_network(cfg, agent_id) for agent_id in range(cfg.env.num_agents)]
    critic = build_critic_network(cfg)

    action = torch.stack([actor(obs_batch) for actor in actor_n], dim=1)
    q = critic(obs_batch, action)

    assert action.shape == (1, cfg.env.num_agents, cfg.runtime.action_dim)
    assert q.shape == (1, 1)


def test_graph_model_assembly_forward(tmp_path):
    case_dir = make_case_dir(tmp_path, "model_graph")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.model.family = "graph"
    obs_batch = _prepare_runtime(cfg)
    validate_and_finalize_model_config(cfg)

    actor_n = [build_actor_network(cfg, agent_id) for agent_id in range(cfg.env.num_agents)]
    critic = build_critic_network(cfg)

    action = torch.stack([actor(obs_batch) for actor in actor_n], dim=1)
    q = critic(obs_batch, action)

    assert action.shape == (1, cfg.env.num_agents, cfg.runtime.action_dim)
    assert q.shape == (1, 1)


def test_matd3_safe_poc_projector_clamps_joint_action(tmp_path):
    case_dir = make_case_dir(tmp_path, "model_safe_poc")
    cfg = make_smoke_config(case_dir, algorithm="MATD3_SAFE_POC")
    obs_batch = _prepare_runtime(cfg)
    validate_and_finalize_model_config(cfg)

    agent_cls = get_agent_cls("MATD3_SAFE_POC")
    agent = agent_cls(cfg, agent_id=0)
    raw_action = torch.full((1, cfg.env.num_agents, cfg.runtime.action_dim), 1.5, dtype=torch.float32)
    projected_action, diagnostics = agent.safety_projector.project_actions(
        obs_batch,
        raw_action,
        return_diagnostics=True,
    )

    assert "safety_local" in obs_batch
    assert projected_action.shape == raw_action.shape
    assert torch.all(projected_action <= 1.0 + 1e-6)
    assert torch.all(projected_action >= -1.0 - 1e-6)
    assert diagnostics["enabled"] is True

    direct_projected_action = agent.safety_projector.project_actions_from_safety_local(
        obs_batch["safety_local"],
        raw_action,
    )
    assert torch.allclose(projected_action, direct_projected_action, atol=1e-6)


def test_matd3_safe_poc_fast_projector_mode_supported(tmp_path):
    case_dir = make_case_dir(tmp_path, "model_safe_poc_fast")
    cfg = make_smoke_config(case_dir, algorithm="MATD3_SAFE_POC")
    cfg.safety.projector_mode = "joint_linearized_fast"
    cfg.safety.projection_iters = 3
    obs_batch = _prepare_runtime(cfg)
    validate_and_finalize_model_config(cfg)

    agent_cls = get_agent_cls("MATD3_SAFE_POC")
    agent = agent_cls(cfg, agent_id=0)
    raw_action = torch.full((1, cfg.env.num_agents, cfg.runtime.action_dim), 1.5, dtype=torch.float32)
    projected_action = agent.safety_projector.project_actions_from_safety_local(
        obs_batch["safety_local"],
        raw_action,
    )

    assert agent.safety_projector.projector_mode == "joint_linearized_fast"
    assert projected_action.shape == raw_action.shape
    assert torch.all(projected_action <= 1.0 + 1e-6)
    assert torch.all(projected_action >= -1.0 - 1e-6)
