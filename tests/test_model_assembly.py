import torch

from common.nested import add_batch_dim, to_torch_nested
from core.builder import build_env
from models import build_actor_network, build_critic_network, validate_and_finalize_model_config
from tests.helpers import make_case_dir, make_smoke_config


def _prepare_runtime(cfg):
    env = build_env(cfg, mode="test")
    try:
        cfg.runtime.observation_schema = dict(env.observation_schema)
        cfg.runtime.observation_layout = dict(env.observation_layout)
        cfg.runtime.action_dim = int(env.action_space[0].shape[0])
        obs = env.reset(episode_idx=0)
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
