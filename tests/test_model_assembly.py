from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controllers.action_feasibility import (
    build_safety_local_numpy,
    enforce_local_action_feasibility_torch,
    validate_executed_actions_numpy,
)
from controllers.madrl_controller import MADRLController
from controllers.madrl.registry import get_agent_cls
from models.assembly import build_actor_network, build_critic_network, validate_and_finalize_model_config
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


def test_local_feasibility_clamps_round_trip_residual_before_validation():
    safety_local_np = build_safety_local_numpy(
        soc=np.asarray([0.05, 0.50], dtype=np.float32),
        load_raw=np.asarray([1.0, 1.2], dtype=np.float32),
        pv_raw=np.asarray([0.0, 0.0], dtype=np.float32),
        battery_capacity_kwh=np.asarray([5.0, 6.0], dtype=np.float32),
        p_max_kw=np.asarray([2.5, 3.0], dtype=np.float32),
    )
    safety_local_t = torch.as_tensor(safety_local_np, dtype=torch.float32).unsqueeze(0)
    projected_action_t = torch.tensor(
        [[[-1.0e-4 / 2.5, 1.0], [0.25, -0.25]]],
        dtype=torch.float32,
    )

    executed_action_t, residual_info = enforce_local_action_feasibility_torch(
        safety_local_t,
        projected_action_t,
        efficiency=0.95,
        dt_hours=1.0,
        soc_min=0.05,
        soc_max=0.95,
    )

    validate_executed_actions_numpy(
        safety_local_np,
        executed_action_t.squeeze(0).cpu().numpy(),
        efficiency=0.95,
        dt_hours=1.0,
        soc_min=0.05,
        soc_max=0.95,
    )
    assert executed_action_t[0, 0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert residual_info["soc_penalty_unweighted"][0, 0].item() > 0.0


def test_madrl_controller_projection_path_preserves_raw_request_and_uses_projection_residual():
    cfg = SimpleNamespace(env=SimpleNamespace(efficiency=0.95, dt=1.0, soc_min=0.05, soc_max=0.95))
    dummy_agents = [
        SimpleNamespace(cfg=cfg, device=torch.device("cpu")),
        SimpleNamespace(cfg=cfg, device=torch.device("cpu")),
    ]

    class _RoundTripProjector:
        def project_actions_from_safety_local(self, safety_local, action_t):
            projected = action_t.clone()
            p_max_kw = torch.clamp(safety_local[..., 4], min=1e-6)
            projected[:, 0, 0] = -1.0e-4 / p_max_kw[:, 0]
            return projected

    controller = MADRLController(dummy_agents, projector=_RoundTripProjector())
    obs = {
        "local": np.zeros((2, 1), dtype=np.float32),
        "safety_local": build_safety_local_numpy(
            soc=np.asarray([0.05, 0.50], dtype=np.float32),
            load_raw=np.asarray([1.0, 1.2], dtype=np.float32),
            pv_raw=np.asarray([0.0, 0.0], dtype=np.float32),
            battery_capacity_kwh=np.asarray([5.0, 6.0], dtype=np.float32),
            p_max_kw=np.asarray([2.5, 3.0], dtype=np.float32),
        ),
    }
    raw_actions = [
        np.asarray([-1.0, 0.2], dtype=np.float32),
        np.asarray([0.25, -0.2], dtype=np.float32),
    ]

    executed_actions, action_info = controller._postprocess_joint_actions(obs, raw_actions)

    assert controller.apply_action_penalty is True
    validate_executed_actions_numpy(
        obs["safety_local"],
        np.stack(executed_actions, axis=0),
        efficiency=0.95,
        dt_hours=1.0,
        soc_min=0.05,
        soc_max=0.95,
    )
    assert action_info is not None
    assert action_info["battery_action_req"][0] == pytest.approx(-1.0)
    assert action_info["battery_action_exec"][0] == pytest.approx(0.0, abs=1e-7)
    assert action_info["battery_power_req_kw"][0] == pytest.approx(-2.5)
    assert action_info["battery_power_exec_kw"][0] == pytest.approx(0.0, abs=1e-7)
    assert action_info["controller_action_gap"][0] > 0.9
    assert action_info["soc_penalty_unweighted"][0] == pytest.approx(4.0e-5, rel=1e-2, abs=1e-6)
    assert action_info["action_penalty_unweighted"][0] == pytest.approx(
        action_info["soc_penalty_unweighted"][0],
        abs=1e-8,
    )
