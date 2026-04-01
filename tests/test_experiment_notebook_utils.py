from __future__ import annotations

from types import SimpleNamespace

from scripts.utils.experiment_notebook_utils import evaluate_runner, load_madrl_controller
from tests.support.helpers import make_smoke_config


class _FakeEnv:
    def __init__(self) -> None:
        self.observation_schema = {"node_features": (2, 4)}
        self.observation_layout = {"node_features": {"shape": (2, 4)}}
        self.action_space = [SimpleNamespace(shape=(2,))]
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeAgent:
    def __init__(self, cfg, agent_id: int) -> None:
        del cfg
        self.agent_id = int(agent_id)
        self.loaded = None

    def load_model(self, algo_dir, saved_episode_tag) -> None:
        self.loaded = (algo_dir, saved_episode_tag)


def test_evaluate_runner_skips_forecast_preflight_in_shared_data_mode(tmp_path, monkeypatch):
    cfg = make_smoke_config(tmp_path, algorithm="MADDPG")
    cfg.runtime.shared_data_dir = str(tmp_path / "shared_data")
    fake_env = _FakeEnv()
    runner = SimpleNamespace(agent_n=["a", "b"], noise_std=0.2)

    def _unexpected_forecast_ready(_cfg):
        raise AssertionError("ensure_forecast_ready should be skipped in shared-data mode")

    monkeypatch.setattr("scripts.utils.grid_notebook_workflow.ensure_forecast_ready", _unexpected_forecast_ready)
    monkeypatch.setattr("scripts.utils.experiment_notebook_utils.build_env", lambda _cfg, mode: fake_env)
    monkeypatch.setattr(
        "scripts.utils.experiment_notebook_utils.MADRLController",
        lambda agents, noise_std: SimpleNamespace(agents=agents, noise_std=noise_std),
    )
    monkeypatch.setattr(
        "scripts.utils.experiment_notebook_utils.evaluate_controller",
        lambda **kwargs: {"episodes": kwargs["n_episodes"], "deterministic": kwargs["deterministic"]},
    )

    result = evaluate_runner(runner, cfg, n_episodes=2, deterministic=False)

    assert result == {"episodes": 2, "deterministic": False}
    assert cfg.runtime.forecast_ready is None
    assert fake_env.closed is True


def test_load_madrl_controller_skips_forecast_preflight_in_shared_data_mode(tmp_path, monkeypatch):
    cfg = make_smoke_config(tmp_path, algorithm="MADDPG")
    cfg.runtime.shared_data_dir = str(tmp_path / "shared_data")
    fake_env = _FakeEnv()

    def _unexpected_forecast_ready(_cfg):
        raise AssertionError("ensure_forecast_ready should be skipped in shared-data mode")

    monkeypatch.setattr("scripts.utils.grid_notebook_workflow.ensure_forecast_ready", _unexpected_forecast_ready)
    monkeypatch.setattr("scripts.utils.experiment_notebook_utils.build_env", lambda _cfg, mode: fake_env)
    monkeypatch.setattr(
        "scripts.utils.experiment_notebook_utils.resolve_madrl_model_root",
        lambda **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        "scripts.utils.experiment_notebook_utils.resolve_checkpoint_to_load",
        lambda *args, **kwargs: {"algo_dir": tmp_path, "saved_episode_tag": 7},
    )
    monkeypatch.setattr("scripts.utils.experiment_notebook_utils.get_agent_cls", lambda name: _FakeAgent)
    monkeypatch.setattr(
        "scripts.utils.experiment_notebook_utils.MADRLController",
        lambda agents, noise_std: SimpleNamespace(agents=agents, noise_std=noise_std),
    )

    loaded = load_madrl_controller(
        cfg,
        model_root=tmp_path,
        prediction_mode="normal",
    )

    assert loaded["cfg"].runtime.forecast_ready is None
    assert loaded["cfg"].runtime.action_dim == 2
    assert len(loaded["controller"].agents) == cfg.env.num_agents
    assert fake_env.closed is True
