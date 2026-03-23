from scripts.utils.experiment_notebook_utils import (
    get_lstm_artifact_root,
    sanity_check_runner,
)
from tests.support.helpers import make_case_dir, make_smoke_config


def test_sanity_check_runner_reports_core_shapes(tmp_path):
    case_dir = make_case_dir(tmp_path, "sanity_runner")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    summary = sanity_check_runner(cfg, seed=0)

    assert "observation_schema" in summary
    assert "rollout_obs_shapes" in summary
    assert summary["action_dim"] == 1


def test_notebook_utils_import_smoke_uses_common_location(tmp_path):
    case_dir = make_case_dir(tmp_path, "notebook_utils_import")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    artifact_root = get_lstm_artifact_root(case_dir)

    assert artifact_root.name == "lstm"
    assert cfg.runtime.device.type == "cpu"
