from controllers import ClassicDRLController, MPCController, ZeroController
from core.builder import build_env
from evaluation import comparison_records_to_rows, evaluate_controller_suite
from tests.helpers import make_case_dir, make_smoke_config


def test_compare_suite_marks_placeholders_as_not_implemented(tmp_path):
    case_dir = make_case_dir(tmp_path, "compare_suite")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    records = evaluate_controller_suite(
        env_factory=lambda: build_env(cfg, mode="test"),
        controller_builders={
            "zero": lambda: ZeroController(action_dim_n=[1] * cfg.env.num_agents),
            "mpc": lambda: MPCController(),
            "classic_drl": lambda: ClassicDRLController(),
        },
        n_episodes=1,
        deterministic=True,
    )

    rows = comparison_records_to_rows(records)
    status_map = {row["controller"]: row["status"] for row in rows}

    assert status_map["zero"] == "ok"
    assert status_map["mpc"] == "not_implemented"
    assert status_map["classic_drl"] == "not_implemented"
