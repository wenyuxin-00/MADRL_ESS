from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.utils.grid_notebook_workflow import RolloutResult, load_rollout_record, save_rollout_record


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_compare_notebook_code_cells_compile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "compare.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 2
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_compare_notebook_reads_saved_records_instead_of_recomputing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "compare.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    required_tokens = [
        "COMPARE_SCHEME_ORDER",
        "RECORD_SCHEME_CATEGORIES",
        "load_rollout_record(",
        "compare_rollout_metrics(*rollouts)",
        "build_compare_economic_table",
        "build_compare_safety_table",
    ]
    forbidden_tokens = [
        "collect_global_full_horizon_rollout",
        "collect_local_mpc_rollout",
        "collect_admm_mpc_rollout",
        "collect_madrl_rollout",
        "run_external_train_mainline",
        "ensure_madrl_shared_data",
        "resolve_madrl_model_root",
    ]

    for token in required_tokens:
        assert token in joined_source
    for token in forbidden_tokens:
        assert token not in joined_source


def _empty_madrl_rollout(*, soc_mode: str) -> RolloutResult:
    return RolloutResult(
        step_df=pd.DataFrame(),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "DRL (forecast_eval)",
            "soc_mode": soc_mode,
            "dt_hours": 0.25,
            "import_price_markup_eur_per_kwh": 0.0,
            "selected_episode_start_timestamps": ["2020-01-01T00:00:00"],
            "selected_episode_end_timestamps": ["2020-01-01T23:45:00"],
        },
    )


def test_compare_rejects_madrl_reset_record(tmp_path) -> None:
    save_rollout_record(_empty_madrl_rollout(soc_mode="reset"), category="madrl", scheme_name="madrl_base", root=tmp_path)

    with pytest.raises(ValueError, match="soc_mode='continuous'"):
        load_rollout_record(category="madrl", scheme_name="madrl_base", root=tmp_path)


def test_compare_accepts_madrl_continuous_record(tmp_path) -> None:
    save_rollout_record(
        _empty_madrl_rollout(soc_mode="continuous"),
        category="madrl",
        scheme_name="madrl_base",
        root=tmp_path,
    )

    loaded = load_rollout_record(category="madrl", scheme_name="madrl_base", root=tmp_path)

    assert loaded.meta["soc_mode"] == "continuous"


def test_collect_madrl_rollout_uses_continuous_soc(monkeypatch) -> None:
    from scripts.mainline_compare import collect_madrl_rollout

    cfg = SimpleNamespace(
        forecast=SimpleNamespace(type="lstm"),
        runtime=SimpleNamespace(device="cpu"),
    )
    loaded_cfg = SimpleNamespace()
    expected_controller = object()
    observed: dict[str, object] = {}

    def _fake_load_madrl_controller(*args, **kwargs):
        observed["load_kwargs"] = dict(kwargs)
        return {"cfg": loaded_cfg, "controller": expected_controller}

    def _fake_collect_controller_rollout(local_cfg, *, label, controller=None, episode_indices=None, soc_mode="reset"):
        observed["soc_mode"] = soc_mode
        observed["label"] = label
        observed["episode_indices"] = episode_indices
        assert local_cfg is loaded_cfg
        assert controller is expected_controller
        return _empty_madrl_rollout(soc_mode=soc_mode)

    monkeypatch.setattr("scripts.mainline_madrl.load_madrl_controller", _fake_load_madrl_controller)
    monkeypatch.setattr("scripts.utils.grid_notebook_workflow.collect_controller_rollout", _fake_collect_controller_rollout)

    rollout = collect_madrl_rollout(cfg, episode_indices=[0, 1])

    assert observed["soc_mode"] == "continuous"
    assert observed["episode_indices"] == [0, 1]
    assert rollout.meta["soc_mode"] == "continuous"
