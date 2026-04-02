from __future__ import annotations

from copy import deepcopy

from scripts.utils.madrl_shared_data import build_shared_data_status_summary, ensure_madrl_shared_data
from tests.support.helpers import make_smoke_config


def test_shared_data_signature_normalizes_float_fields(tmp_path) -> None:
    cfg_a = make_smoke_config(tmp_path / "a", algorithm="MATD3")
    cfg_b = make_smoke_config(tmp_path / "b", algorithm="MATD3")

    cfg_a.data.load_scale = [0.3, 0.30000000000000004]
    cfg_b.data.load_scale = [0.30000000000000004, 0.3]
    cfg_a.data.pv_scale = [1.0, 1.0]
    cfg_b.data.pv_scale = [1.0000000000000002, 0.9999999999999999]

    result_a = ensure_madrl_shared_data(cfg_a, root=tmp_path / "shared")
    result_b = ensure_madrl_shared_data(cfg_b, root=tmp_path / "shared")

    assert result_a.signature_hash == result_b.signature_hash
    assert result_a.shared_data_dir == result_b.shared_data_dir


def test_shared_data_signature_ignores_obs_fields_that_do_not_change_payload(tmp_path) -> None:
    cfg_base = make_smoke_config(tmp_path / "base", algorithm="MATD3")
    cfg_variant = deepcopy(cfg_base)
    cfg_variant.obs.adjacency_type = "ring"
    cfg_variant.obs.local_features = ["calendar_time", "soc", "time"]

    base_result = ensure_madrl_shared_data(cfg_base, root=tmp_path / "shared")
    variant_result = ensure_madrl_shared_data(cfg_variant, root=tmp_path / "shared")

    assert base_result.signature_hash == variant_result.signature_hash
    assert base_result.shared_data_dir == variant_result.shared_data_dir


def test_ensure_madrl_shared_data_reuses_existing_directory(tmp_path) -> None:
    cfg = make_smoke_config(tmp_path / "case", algorithm="MATD3")

    first = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")
    second = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    assert first.reused is False
    assert second.reused is True
    assert first.signature_hash == second.signature_hash
    assert first.shared_data_dir == second.shared_data_dir


def test_ensure_madrl_shared_data_changes_when_test_window_changes(tmp_path) -> None:
    cfg = make_smoke_config(tmp_path / "case", algorithm="MATD3")
    first = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    cfg.data.test_start_date = "2020-01-01"
    cfg.data.test_end_date = "2020-01-01"
    second = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    assert first.signature_hash != second.signature_hash
    assert first.shared_data_dir != second.shared_data_dir


def test_build_shared_data_status_summary_reports_reuse_and_generation_states(tmp_path) -> None:
    cfg = make_smoke_config(tmp_path / "case", algorithm="MATD3")

    created = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")
    reused = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    created_summary = build_shared_data_status_summary(
        created,
        test_start_date="2020-01-01",
        test_end_date="2020-01-02",
    )
    reused_summary = build_shared_data_status_summary(
        reused,
        test_start_date="2020-01-01",
        test_end_date="2020-01-02",
    )

    assert created_summary["shared_data_status"] == "generated_new_shared_data"
    assert created_summary["shared_data_reused"] is False
    assert "Generated a new shared MADRL data package" in created_summary["shared_data_message"]
    assert reused_summary["shared_data_status"] == "reused_existing_shared_data"
    assert reused_summary["shared_data_reused"] is True
    assert "Reused existing shared MADRL data package" in reused_summary["shared_data_message"]
