from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.utils.madrl_shared_data import (
    build_shared_data_status_summary,
    ensure_madrl_shared_data,
    select_shared_data_episode_indices,
)
from tests.support.helpers import make_smoke_config, write_prosumer_processed_dataset


def _make_multiday_cfg(tmp_path, *, evaluation_days: int = 5):
    cfg = make_smoke_config(tmp_path, algorithm="MATD3")
    cfg.env.episode_limit = 96
    cfg.env.future_horizon = 1
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    write_prosumer_processed_dataset(
        cfg.data.data_dir,
        agent_profiles=list(cfg.data.agent_profiles),
        train_year=2019,
        test_year=2020,
        train_steps=96 * evaluation_days,
        test_steps=96 * evaluation_days,
    )
    return cfg


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


def test_ensure_madrl_shared_data_cross_year_reuses_signature_when_test_window_changes(tmp_path) -> None:
    cfg = _make_multiday_cfg(tmp_path / "case", evaluation_days=5)
    first = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    cfg.data.test_start_date = "2020-01-02"
    cfg.data.test_end_date = "2020-01-04"
    second = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    assert first.signature_hash == second.signature_hash
    assert first.shared_data_dir == second.shared_data_dir
    assert first.manifest["data_controls"]["test_window_strategy"] == "full_year_runtime_slice"
    selected = select_shared_data_episode_indices(
        second.manifest["splits"]["test"],
        start_date="2020-01-02",
        end_date="2020-01-04",
    )
    assert selected == [1, 2, 3]


def test_ensure_madrl_shared_data_same_year_implicit_exclusion_changes_signature_with_test_window(tmp_path) -> None:
    cfg = _make_multiday_cfg(tmp_path / "same_year_implicit", evaluation_days=5)
    cfg.data.train_year = 2020
    cfg.data.test_year = 2020
    first = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    cfg.data.test_start_date = "2020-01-02"
    cfg.data.test_end_date = "2020-01-03"
    second = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    assert first.signature_hash != second.signature_hash
    assert first.shared_data_dir != second.shared_data_dir
    assert second.manifest["data_controls"]["test_window_strategy"] == "cfg_window"


def test_ensure_madrl_shared_data_same_year_explicit_train_range_reuses_signature_when_test_window_changes(tmp_path) -> None:
    cfg = _make_multiday_cfg(tmp_path / "same_year_explicit", evaluation_days=5)
    cfg.data.train_year = 2020
    cfg.data.test_year = 2020
    cfg.data.train_start_date = "2020-01-01"
    cfg.data.train_end_date = "2020-01-02"
    first = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    cfg.data.test_start_date = "2020-01-03"
    cfg.data.test_end_date = "2020-01-05"
    second = ensure_madrl_shared_data(cfg, root=tmp_path / "shared")

    assert first.signature_hash == second.signature_hash
    assert first.shared_data_dir == second.shared_data_dir
    assert second.manifest["data_controls"]["test_window_strategy"] == "full_year_runtime_slice"


def test_select_shared_data_episode_indices_requires_fully_contained_episodes() -> None:
    manifest = {
        "episodes": [
            {
                "episode_idx": 0,
                "first_local_date": "2020-01-01",
                "last_local_date": "2020-01-02",
            },
            {
                "episode_idx": 1,
                "first_local_date": "2020-01-02",
                "last_local_date": "2020-01-02",
            },
            {
                "episode_idx": 2,
                "first_local_date": "2020-01-03",
                "last_local_date": "2020-01-03",
            },
        ]
    }

    selected = select_shared_data_episode_indices(
        manifest,
        start_date="2020-01-02",
        end_date="2020-01-03",
    )

    assert selected == [1, 2]


def test_select_shared_data_episode_indices_raises_on_empty_selection() -> None:
    manifest = {
        "episodes": [
            {
                "episode_idx": 0,
                "first_local_date": "2020-01-01",
                "last_local_date": "2020-01-01",
            }
        ]
    }

    with pytest.raises(ValueError, match="No shared-data episodes fall fully inside"):
        select_shared_data_episode_indices(
            manifest,
            start_date="2020-01-02",
            end_date="2020-01-02",
        )


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
