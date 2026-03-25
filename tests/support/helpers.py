"""Factories and tiny datasets shared across tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import torch

from configs.experiment_config import ExperimentConfig
from data.loaders.constants import TZ_LOCAL

DEFAULT_TEST_BUSES = [10, 6, 12, 7, 8]
DEFAULT_PROSUMER_TEST_PROFILES = ["SFH12", "SFH14", "SFH16", "SFH18"]


def synthetic_prosumer_household_kw(agent_idx: int, step: int, *, year_offset: int = 0) -> float:
    return 1.0 + 0.2 * agent_idx + 0.05 * step + 0.3 * year_offset


def synthetic_prosumer_heatpump_kw(agent_idx: int, step: int, *, year_offset: int = 0) -> float:
    return 0.4 + 0.1 * agent_idx + 0.02 * step + 0.05 * year_offset


def synthetic_prosumer_ref_pv_kw(orientation: str, step: int, *, year_offset: int = 0) -> float:
    base = {
        "east": 0.25,
        "south": 0.50,
        "west": 0.20,
    }[orientation]
    slope = {
        "east": 0.01,
        "south": 0.02,
        "west": 0.015,
    }[orientation]
    return base + slope * step + 0.05 * year_offset


def synthetic_prosumer_price_eur_per_kwh(step: int, *, year_offset: int = 0) -> float:
    return 0.05 + 0.001 * step + 0.01 * year_offset


def write_dataset(path: str | Path, num_agents: int, total_steps: int) -> None:
    path = Path(path)
    fieldnames = ["segment_id", "price"]
    fieldnames += [f"load{i + 1}" for i in range(num_agents)]
    fieldnames += [f"pv{i + 1}" for i in range(num_agents)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(total_steps):
            row = {
                "segment_id": 0,
                "price": 10.0 + 0.1 * step,
            }
            solar_phase = max(0.0, 1.0 - abs((step % 24) - 12) / 12.0)
            for agent_id in range(num_agents):
                row[f"load{agent_id + 1}"] = 1.0 + 0.1 * agent_id + 0.01 * step
                row[f"pv{agent_id + 1}"] = 0.2 + 0.15 * agent_id + 0.5 * solar_phase
            writer.writerow(row)


def write_prosumer_processed_dataset(
    data_root: str | Path,
    *,
    agent_profiles: list[str] | None = None,
    train_year: int = 2019,
    test_year: int = 2020,
    train_steps: int = 12,
    test_steps: int = 12,
) -> Path:
    data_root = Path(data_root)
    prosumer_dir = data_root / "processed" / "prosumer"
    prosumer_dir.mkdir(parents=True, exist_ok=True)

    agent_profiles = list(agent_profiles or DEFAULT_PROSUMER_TEST_PROFILES)

    def build_year_frames(year: int, total_steps: int, year_offset: int) -> tuple[pd.DataFrame, ...]:
        timestamps_local = pd.date_range(
            start=f"{year}-01-01 00:00:00",
            periods=total_steps,
            freq="15min",
            tz=TZ_LOCAL,
        )
        timestamps_utc = timestamps_local.tz_convert("UTC")

        household_rows: list[dict[str, object]] = []
        heatpump_rows: list[dict[str, object]] = []
        pv_rows: list[dict[str, object]] = []
        price_rows: list[dict[str, object]] = []

        for step, timestamp in enumerate(timestamps_utc):
            timestamp_str = timestamp.isoformat()
            household_row: dict[str, object] = {"timestamp": timestamp_str}
            heatpump_row: dict[str, object] = {"timestamp": timestamp_str}
            for agent_idx, profile in enumerate(agent_profiles):
                household_row[profile] = synthetic_prosumer_household_kw(agent_idx, step, year_offset=year_offset)
                heatpump_row[profile] = synthetic_prosumer_heatpump_kw(agent_idx, step, year_offset=year_offset)
            household_rows.append(household_row)
            heatpump_rows.append(heatpump_row)
            pv_rows.append(
                {
                    "timestamp": timestamp_str,
                    "ref_east": synthetic_prosumer_ref_pv_kw("east", step, year_offset=year_offset),
                    "ref_south": synthetic_prosumer_ref_pv_kw("south", step, year_offset=year_offset),
                    "ref_west": synthetic_prosumer_ref_pv_kw("west", step, year_offset=year_offset),
                }
            )
            price_rows.append(
                {
                    "timestamp": timestamp_str,
                    "price": synthetic_prosumer_price_eur_per_kwh(step, year_offset=year_offset),
                }
            )

        return (
            pd.DataFrame(household_rows),
            pd.DataFrame(heatpump_rows),
            pd.DataFrame(pv_rows),
            pd.DataFrame(price_rows),
        )

    train_frames = build_year_frames(train_year, train_steps, year_offset=0)
    test_frames = build_year_frames(test_year, test_steps, year_offset=1)

    household = pd.concat([train_frames[0], test_frames[0]], ignore_index=True)
    heatpump = pd.concat([train_frames[1], test_frames[1]], ignore_index=True)
    pv_reference = pd.concat([train_frames[2], test_frames[2]], ignore_index=True)
    price = pd.concat([train_frames[3], test_frames[3]], ignore_index=True)

    household.to_csv(prosumer_dir / "household.csv", index=False)
    heatpump.to_csv(prosumer_dir / "heatpump.csv", index=False)
    pv_reference.to_csv(prosumer_dir / "pv_reference.csv", index=False)
    price.to_csv(prosumer_dir / "price.csv", index=False)
    return prosumer_dir


def make_smoke_config(tmp_path: str | Path, algorithm: str = "MADDPG") -> ExperimentConfig:
    tmp_path = Path(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_limit = 3
    total_steps = episode_limit * 4
    agent_profiles = ["SFH12", "SFH14"]
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=agent_profiles,
        train_steps=total_steps,
        test_steps=total_steps,
    )

    cfg = ExperimentConfig()
    cfg.algo.name = algorithm
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = list(agent_profiles)
    cfg.data.train_year = 2019
    cfg.data.test_year = 2020
    cfg.data.load_components = ["household", "heatpump"]
    cfg.data.pv_reference = "south"
    cfg.data.pv_capacity_kw = []
    cfg.reward.type = "grid_composite"
    cfg.obs.local_features = ["time", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.forecast.target_signals = ["price", "load", "pv"]
    cfg.env.num_agents = 2
    cfg.env.episode_limit = episode_limit
    cfg.env.future_horizon = 1
    cfg.grid.agent_bus_ids = DEFAULT_TEST_BUSES[: cfg.env.num_agents]
    cfg.train.train_episodes = 2
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
    cfg.train.batch_size = 2
    cfg.train.buffer_size = 32
    cfg.train.use_noise_decay = False
    cfg.model.hidden_dim = 16
    cfg.model.use_orthogonal_init = False
    cfg.runtime.device = torch.device("cpu")
    return cfg


def make_case_dir(tmp_path: str | Path, label: str) -> Path:
    _ = label
    return Path(tmp_path)
