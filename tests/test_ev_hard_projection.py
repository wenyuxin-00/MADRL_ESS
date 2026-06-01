from __future__ import annotations

from dataclasses import replace
import sys
import types

import numpy as np

from configs.cfg import Cfg

stub_data = types.ModuleType("data")
stub_share = types.ModuleType("data.share_data")
stub_share.ShareData = object
sys.modules["data"] = stub_data
sys.modules["data.share_data"] = stub_share

from envs.grid_env import _is_ev_emergency_step, apply_ev_emergency_charging, count_future_connected_steps_until_departure, project_ev_action_to_departure_soc


def _cfg() -> Cfg:
    cfg = Cfg()
    return replace(cfg, env=replace(cfg.env, ev_enabled=True, ev_departure_constraint_mode="hard", ev_hard_projection_enabled=True))


def _emergency_cfg() -> Cfg:
    cfg = Cfg()
    return replace(cfg, env=replace(cfg.env, ev_enabled=True, ev_departure_constraint_mode="emergency", ev_emergency_charging_enabled=True, ev_emergency_window_hours=1.0, ev_emergency_strategy="required_power"))


def test_future_connected_steps_cross_midnight() -> None:
    cfg = _cfg()
    assert count_future_connected_steps_until_departure(cfg, 72) == 51
    assert count_future_connected_steps_until_departure(cfg, 26) == 1
    assert count_future_connected_steps_until_departure(cfg, 27) == 0
    assert count_future_connected_steps_until_departure(cfg, 28) == 0
    assert count_future_connected_steps_until_departure(cfg, 40) == 0


def test_ev_hard_projection_no_change_when_soc_is_feasible() -> None:
    cfg = _cfg()
    n = int(cfg.env.num_agents)
    cap = np.asarray(cfg.env.ev_capacity_kwh, dtype=np.float32)
    pmax = np.asarray(cfg.env.ev_max_charge_kw, dtype=np.float32)
    rl = np.zeros((n,), dtype=np.float32)
    projected, gap, required, infeasible = project_ev_action_to_departure_soc(cfg, 72, np.full((n,), 0.90, dtype=np.float32), rl, cap, pmax)
    np.testing.assert_allclose(projected, rl)
    np.testing.assert_allclose(gap, 0.0)
    np.testing.assert_allclose(required, 0.0)
    assert not bool(np.any(infeasible))


def test_ev_hard_projection_raises_power_when_needed() -> None:
    cfg = _cfg()
    n = int(cfg.env.num_agents)
    cap = np.asarray(cfg.env.ev_capacity_kwh, dtype=np.float32)
    pmax = np.asarray(cfg.env.ev_max_charge_kw, dtype=np.float32)
    rl = np.zeros((n,), dtype=np.float32)
    projected, gap, required, infeasible = project_ev_action_to_departure_soc(cfg, 26, np.full((n,), 0.82, dtype=np.float32), rl, cap, pmax)
    assert bool(np.all(required > 0.0))
    assert bool(np.all(projected > rl))
    assert bool(np.all(gap > 0.0))
    assert not bool(np.any(infeasible))


def test_ev_hard_projection_last_step_and_departure_daytime_cases() -> None:
    cfg = _cfg()
    n = int(cfg.env.num_agents)
    cap = np.asarray(cfg.env.ev_capacity_kwh, dtype=np.float32)
    pmax = np.asarray(cfg.env.ev_max_charge_kw, dtype=np.float32)
    rl = np.zeros((n,), dtype=np.float32)
    projected, gap, required, infeasible = project_ev_action_to_departure_soc(cfg, 27, np.full((n,), 0.15, dtype=np.float32), rl, cap, pmax)
    assert bool(np.all(required > pmax))
    np.testing.assert_allclose(projected, pmax)
    assert bool(np.all(gap > 0.0))
    assert bool(np.all(infeasible))

    for step in (28, 40):
        projected, gap, required, infeasible = project_ev_action_to_departure_soc(cfg, step, np.full((n,), 0.15, dtype=np.float32), pmax, cap, pmax)
        np.testing.assert_allclose(projected, 0.0)
        np.testing.assert_allclose(gap, 0.0)
        np.testing.assert_allclose(required, 0.0)
        assert not bool(np.any(infeasible))


def test_ev_emergency_window_and_charging_rule() -> None:
    cfg = _emergency_cfg()
    n = int(cfg.env.num_agents)
    cap = np.asarray(cfg.env.ev_capacity_kwh, dtype=np.float32)
    pmax = np.asarray(cfg.env.ev_max_charge_kw, dtype=np.float32)
    rl = np.zeros((n,), dtype=np.float32)
    assert not _is_ev_emergency_step(cfg, 23)
    assert _is_ev_emergency_step(cfg, 24)
    assert _is_ev_emergency_step(cfg, 27)
    assert not _is_ev_emergency_step(cfg, 28)

    charge, added, required = apply_ev_emergency_charging(cfg, 23, np.full((n,), 0.15, dtype=np.float32), rl, cap, pmax)
    np.testing.assert_allclose(charge, rl)
    np.testing.assert_allclose(added, 0.0)
    np.testing.assert_allclose(required, 0.0)

    charge, added, required = apply_ev_emergency_charging(cfg, 24, np.full((n,), 0.15, dtype=np.float32), rl, cap, pmax)
    assert bool(np.all(required > 0.0))
    assert bool(np.all(charge >= rl))
    assert bool(np.all(added >= 0.0))

    charge, added, required = apply_ev_emergency_charging(cfg, 27, np.full((n,), 0.15, dtype=np.float32), rl, cap, pmax)
    np.testing.assert_allclose(charge, pmax)
    np.testing.assert_allclose(required, pmax)
    assert bool(np.all(added > 0.0))

    for step in (28, 40):
        charge, added, required = apply_ev_emergency_charging(cfg, step, np.full((n,), 0.15, dtype=np.float32), pmax, cap, pmax)
        np.testing.assert_allclose(charge, 0.0)
        np.testing.assert_allclose(added, 0.0)
        np.testing.assert_allclose(required, 0.0)

    charge, added, required = apply_ev_emergency_charging(cfg, 24, np.full((n,), 0.90, dtype=np.float32), rl, cap, pmax)
    np.testing.assert_allclose(charge, rl)
    np.testing.assert_allclose(added, 0.0)
    np.testing.assert_allclose(required, 0.0)
