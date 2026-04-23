# SOC Repair Plan

## Problem

2020-06-01 to 2020-06-07 MPC rollouts were not using the same SoC accounting as the Global MISOCP full-horizon run. Global MISOCP stitches the seven daily episodes into one continuous horizon, while Local MPC and ADMM MPC previously reset `GridEnv.soc` at every daily `env.reset()`. That gave MPC rollouts free cross-day energy and made the comparison unfair.

## Fix Plan

1. Keep `collect_controller_rollout(...)` default behavior as `soc_mode="reset"` so ordinary DRL daily evaluation remains unchanged.
2. Add an explicit `soc_mode="continuous"` rollout contract.
3. When continuous mode is requested, after each daily `env.reset(episode_idx=...)`, restore the previous episode's terminal SoC before building observations or asking the controller for actions.
4. Require continuous-mode episode indices to be sorted chronologically.
5. Record `global_step`, `soc_start`, and `soc_end` in MPC rollout tables so cross-day SoC continuity is visible.
6. Make Local MPC and ADMM MPC call `collect_controller_rollout(..., soc_mode="continuous")`.
7. Add regression coverage proving episode 1 starts from episode 0's ending SoC, not `cfg.env.init_soc`.
8. Regenerate 2020-06-01 to 2020-06-07 MPC records after the code fix.

## Impact Analysis

GitNexus impact was run before editing.

- `collect_controller_rollout`: HIGH risk. Direct callers are `collect_local_mpc_rollout`, `collect_admm_mpc_rollout`, `collect_madrl_rollout`, and rollout tests.
- `collect_local_mpc_rollout`: LOW risk. Directly covered by local MPC notebook workflow tests.
- `collect_admm_mpc_rollout`: LOW risk. Directly covered by ADMM MPC helper tests.
- `_build_rollout_records`: CRITICAL risk, so it was not edited. SoC boundary display fields are appended by `collect_controller_rollout` instead.

## Executed Results

Regenerated records under `notebooks/record/mpc/` for the canonical 2020-06-01 to 2020-06-07 test window:

- `admm_mpc_lstm`: `total_cost_eur = 634.935662`, `soc_mode = continuous`, final SoC `[0.275560, 0.382363, 0.381746]`.
- `local_mpc_lstm`: `total_cost_eur = 636.568857`, `soc_mode = continuous`, final SoC `[0.393698, 0.232754, 0.185030]`.
- `local_mpc_perfect`: `total_cost_eur = 606.893657`, `soc_mode = continuous`, final SoC `[0.050000, 0.050000, 0.050000]`.

ADMM boundary check after repair:

- day 0 end SoC `[0.354771, 0.401474, 0.463903]`
- day 1 start SoC `[0.354771, 0.401474, 0.463903]`
- day 5 end SoC `[0.458911, 0.517983, 0.275729]`
- day 6 start SoC `[0.458911, 0.517983, 0.275729]`

This confirms MPC no longer receives a daily SoC reset.

## Verification

- `python -m py_compile scripts/utils/grid_notebook_workflow.py scripts/mainline_compare.py scripts/utils/admm_mpc_notebook_helpers.py`
- `pytest tests/test_grid_notebook_workflow.py::test_collect_controller_rollout_can_carry_soc_across_episodes tests/test_grid_notebook_workflow.py::test_collect_local_mpc_rollout_preserves_interface_for_both_prediction_modes tests/test_grid_notebook_workflow.py::test_collect_local_mpc_rollout_rewrites_objective_to_economic_only tests/test_admm_mpc_notebook_helpers.py::test_collect_admm_mpc_rollout_sets_meta_and_step_diagnostics -q`

## Remaining Comparison Note

This repair fixes the cross-day SoC reset bug. It does not make Local MPC or ADMM MPC impose the same hard final `soc_target` equality as Global MISOCP. Local MPC perfect can still end at `soc_min`, so it may still be cheaper than Global MISOCP for a different reason: missing terminal target equality, not daily SoC reset.
