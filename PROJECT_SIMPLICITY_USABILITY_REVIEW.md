# MADRL_ESS Code Review

Date: 2026-03-20

## Overview

This review focuses on two dimensions:

- Simplicity: whether the code structure stays easy to understand and maintain
- Usability: whether the project is easy to run, extend, and debug in day-to-day work

Scope reviewed:

- Project structure and README guidance
- GitNexus flow/context for core training, config, and evaluation paths
- Representative modules in `configs/`, `scripts/`, `envs/`, `models/`, `predictors/`, `controllers/`
- Existing tests across training, observation schema, model assembly, and grid environment integration

Overall judgment:

The architecture base is solid. For an RL project of this size, the modular split is reasonable rather than excessive. The main issues are not "too many files", but a few high-traffic paths with implicit side effects and duplicated environment logic. Those are the parts worth simplifying first.

## What Is Working Well

### 1. The project has a clear core entry path

The main experiment path is relatively concentrated:

- Config composition: `configs/profiles.py:170`
- Runner construction: `scripts/builder.py:140`
- Training loop: `scripts/train.py:137`

That is a good sign. It means the project already has a recognizable "happy path" instead of forcing users to stitch together random modules.

### 2. Extension points are designed consistently

The project uses a fairly uniform registry/builder style for:

- environments
- datasets
- observation builders
- forecasters
- model components

This is a good tradeoff for a research-oriented codebase. It makes experimentation easier without forcing hard-coded condition trees everywhere.

Representative files:

- `data/loaders/registry.py:16`
- `predictors/registry.py:25`
- `envs/observation/default_builder.py:14`
- `models/assembly.py:157`

### 3. Tests cover real integration paths

The project does not only test isolated helpers. It also tests practical behavior:

- canonical config behavior: `tests/test_canonical_config.py:8`
- structured observation outputs: `tests/test_observation_schema.py:21`
- model assembly forward passes: `tests/test_model_assembly.py:16`
- training smoke path: `tests/test_training_smoke.py:13`
- grid environment integration: `tests/test_grid_env.py:61`

That gives the codebase more safety than typical RL repositories.

## Main Issues

### 1. Environment logic is duplicated across `EnergyStorageEnv` and `GridEnv`

This is the clearest simplification opportunity in the whole project.

The following behaviors are implemented in both environments with very similar logic:

- signal canonicalization
- episode meta parsing
- storage parameter resolution
- episode loading
- signal access helpers
- future mean price calculation
- reset flow
- large parts of step preprocessing

Representative duplicated sections:

- `envs/hems_env.py:146`
- `envs/hems_env.py:166`
- `envs/hems_env.py:186`
- `envs/hems_env.py:195`
- `envs/hems_env.py:247`
- `envs/hems_env.py:265`
- `envs/hems_env.py:285`
- `envs/grid_env.py:158`
- `envs/grid_env.py:169`
- `envs/grid_env.py:181`
- `envs/grid_env.py:189`
- `envs/grid_env.py:244`
- `envs/grid_env.py:258`
- `envs/grid_env.py:281`

The code already acknowledges this through TODO comments:

- `envs/grid_env.py:39`
- `envs/grid_env.py:160`
- `envs/grid_env.py:299`

Why this matters:

- duplicated environment state logic increases maintenance cost
- behavior can drift subtly between HEMS and grid modes
- bug fixes will require changes in two places
- onboarding is harder because the true lifecycle is split across near-copies

Recommendation:

Extract a shared base environment or a shared episode/state engine. Keep only the domain-specific parts separate:

- HEMS-specific reward/state output
- GridCore interaction and grid-specific info fields

This should be the first simplification refactor.

### 2. Builder path has too many implicit side effects

The builder layer is convenient, but currently does more than its name suggests.

Examples:

- `build_env()` auto-builds dataset, reward function, forecaster, and observation builder: `scripts/builder.py:28`
- `_finalize_runtime_from_env()` mutates runtime state back into config: `scripts/builder.py:125`
- `build_train_runner()` both validates model config and may auto-train missing LSTM artifacts: `scripts/builder.py:140`
- `build_forecaster()` may trigger artifact preparation when using LSTM: `predictors/registry.py:57`

Why this matters:

- users cannot easily tell which calls are pure assembly vs resource preparation
- debugging becomes harder because config mutation happens implicitly
- notebook convenience leaks into production-style usage
- repeated validation/finalization blurs the lifecycle

Recommendation:

Split the current flow into explicit stages:

1. Resolve config
2. Prepare runtime resources
3. Build components
4. Start training or evaluation

Concretely, separating "prepare LSTM artifacts" from "build forecaster" would improve predictability immediately.

### 3. `TrainRunner.run()` is doing too much in one method

The training loop is functional, but it is too dense for a high-traffic method:

- action selection
- env stepping
- history recording
- replay buffer writes
- episode finalization
- noise scheduling
- parameter updates
- progress rendering
- performance accounting

Main method:

- `scripts/train.py:137`

Why this matters:

- harder to read and modify safely
- harder to profile or test stage-by-stage
- behavior changes tend to pile into one large loop

Recommendation:

Split the loop into a few internal methods, for example:

- `_collect_step()`
- `_finalize_finished_episodes()`
- `_maybe_train_step()`
- `_update_progress()`

This would improve readability without changing architecture.

### 4. Command-line usability is weak compared with notebook usability

The project is more notebook-friendly than CLI-friendly.

What exists:

- a small debug entry: `scripts/run_debug_training.py:13`

What is missing:

- one unified CLI for train/eval/compare/prepare workflows
- one consistent way to choose profile, algorithm, model family, and runtime mode from command line

Related workflow files:

- `scripts/run_debug_training.py:13`
- `scripts/evaluate.py:13`
- `scripts/comparison.py:15`

Why this matters:

- new users will understand the architecture from README, but still not have one obvious command surface
- repeated manual setup is likely in day-to-day use
- automation and reproducible runs are harder than they need to be

Recommendation:

Add a lightweight CLI wrapper with commands such as:

- `train`
- `eval`
- `compare`
- `prepare-forecast`

This is one of the highest usability wins for relatively low implementation effort.

### 5. Top-level controller API exposes placeholders too early

The project exports controller classes that are still placeholders:

- `controllers/mpc/mpc_controller.py:17`
- `controllers/drl/classic_drl_controller.py:16`

Tests explicitly validate that they raise `NotImplementedError`:

- `tests/test_controller_eval.py:73`
- `tests/test_compare_suite.py:6`

Why this matters:

- top-level exports imply ready-to-use surface area
- users may assume these controllers are usable because they are presented alongside working ones
- it weakens trust in the public API

Recommendation:

If these are intentionally future-facing, treat them as experimental:

- label them clearly in docs
- avoid presenting them as default options in CLI or examples
- consider hiding them from the most visible API surface until implemented

## Priority Order

If the goal is "simpler and easier to use" with the highest return, the order should be:

1. Deduplicate environment lifecycle/state logic
2. Separate build-time assembly from side-effectful preparation
3. Break up `TrainRunner.run()`
4. Add a unified CLI
5. Reposition placeholder controllers as experimental

## Suggested Refactor Plan

### Phase 1: Low-risk clarity improvements

- Add a thin CLI entry for train/eval/compare
- Mark placeholder controllers as experimental in docs and entrypoints
- Make builder responsibilities more explicit in naming and docstrings

### Phase 2: Medium-risk structural cleanup

- Extract shared env episode/state logic from `EnergyStorageEnv` and `GridEnv`
- Isolate grid-specific behavior behind a smaller override surface

### Phase 3: Training loop cleanup

- Split `TrainRunner.run()` into staged internal helpers
- Keep behavior identical while improving readability and testability

## Final Assessment

This is not a messy codebase. It is a reasonably disciplined research/engineering hybrid project with solid modular boundaries and practical tests.

The main improvement target is not "reduce module count". The real target is:

- reduce duplicated domain logic
- reduce hidden side effects in the assembly path
- create one obvious operational interface for users

If those are addressed, the project will feel much simpler without losing flexibility.
