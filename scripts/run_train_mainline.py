"""CLI entrypoint for the Grid MADRL training mainline."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)

import torch

from configs import compose_experiment_config, print_experiment_summary
from controllers.madrl.safety_projector import is_safe_poc_algorithm
from scripts.builder import build_train_runner
from scripts.checkpoints import (
    build_training_run_paths,
    resolve_checkpoint_to_load,
    slugify_checkpoint_token,
)
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings, ensure_forecast_ready
from scripts.utils.project_paths import get_checkpoint_root, get_tensorboard_run_dir, project_root
from scripts.utils.torch_runtime import configure_torch_runtime, describe_device


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return target_path


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _apply_train_controls(cfg, train_controls: dict[str, Any]) -> None:
    field_names = (
        "train_episodes",
        "max_train_steps",
        "num_envs",
        "vec_env_type",
        "parallel_episode_sampling",
        "batch_size",
        "buffer_size",
        "update_interval",
        "updates_per_step",
        "actor_lr",
        "critic_lr",
        "noise_std_init",
        "noise_std_min",
        "noise_decay_steps",
        "use_noise_decay",
        "show_progress",
        "progress_postfix_interval",
        "progress_episode_interval",
        "progress_write_interval_seconds",
    )
    for field_name in field_names:
        if field_name in train_controls:
            setattr(cfg.train, field_name, train_controls[field_name])
    if "policy_update_freq" in train_controls:
        cfg.algo.policy_update_freq = int(train_controls["policy_update_freq"])


def _apply_model_controls(cfg, model_controls: dict[str, Any] | None) -> None:
    controls = dict(model_controls or {})
    if "hidden_dim" in controls:
        hidden_dim = int(controls["hidden_dim"])
        if hidden_dim <= 0:
            raise ValueError(f"model_controls.hidden_dim must be positive, got {hidden_dim}.")
        cfg.model.hidden_dim = hidden_dim


def _apply_runtime_controls(cfg, runtime_controls: dict[str, Any] | None) -> None:
    controls = dict(runtime_controls or {})
    bool_fields = (
        "pin_memory",
        "non_blocking_transfers",
        "enable_amp",
        "enable_compile",
        "compile_fullgraph",
        "compile_dynamic",
    )
    str_fields = ("amp_dtype", "compile_mode", "matmul_precision")
    deprecated_fields = (
        "forecast_" "data_source",
        "observation_" "cache_root",
        "observation_" "cache_batch_size",
        "refresh_" "observation_cache",
    )
    deprecated_hits = [field_name for field_name in deprecated_fields if field_name in controls]
    if deprecated_hits:
        joined = ", ".join(sorted(deprecated_hits))
        raise ValueError(
            "runtime_controls no longer supports legacy cache fields: "
            f"{joined}. Use shared_data_dir/shared_data_signature for precomputed data, "
            "or omit them to use the live forecaster path."
        )

    for field_name in bool_fields:
        if field_name in controls:
            setattr(cfg.runtime, field_name, bool(controls[field_name]))
    for field_name in str_fields:
        if field_name in controls:
            setattr(cfg.runtime, field_name, str(controls[field_name]))
    if "shared_data_dir" in controls:
        cfg.runtime.shared_data_dir = None if controls["shared_data_dir"] in (None, "") else str(
            Path(controls["shared_data_dir"]).resolve()
        )
    if "shared_data_signature" in controls:
        cfg.runtime.shared_data_signature = None if controls["shared_data_signature"] in (None, "") else str(
            controls["shared_data_signature"]
        )


def _apply_reward_controls(cfg, reward_controls: dict[str, Any] | None) -> None:
    controls = dict(reward_controls or {})
    removed_reward_keys = {
        "lambda_throughput": "Remove 'lambda_throughput'; it is no longer supported.",
        "w_action_pen": "Use 'w_soc_pen' instead of 'w_action_pen'.",
    }
    removed_hits = sorted(set(controls).intersection(removed_reward_keys))
    if removed_hits:
        details = " ".join(removed_reward_keys[key] for key in removed_hits)
        raise ValueError(f"Legacy reward_controls key(s) are no longer supported: {removed_hits}. {details}")
    _KNOWN_REWARD_KEYS = {
        "export_subsidy_eur_per_kwh",
        "import_price_markup_eur_per_kwh",
        "w_soc_pen",
        "w_voltage_pen",
        "w_line_pen",
        "w_trafo_pen",
    }
    unknown_keys = set(controls.keys()) - _KNOWN_REWARD_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown reward_controls key(s): {sorted(unknown_keys)}. "
            f"Supported keys: {sorted(_KNOWN_REWARD_KEYS)}."
        )
    if "w_soc_pen" in controls:
        cfg.reward.w_soc_pen = float(controls["w_soc_pen"])
    numeric_fields = (
        "export_subsidy_eur_per_kwh",
        "import_price_markup_eur_per_kwh",
        "w_voltage_pen",
        "w_line_pen",
        "w_trafo_pen",
    )
    for field_name in numeric_fields:
        if field_name in controls:
            setattr(cfg.reward, field_name, float(controls[field_name]))


def _apply_safety_controls(cfg, safety_controls: dict[str, Any] | None) -> None:
    controls = dict(safety_controls or {})
    numeric_fields = (
        "projection_iters",
        "voltage_margin_pu",
        "line_margin_pct",
        "trafo_margin_pct",
        "linearization_delta_kw",
    )
    bool_fields = ("record_diagnostics",)
    str_fields = ("projector_mode",)

    for field_name in numeric_fields:
        if field_name in controls:
            value = controls[field_name]
            casted = int(value) if field_name == "projection_iters" else float(value)
            setattr(cfg.safety, field_name, casted)
    for field_name in bool_fields:
        if field_name in controls:
            setattr(cfg.safety, field_name, bool(controls[field_name]))
    for field_name in str_fields:
        if field_name in controls:
            setattr(cfg.safety, field_name, str(controls[field_name]))

    cfg.safety.enabled = bool(is_safe_poc_algorithm(cfg) and controls.get("enabled", True))


def _serialize_safety_cfg(cfg) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(cfg.safety, "enabled", False)),
        "projector_mode": str(getattr(cfg.safety, "projector_mode", "joint_linearized")),
        "projection_iters": int(getattr(cfg.safety, "projection_iters", 6)),
        "voltage_margin_pu": float(getattr(cfg.safety, "voltage_margin_pu", 0.005)),
        "line_margin_pct": float(getattr(cfg.safety, "line_margin_pct", 5.0)),
        "trafo_margin_pct": float(getattr(cfg.safety, "trafo_margin_pct", 5.0)),
        "linearization_delta_kw": float(getattr(cfg.safety, "linearization_delta_kw", 0.25)),
        "record_diagnostics": bool(getattr(cfg.safety, "record_diagnostics", True)),
    }


def _public_vec_env_name(env: Any) -> str:
    return type(env).__name__


def _resolve_save_dir(
    *,
    args,
    experiment_controls: dict[str, Any],
    data_controls: dict[str, Any],
    train_controls: dict[str, Any],
    checkpoint_controls: dict[str, Any],
) -> Path:
    if args.save_dir is not None:
        return Path(args.save_dir).resolve()

    checkpoint_root = Path(
        checkpoint_controls.get("checkpoint_root") or get_checkpoint_root(project_root())
    ).resolve()
    algorithm = str(
        experiment_controls.get("algorithm")
        or ("MATD3" if str(train_controls.get("profile", "base")) == "gpu_fast" else "MADDPG")
    )
    prediction_mode = str(data_controls.get("prediction_mode", "perfect"))
    experiment_name = checkpoint_controls.get("experiment_name", "grid_mainline")
    run_paths = build_training_run_paths(
        checkpoint_root,
        algorithm=algorithm,
        prediction_mode=prediction_mode,
        experiment_name=experiment_name,
        train_episodes=train_controls.get("train_episodes"),
        max_train_steps=train_controls.get("max_train_steps"),
    )
    return Path(run_paths["model_root"]).resolve()


def _build_result_payload(
    *,
    cfg,
    experiment_controls: dict[str, Any],
    data_controls: dict[str, Any],
    battery_controls: dict[str, Any],
    train_controls: dict[str, Any],
    checkpoint_controls: dict[str, Any],
    runtime_state,
    forecast_ready,
    summary: dict[str, Any],
    runner,
    episodes_completed: int,
    save_dir: Path,
    meta_dir: Path,
    log_path: Path,
    reward_summary_path: Path,
    env_name: str,
    run_number: int,
    seed: int,
    applied_controls: dict[str, Any],
) -> dict[str, Any]:
    experiment_name = slugify_checkpoint_token(
        checkpoint_controls.get("experiment_name", save_dir.parent.name),
        default="grid_mainline",
    )
    run_metadata = dict(getattr(runner, "run_metadata", {}))
    return {
        "algorithm": str(cfg.algo.name),
        "prediction_mode": str(applied_controls["prediction_mode"]),
        "evaluation_mode": str(applied_controls["evaluation_mode"]),
        "experiment_name": experiment_name,
        "run_label": str(save_dir.name),
        "episodes_completed": int(episodes_completed),
        "saved_episode_tag": int(episodes_completed),
        "save_dir": str(save_dir),
        "model_root": str(save_dir),
        "meta_dir": str(meta_dir),
        "log_path": str(log_path),
        "reward_summary_path": str(reward_summary_path),
        "checkpoint_info": resolve_checkpoint_to_load(save_dir, cfg.algo.name, episode_tag=episodes_completed),
        "tensorboard_dir": str(
            get_tensorboard_run_dir(
                cfg.algo.name,
                env_name,
                run_number=run_number,
                seed=seed,
                root=project_root(),
            )
        ),
        "device": str(cfg.runtime.device),
        "vec_env": _public_vec_env_name(runner.env),
        "perf_summary": dict(runner.perf_summary),
        "shared_data_dir": getattr(cfg.runtime, "shared_data_dir", None),
        "shared_data_signature": getattr(cfg.runtime, "shared_data_signature", None),
        "started_at": run_metadata.get("started_at"),
        "finished_at": run_metadata.get("finished_at"),
        "elapsed_seconds": run_metadata.get("elapsed_seconds"),
        "estimated_end_time": run_metadata.get("estimated_end_time"),
        "summary": summary,
        "safety_summary": runner.build_safety_summary(),
        "device_info": describe_device(runtime_state),
        "experiment_controls": dict(experiment_controls),
        "data_controls": dict(data_controls),
        "battery_controls": dict(battery_controls),
        "train_controls": dict(train_controls),
        "checkpoint_controls": dict(checkpoint_controls),
        "safety_controls": _serialize_safety_cfg(cfg),
        "forecast_ready": forecast_ready,
        "pid": int(os.getpid()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Grid MADRL training mainline.")
    parser.add_argument("--experiment-controls", required=True)
    parser.add_argument("--data-controls", required=True)
    parser.add_argument("--battery-controls")
    parser.add_argument("--train-controls", required=True)
    parser.add_argument("--checkpoint-controls")
    parser.add_argument("--data-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--result-json")
    parser.add_argument("--env-name", default="GridTrainMainline")
    parser.add_argument("--run-number", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, RuntimeError):
        pass

    experiment_controls = _load_json(args.experiment_controls)
    data_controls = _load_json(args.data_controls)
    battery_controls = _load_json(args.battery_controls) if args.battery_controls else {}
    train_controls = _load_json(args.train_controls)
    checkpoint_controls = _load_json(args.checkpoint_controls) if args.checkpoint_controls else {}

    seed = int(experiment_controls.get("seed", 0))
    data_dir = Path(args.data_dir).resolve() if args.data_dir is not None else (project_root() / "data")
    save_dir = _resolve_save_dir(
        args=args,
        experiment_controls=experiment_controls,
        data_controls=data_controls,
        train_controls=train_controls,
        checkpoint_controls=checkpoint_controls,
    )
    meta_dir = (save_dir / "_meta").resolve()
    result_json = (
        Path(args.result_json).resolve()
        if args.result_json is not None
        else meta_dir / "train_result.json"
    )
    log_path = meta_dir / "train.log"
    progress_json = meta_dir / "progress.json"
    reward_summary_path = meta_dir / "train_reward_summary.json"
    meta_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_experiment_config(
        profile=str(train_controls.get("profile", "base")),
        algorithm=experiment_controls.get("algorithm"),
        model_family=str(train_controls.get("model_family", "mlp")),
        vec_env_type=train_controls.get("vec_env_type"),
        data_dir=data_dir,
        device=experiment_controls.get("device_request"),
        runtime_mode=str(experiment_controls.get("runtime_mode", "performance")),
        seed=seed,
        require_cuda=experiment_controls.get("require_cuda"),
    )
    _apply_model_controls(cfg, experiment_controls.get("model_controls"))
    _apply_runtime_controls(cfg, experiment_controls.get("runtime_controls"))
    _apply_reward_controls(cfg, experiment_controls.get("reward_controls"))
    _apply_safety_controls(cfg, experiment_controls.get("safety_controls"))
    agent_profiles = list(data_controls.get("agent_profiles", cfg.data.agent_profiles))
    n_requested_agents = len(agent_profiles)
    applied_controls = apply_notebook_experiment_settings(
        cfg,
        prediction_mode=str(data_controls.get("prediction_mode", "perfect")),
        test_start_date=data_controls.get("test_start_date"),
        test_end_date=data_controls.get("test_end_date"),
        agent_profiles=agent_profiles,
        agent_bus_ids=data_controls.get("agent_bus_ids"),
        load_scale=data_controls.get("load_scale", cfg.data.load_scale or [1.0] * n_requested_agents),
        pv_scale=data_controls.get("pv_scale", cfg.data.pv_scale or [1.0] * n_requested_agents),
        battery_controls=battery_controls,
        forecast_controls=experiment_controls.get("forecast_controls"),
        future_horizon=int(data_controls.get("future_horizon", cfg.env.future_horizon)),
        train_year=data_controls.get("train_year"),
        test_year=data_controls.get("test_year"),
    )
    _apply_train_controls(cfg, train_controls)
    cfg.train.noise_decay_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.runtime.progress_state_path = str(progress_json)

    runtime_state = configure_torch_runtime(
        cfg,
        device=experiment_controls.get("device_request"),
        seed=seed,
        require_cuda=experiment_controls.get("require_cuda"),
    )
    if getattr(cfg.runtime, "shared_data_dir", None):
        forecast_ready = None
    else:
        forecast_ready = ensure_forecast_ready(cfg)
    cfg.runtime.forecast_ready = forecast_ready

    summary = print_experiment_summary(cfg)
    summary["applied_controls"] = applied_controls
    summary["device_info"] = describe_device(runtime_state)
    summary["battery_controls"] = dict(battery_controls)
    summary["checkpoint_controls"] = dict(checkpoint_controls)
    summary["model_root"] = str(save_dir)
    summary["meta_dir"] = str(meta_dir)
    summary["log_path"] = str(log_path)
    summary["run_label"] = str(save_dir.name)
    summary["training_backend"] = "mainline"
    summary["shared_data_dir"] = getattr(cfg.runtime, "shared_data_dir", None)
    summary["shared_data_signature"] = getattr(cfg.runtime, "shared_data_signature", None)
    summary["safety"] = _serialize_safety_cfg(cfg)

    runner = build_train_runner(cfg, seed=seed, env_name=args.env_name, number=args.run_number)
    shared_data_meta = dict(getattr(runner, "shared_data_metadata", {}) or {})
    summary["shared_data_metadata"] = shared_data_meta
    episodes_completed = 0
    try:
        episodes_completed = runner.run()
        save_dir.mkdir(parents=True, exist_ok=True)
        runner.save_model(str(save_dir), episode=episodes_completed)
        reward_summary_payload = runner.build_reward_summary()
        _write_json(reward_summary_path, reward_summary_payload)
        result_payload = _build_result_payload(
            cfg=cfg,
            experiment_controls=experiment_controls,
            data_controls=data_controls,
            battery_controls=battery_controls,
            train_controls=train_controls,
            checkpoint_controls=checkpoint_controls,
            runtime_state=runtime_state,
            forecast_ready=forecast_ready,
            summary=summary,
            runner=runner,
            episodes_completed=episodes_completed,
            save_dir=save_dir,
            meta_dir=meta_dir,
            log_path=log_path,
            reward_summary_path=reward_summary_path,
            env_name=args.env_name,
            run_number=int(args.run_number),
            seed=seed,
            applied_controls=applied_controls,
        )
        result_payload["perf_summary"].update(
            {
                "shared_data_enabled": bool(getattr(cfg.runtime, "shared_data_dir", None)),
                "shared_data_dir": getattr(cfg.runtime, "shared_data_dir", None),
                "shared_data_signature": getattr(cfg.runtime, "shared_data_signature", None),
            }
        )
        _write_json(result_json, result_payload)
        print(json.dumps(result_payload, indent=2, default=_json_default))
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())


