from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from configs.cfg import Cfg
from controllers.protocol import CONTROLLER_NAMES, Controller, build_controller
from data.share_data import ShareData
from envs.grid_env import build_env
from utils.records import controller_window_from_obs
from utils.run_artifacts import eval_result_exists, load_eval_metrics, save_eval_result, save_eval_summary


def eval_controller(cfg: Cfg, controller: Controller, forecast_mode: str = "perfect", n_episodes: int | None = None, share_data: ShareData | None = None) -> dict:
    env = build_env(cfg, "eval", forecast_mode=forecast_mode, share_data=share_data)
    n = int(cfg.eval.n_episodes if n_episodes is None else n_episodes)
    indices = list(range(int(share_data.eval["price"].shape[0]))) if cfg.eval.episode_indices is None else list(cfg.eval.episode_indices)
    rewards, violations, profits, min_vms, max_vms, trafo_maxes, act_times = [], [], [], [], [], [], []
    voltage_trace, trafo_trace, trafo_overload_steps = [], [], 0
    episodes = tqdm(indices[:n], desc=f"eval {controller.name}/{forecast_mode}", unit="episode", ascii=True)
    for episode in episodes:
        obs, state = env.reset(int(episode)); controller.reset(state)
        ep_r = ep_v = ep_p = 0.0; ep_min = 1.0; ep_max = 1.0; ep_trafo = 0.0
        for _ in tqdm(range(int(cfg.env.episode_steps)), desc=f"{controller.name} ep {episode}", unit="step", leave=False, ascii=True):
            start = time.perf_counter()
            if callable(controller):
                action, _, solve_meta = controller(env, controller_window_from_obs(obs))
                act_times.append(float(solve_meta.get("solve_time_sec", time.perf_counter() - start)))
            else:
                action = controller.act(obs); act_times.append(time.perf_counter() - start)
            obs, reward, done, _, info = env.step(action)
            ep_r += float(np.sum(reward)); ep_v += int(info["voltage_violation_count"]); ep_p += float(info["storage_profit_eur"])
            ep_min = min(ep_min, float(info["min_vm_pu"])); ep_max = max(ep_max, float(info["max_vm_pu"])); ep_trafo = max(ep_trafo, float(np.max(info["trafo_loading_pct"])))
            voltage_trace.append(info["vm_pu"]); trafo_trace.append(info["trafo_loading_pct"]); trafo_overload_steps += int(np.max(info["trafo_loading_pct"]) > 100.0)
            if done: break
        rewards.append(ep_r); violations.append(ep_v); profits.append(ep_p); min_vms.append(ep_min); max_vms.append(ep_max); trafo_maxes.append(ep_trafo)
        episodes.set_postfix(reward=round(ep_r, 3), violations=int(ep_v), trafo=round(ep_trafo, 2))
    return {"controller": controller.name, "forecast_mode": forecast_mode, "cfg_hash": cfg.hash8(), "n_episodes": n, "eval_episode_indices": indices[:n], "episode_reward_mean": float(np.mean(rewards)), "storage_profit_eur_mean": float(np.mean(profits)), "voltage_violation_steps_mean": float(np.mean(violations)), "min_vm_pu": float(min(min_vms)), "max_vm_pu": float(max(max_vms)), "trafo_loading_max_pct": float(max(trafo_maxes)), "trafo_overload_steps": int(trafo_overload_steps), "act_time_s_mean": float(np.mean(act_times)), "voltage_trace": np.asarray(voltage_trace, dtype=np.float32), "trafo_trace": np.asarray(trafo_trace, dtype=np.float32)}


def eval_and_save_controller(cfg: Cfg, controller: Controller, run_dir: Path, forecast_mode: str, n_episodes: int | None = None, share_data: ShareData | None = None) -> dict:
    result = eval_controller(cfg, controller, forecast_mode=forecast_mode, n_episodes=n_episodes, share_data=share_data)
    save_eval_result(run_dir, result)
    return result


def eval_all(cfg: Cfg, madrl_models: dict[str, Path], run_dir: Path, forecast_modes: tuple[str, ...] | None = None, n_episodes: int | None = None, controller_names: tuple[str, ...] = CONTROLLER_NAMES, share_data: ShareData | None = None, overwrite: bool = False) -> pd.DataFrame:
    rows = []; modes = tuple(cfg.eval.forecast_modes if forecast_modes is None else forecast_modes)
    for fm in tqdm(modes, desc="forecast modes", unit="mode", ascii=True):
        for name in tqdm([x for x in ("MISOCP", "LOCAL_MPC", "ADMM_MPC") if x in controller_names], desc=f"controllers {fm}", unit="controller", leave=False, ascii=True):
            rows.append(load_eval_metrics(run_dir, cfg, fm, name) if (not overwrite and eval_result_exists(run_dir, fm, name)) else eval_and_save_controller(cfg, build_controller(name, cfg), run_dir, fm, n_episodes, share_data))
        for name, model_path in tqdm(tuple(madrl_models.items()), desc=f"MADRL eval {fm}", unit="controller", leave=False, ascii=True):
            if name in controller_names:
                rows.append(load_eval_metrics(run_dir, cfg, fm, name) if (not overwrite and eval_result_exists(run_dir, fm, name)) else eval_and_save_controller(cfg, build_controller(name, cfg, model_path), run_dir, fm, n_episodes, share_data))
    df = pd.DataFrame(rows); save_eval_summary(run_dir, df); return df
