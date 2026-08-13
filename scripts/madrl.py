from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
import json, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from configs.cfg import Cfg
from controllers.madrl import MADRLController
from controllers.madrl_safety import JointGridSafetyProjector, enforce_local_action_feasibility, local_bounds_torch, map_actor_output_to_soc_feasible_action
from data.share_data import ShareData
from envs.grid_env import build_env
from envs.vec_env import SubprocVecEnv
from models.assembly import Actor, Critic, SharedTwinCritic, build_actors, to_torch_obs
from utils.records import collect_rollout, compute_ev_session_cost_summary as _record_ev_session_cost_summary, compute_ev_session_summary as _record_ev_session_summary, load_record, plot_basic_records, save_rollout
from utils.torch_runtime import resolve_device, set_seed

SCHEMES = (
    {"scheme": "madrl_base", "controller": "MADRL_BASE", "label": "MADRL + No Safety + LSTM Forecast", "algo": "MATD3", "projection": False, "reward": (0.0, 0.0, 0.0), "episodes": 500},
    {"scheme": "madrl_base_safe", "controller": "MADRL_PENALTY", "label": "MADRL + Safety Penalty + LSTM Forecast", "algo": "MATD3", "projection": False, "reward": (400.0, 0.0, 10.0), "episodes": 500},
    {"scheme": "madrl_projection_safe", "controller": "MADRL_PROJECTION", "label": "MADRL + Safety Projection + LSTM Forecast", "algo": "MATD3_SAFE_POC", "projection": True, "reward": (400.0, 0.0, 10.0), "episodes": 500},
)
OBS_KEYS = ("madrl_local", "safety_local", "wholesale_price_relative_seq", "wholesale_price_spread_seq", "load_seq", "pv_seq")
REWARD_COMPONENT_SPECS = (
    ("madrl_r_inc", 1.0, "income", "#2563eb"),
    ("madrl_r_action_penalty", -1.0, "action penalty", "#f59e0b"),
    ("madrl_r_soc_regularization", -1.0, "soc regularization", "#16a34a"),
    ("madrl_r_ev_progress_penalty", -1.0, "EV progress penalty", "#9333ea"),
    ("madrl_r_ev_price_aware_penalty", -1.0, "EV price-aware penalty", "#0f766e"),
    ("madrl_r_throughput_bonus", 1.0, "throughput bonus", "#14b8a6"),
    ("madrl_r_safe_v", -1.0, "voltage safety", "#dc2626"),
    ("madrl_r_safe_line", -1.0, "line safety", "#7c3aed"),
    ("madrl_r_safe_trafo", -1.0, "trafo safety", "#e11d48"),
    ("madrl_r_safe_total", -1.0, "safety total", "#475569"),
)
REWARD_COMPONENT_COLUMNS = tuple(item[0] for item in REWARD_COMPONENT_SPECS)
EV_AGENT_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2")


def _scheme_cfg(cfg: Cfg, spec: dict[str, Any], episodes: int) -> Cfg:
    w_v, w_l, w_t = spec["reward"]
    return replace(cfg, algo=replace(cfg.algo, name=str(spec["algo"])), train=replace(cfg.train, train_episodes=int(episodes)), reward=replace(cfg.reward, w_voltage_pen=float(w_v), w_line_pen=float(w_l), w_trafo_pen=float(w_t)), safety=replace(cfg.safety, enabled=bool(spec["projection"])))


def _slice_obs(obs: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[idx], dtype=np.float32).copy() for key, value in obs.items()}


class ReplayBuffer:
    def __init__(self, cfg: Cfg, num_envs: int) -> None:
        self.capacity, self.gamma, self.n_step = int(cfg.train.buffer_size), float(cfg.algo.gamma), max(int(cfg.train.n_step_return), 1)
        self.pos = self.size = 0; self.queues = [deque() for _ in range(int(num_envs))]
        self.obs_buffers: dict[str, np.ndarray] | None = None; self.next_obs_buffers: dict[str, np.ndarray] | None = None
        self.action_buffer: np.ndarray | None = None; self.reward_buffer: np.ndarray | None = None
        self.terminated_buffer: np.ndarray | None = None; self.bootstrap_discount_buffer: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.size)

    def _ensure_buffers(self, item: dict[str, Any]) -> None:
        if self.obs_buffers is not None:
            return
        self.obs_buffers = {key: np.zeros((self.capacity, *np.asarray(item["obs"][key], dtype=np.float32).shape), dtype=np.float32) for key in OBS_KEYS}
        self.next_obs_buffers = {key: np.zeros((self.capacity, *np.asarray(item["next_obs"][key], dtype=np.float32).shape), dtype=np.float32) for key in OBS_KEYS}
        self.action_buffer = np.zeros((self.capacity, *np.asarray(item["action"], dtype=np.float32).shape), dtype=np.float32)
        self.reward_buffer = np.zeros((self.capacity, *np.asarray(item["reward"], dtype=np.float32).shape), dtype=np.float32)
        self.terminated_buffer = np.zeros((self.capacity, *np.asarray(item["terminated"], dtype=np.float32).shape), dtype=np.float32)
        self.bootstrap_discount_buffer = np.zeros((self.capacity, *np.asarray(item["bootstrap_discount"], dtype=np.float32).shape), dtype=np.float32)

    def _store(self, item: dict[str, Any]) -> None:
        self._ensure_buffers(item)
        assert self.obs_buffers is not None and self.next_obs_buffers is not None and self.action_buffer is not None and self.reward_buffer is not None and self.terminated_buffer is not None and self.bootstrap_discount_buffer is not None
        idx = int(self.pos)
        for key in OBS_KEYS:
            self.obs_buffers[key][idx] = np.asarray(item["obs"][key], dtype=np.float32)
            self.next_obs_buffers[key][idx] = np.asarray(item["next_obs"][key], dtype=np.float32)
        self.action_buffer[idx] = np.asarray(item["action"], dtype=np.float32); self.reward_buffer[idx] = np.asarray(item["reward"], dtype=np.float32)
        self.terminated_buffer[idx] = np.asarray(item["terminated"], dtype=np.float32); self.bootstrap_discount_buffer[idx] = np.asarray(item["bootstrap_discount"], dtype=np.float32)
        self.size = min(self.size + 1, self.capacity); self.pos = (self.pos + 1) % self.capacity

    def _emit(self, lane: int) -> None:
        queue = self.queues[lane]; horizon = min(len(queue), self.n_step)
        reward = sum((self.gamma ** k) * queue[k]["reward"] for k in range(horizon))
        last = queue[horizon - 1]
        self._store({"obs": queue[0]["obs"], "action": queue[0]["action"], "reward": reward.astype(np.float32), "next_obs": last["next_obs"], "terminated": np.full_like(reward, bool(last["done"]), dtype=np.float32), "bootstrap_discount": np.full_like(reward, 0.0 if last["done"] else self.gamma ** horizon, dtype=np.float32)})
        queue.popleft()

    def add_batch(self, obs: dict[str, np.ndarray], action: np.ndarray, reward: np.ndarray, next_obs: dict[str, np.ndarray], done: np.ndarray) -> None:
        for lane in range(int(action.shape[0])):
            self.queues[lane].append({"obs": _slice_obs(obs, lane), "action": np.asarray(action[lane], dtype=np.float32).copy(), "reward": np.asarray(reward[lane], dtype=np.float32).copy(), "next_obs": _slice_obs(next_obs, lane), "done": bool(done[lane])})
            if len(self.queues[lane]) >= self.n_step:
                self._emit(lane)
            if bool(done[lane]):
                while self.queues[lane]:
                    self._emit(lane)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        if self.obs_buffers is None or self.next_obs_buffers is None or self.action_buffer is None or self.reward_buffer is None or self.terminated_buffer is None or self.bootstrap_discount_buffer is None:
            raise RuntimeError("ReplayBuffer cannot sample before at least one transition is stored.")
        idxs = rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": {key: values[idxs].astype(np.float32, copy=True) for key, values in self.obs_buffers.items()},
            "next_obs": {key: values[idxs].astype(np.float32, copy=True) for key, values in self.next_obs_buffers.items()},
            "action": self.action_buffer[idxs].astype(np.float32, copy=True), "reward": self.reward_buffer[idxs].astype(np.float32, copy=True),
            "terminated": self.terminated_buffer[idxs].astype(np.float32, copy=True), "bootstrap_discount": self.bootstrap_discount_buffer[idxs].astype(np.float32, copy=True),
        }


def _step_optim(cfg: Cfg, optim: torch.optim.Optimizer, loss: torch.Tensor, module: torch.nn.Module) -> None:
    optim.zero_grad(set_to_none=True); loss.backward()
    if bool(cfg.model.use_grad_clip):
        torch.nn.utils.clip_grad_norm_(module.parameters(), float(cfg.model.grad_clip_norm))
    optim.step()


def _soft_update(cfg: Cfg, source: torch.nn.Module, target: torch.nn.Module) -> None:
    tau = float(cfg.algo.tau)
    for src_p, dst_p in zip(source.parameters(), target.parameters(), strict=True):
        dst_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)


class Agent:
    def __init__(self, cfg: Cfg, agent_id: int, device: torch.device) -> None:
        self.cfg, self.agent_id, self.device = cfg, int(agent_id), device
        self.actor = Actor(cfg, agent_id).to(device); self.critic = Critic(cfg, twin=True).to(device)
        self.actor_target = deepcopy(self.actor); self.critic_target = deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr)); self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))
        self.policy_pointer = 0


def _to_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {"obs": to_torch_obs(batch["obs"], device), "next_obs": to_torch_obs(batch["next_obs"], device), **{key: torch.as_tensor(batch[key], dtype=torch.float32, device=device) for key in ("action", "reward", "terminated", "bootstrap_discount")}}


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _signed_reward_component(info: dict[str, Any], key: str, sign: float) -> float:
    return float(sign * np.sum(np.asarray(info[key], dtype=np.float32)))


def _stack_actor_outputs(actors: list[Actor], obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([actor(obs) for actor in actors], dim=1)


def _make_train_env(cfg: Cfg, share_data: ShareData) -> SubprocVecEnv:
    env_factory = partial(build_env, cfg, "train", forecast_mode="lstm", share_data=share_data)
    return SubprocVecEnv(int(cfg.train.num_envs), env_factory, seed=int(cfg.runtime.seed), parallel_episode_sampling=str(cfg.train.parallel_episode_sampling))


def _guard_joint(cfg: Cfg, obs: dict[str, torch.Tensor], raw: torch.Tensor, projector: JointGridSafetyProjector | None = None) -> torch.Tensor:
    action = map_actor_output_to_soc_feasible_action(cfg, obs["safety_local"], raw)
    if projector is not None:
        action = projector.project_actions_from_safety_local(obs["safety_local"], action)
    return enforce_local_action_feasibility(cfg, obs["safety_local"], action)


def _sample_rollout_action(cfg: Cfg, actors: list[Actor], obs: dict[str, np.ndarray], device: torch.device, rng: np.random.Generator, step: int, projector: JointGridSafetyProjector | None) -> np.ndarray:
    obs_t = to_torch_obs({key: obs[key] for key in OBS_KEYS}, device)
    with torch.no_grad():
        raw = _stack_actor_outputs(actors, obs_t)
        noise = max(float(cfg.train.noise_std_min), float(cfg.train.noise_std_init) - (float(cfg.train.noise_std_init) - float(cfg.train.noise_std_min)) * min(float(step) / max(float(cfg.train.noise_decay_steps), 1.0), 1.0))
        raw = torch.clamp(raw + torch.randn_like(raw) * noise, -1.0, 1.0)
        action = map_actor_output_to_soc_feasible_action(cfg, obs_t["safety_local"], raw)
        rate = float(cfg.train.feasible_random_exploration_end) + (float(cfg.train.feasible_random_exploration_start) - float(cfg.train.feasible_random_exploration_end)) * max(0.0, 1.0 - float(step) / max(float(cfg.train.feasible_random_exploration_decay_steps), 1.0))
        mask = torch.as_tensor(rng.random(action[..., 0].shape) < rate, device=device)
        lower, upper, pmax, _ = local_bounds_torch(cfg, obs_t["safety_local"])
        random_battery = (lower + torch.rand_like(lower) * torch.clamp(upper - lower, min=0.0)) / torch.clamp(pmax, min=1e-6)
        action[..., 0] = torch.where(mask, random_battery, action[..., 0])
        if projector is not None:
            action = projector.project_actions_from_safety_local(obs_t["safety_local"], action)
        action = enforce_local_action_feasibility(cfg, obs_t["safety_local"], action)
    return action.detach().cpu().numpy().astype(np.float32)


def _train_agent(cfg: Cfg, agents: list[Agent], agent: Agent, batch: dict[str, Any], projector: JointGridSafetyProjector | None, allow_actor_update: bool) -> dict[str, float]:
    agent.policy_pointer += 1
    obs, next_obs, action, reward, discount = batch["obs"], batch["next_obs"], batch["action"], batch["reward"], batch["bootstrap_discount"]
    with torch.no_grad():
        clean_next = _stack_actor_outputs([item.actor_target for item in agents], next_obs)
        noisy_next = torch.clamp(clean_next + torch.clamp(torch.randn_like(clean_next) * float(cfg.algo.policy_noise), -float(cfg.algo.noise_clip), float(cfg.algo.noise_clip)), -1.0, 1.0)
        next_action = _guard_joint(cfg, next_obs, noisy_next, projector)
        q1_next, q2_next = agent.critic_target(next_obs, next_action)
        target_q = reward[:, agent.agent_id] + discount[:, agent.agent_id] * torch.minimum(q1_next, q2_next)
    q1, q2 = agent.critic(obs, action)
    critic_loss = F.mse_loss(q1.float(), target_q.float()) + F.mse_loss(q2.float(), target_q.float())
    _step_optim(cfg, agent.critic_optim, critic_loss, agent.critic)
    actor_loss_value = float("nan")
    if allow_actor_update and agent.policy_pointer % int(cfg.algo.policy_update_freq) == 0:
        candidate = action.clone()
        candidate[:, agent.agent_id] = map_actor_output_to_soc_feasible_action(cfg, obs["safety_local"][:, agent.agent_id:agent.agent_id + 1], agent.actor(obs).unsqueeze(1))[:, 0]
        policy_action = enforce_local_action_feasibility(cfg, obs["safety_local"], projector.project_actions_from_safety_local(obs["safety_local"], candidate) if projector is not None else candidate)
        q1_policy, _ = agent.critic(obs, policy_action)
        actor_loss = -q1_policy.float().mean(); _step_optim(cfg, agent.actor_optim, actor_loss, agent.actor)
        _soft_update(cfg, agent.actor, agent.actor_target); _soft_update(cfg, agent.critic, agent.critic_target)
        actor_loss_value = float(actor_loss.detach().cpu())
    elif not allow_actor_update:
        _soft_update(cfg, agent.actor, agent.actor_target); _soft_update(cfg, agent.critic, agent.critic_target)
    return {"critic_loss": float(critic_loss.detach().cpu()), "actor_loss": actor_loss_value}


def _save_model(cfg: Cfg, agents: list[Agent], model_dir: Path, meta: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True); path = model_dir / "model.pt"
    torch.save({"actors": [{k: v.detach().cpu() for k, v in agent.actor.state_dict().items()} for agent in agents], "critics": [{k: v.detach().cpu() for k, v in agent.critic.state_dict().items()} for agent in agents], "meta": meta}, path)
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def train_madrl_scheme(cfg: Cfg, run_dir: str | Path, share_data: ShareData, spec: dict[str, Any], *, episodes: int | None = None) -> dict[str, Any]:
    episodes = int(spec["episodes"] if episodes is None else episodes)
    work_cfg = _scheme_cfg(cfg, spec, episodes); device = resolve_device(work_cfg.runtime.device); set_seed(int(work_cfg.runtime.seed))
    agents = [Agent(work_cfg, agent_id, device) for agent_id in range(int(work_cfg.env.num_agents))]
    projector = JointGridSafetyProjector(work_cfg).to(device) if bool(spec["projection"]) else None
    rng = np.random.default_rng(int(work_cfg.runtime.seed))
    env = _make_train_env(work_cfg, share_data)
    buffer = ReplayBuffer(work_cfg, int(work_cfg.train.num_envs))
    obs, _ = env.reset(); lane_rewards = np.zeros((int(work_cfg.train.num_envs),), dtype=np.float32)
    lane_components = {key: np.zeros((int(work_cfg.train.num_envs),), dtype=np.float32) for key in REWARD_COMPONENT_COLUMNS}
    completed = total_steps = updates = 0; rows: list[dict[str, Any]] = []; loss_last = {"critic_loss": np.nan, "actor_loss": np.nan}; started = time.perf_counter()
    timers = {"action_sample_s": 0.0, "env_step_s": 0.0, "replay_add_s": 0.0, "sample_update_s": 0.0}
    progress = tqdm(total=episodes, desc=f"train {spec['scheme']}", unit="episode", ascii=True)
    while completed < episodes:
        tick = time.perf_counter()
        action = _sample_rollout_action(work_cfg, [agent.actor for agent in agents], obs, device, rng, total_steps, projector)
        timers["action_sample_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        next_obs, reward, done, _, infos = env.step(action)
        timers["env_step_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        buffer.add_batch(obs, action, reward, next_obs, done)
        timers["replay_add_s"] += time.perf_counter() - tick
        lane_rewards += np.sum(reward, axis=1).astype(np.float32)
        for lane, info in enumerate(infos):
            for key, sign, _, _ in REWARD_COMPONENT_SPECS:
                lane_components[key][lane] += _signed_reward_component(info, key, sign)
        obs = next_obs; total_steps += int(work_cfg.train.num_envs)
        if len(buffer) >= int(work_cfg.train.learning_starts) and total_steps % max(int(work_cfg.train.update_interval), 1) == 0:
            for _ in range(max(int(work_cfg.train.updates_per_step), 1)):
                tick = time.perf_counter()
                batch = _to_batch(buffer.sample(int(work_cfg.train.batch_size), rng), device); allow_actor = len(buffer) >= int(work_cfg.train.actor_learning_starts)
                metrics = [_train_agent(work_cfg, agents, agent, batch, projector, allow_actor) for agent in agents]
                timers["sample_update_s"] += time.perf_counter() - tick
                loss_last = {"critic_loss": _finite_mean([item["critic_loss"] for item in metrics]), "actor_loss": _finite_mean([item["actor_loss"] for item in metrics])}
                updates += 1
        if bool(np.all(done)):
            for lane in range(int(work_cfg.train.num_envs)):
                if completed >= episodes:
                    break
                completed += 1
                row = {"scheme": spec["scheme"], "controller": spec["controller"], "episode": completed, "total_reward": float(lane_rewards[lane]), **loss_last}
                row.update({key: float(lane_components[key][lane]) for key in REWARD_COMPONENT_COLUMNS})
                rows.append(row)
                lane_rewards[lane] = 0.0
                for key in REWARD_COMPONENT_COLUMNS:
                    lane_components[key][lane] = 0.0
            progress.update(min(int(np.sum(done)), episodes - progress.n)); progress.set_postfix(reward=f"{rows[-1]['total_reward']:.2f}", critic=f"{loss_last['critic_loss']:.3e}", refresh=False)
    progress.close(); env.close()
    elapsed_s = float(time.perf_counter() - started); timed_s = float(sum(timers.values()))
    meta = {"scheme": spec["scheme"], "controller": spec["controller"], "label": spec["label"], "algo": spec["algo"], "projection": bool(spec["projection"]), "episodes": int(episodes), "steps": int(total_steps), "updates": int(updates), "elapsed_s": elapsed_s, "vec_env_kind": "subproc", "replay_buffer_kind": "array", **timers, "other_s": max(0.0, elapsed_s - timed_s), "batch_size": int(work_cfg.train.batch_size), "num_envs": int(work_cfg.train.num_envs), "parallel_episode_sampling": str(work_cfg.train.parallel_episode_sampling)}
    model_path = _save_model(work_cfg, agents, Path(run_dir) / "models" / "madrl" / str(spec["scheme"]), meta)
    table_dir = Path(run_dir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"algo": row["controller"], "episode": row["episode"], "reward": row["total_reward"]} for row in rows]).to_csv(table_dir / f"learning_curve_{spec['scheme']}.csv", index=False)
    pd.DataFrame(rows).to_csv(table_dir / f"madrl_reward_curves_{spec['scheme']}.csv", index=False)
    return {"cfg": work_cfg.with_algo(str(spec["controller"])), "model_path": model_path, "meta": meta, "reward_rows": rows}


def run_madrl_scheme_experiment(cfg: Cfg, run_dir: str | Path, share_data: ShareData, spec: dict[str, Any], *, episodes: int | None = None) -> dict[str, Any]:
    trained = train_madrl_scheme(cfg, run_dir, share_data, spec, episodes=episodes)
    controller = MADRLController.load(trained["cfg"], trained["model_path"])
    rollout = collect_rollout(trained["cfg"], controller, share_data, forecast_mode="lstm", label=str(spec["label"]))
    saved = save_rollout(trained["cfg"], run_dir, rollout, scheme_name=str(spec["scheme"]))
    summary = pd.DataFrame([{**trained["meta"], "model_path": str(trained["model_path"]), "record_dir": str(saved["record_dir"])}])
    summary.to_csv(Path(run_dir) / "tables" / f"madrl_train_summary_{spec['scheme']}.csv", index=False)
    return {"records": {str(spec["scheme"]): saved}, "train_summary": summary, "reward_curves": pd.DataFrame(trained["reward_rows"])}


def run_madrl_experiments(cfg: Cfg, run_dir: str | Path, share_data: ShareData, *, episodes: int | None = None) -> dict[str, Any]:
    records: dict[str, Any] = {}; rewards: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
    for spec in tqdm(SCHEMES, desc="MADRL schemes", unit="scheme", ascii=True):
        result = run_madrl_scheme_experiment(cfg, run_dir, share_data, spec, episodes=episodes)
        records.update(result["records"]); rewards.extend(result["reward_curves"].to_dict("records")); summaries.extend(result["train_summary"].to_dict("records"))
    table_dir = Path(run_dir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(table_dir / "madrl_train_summary.csv", index=False)
    pd.DataFrame(rewards).to_csv(table_dir / "madrl_reward_curves.csv", index=False)
    pd.DataFrame([{"algo": row["controller"], "episode": row["episode"], "reward": row["total_reward"]} for row in rewards]).to_csv(table_dir / "learning_curves.csv", index=False)
    return {"records": records, "train_summary": pd.DataFrame(summaries), "reward_curves": pd.DataFrame(rewards)}


def load_madrl_record(run_dir: str | Path, controller: str) -> dict[str, Any]:
    return load_record(run_dir, "lstm", controller)


def _plot_reward_curves(rewards: pd.DataFrame, run_dir: str | Path, prefix: str) -> Any:
    import matplotlib.pyplot as plt
    rewards = rewards.sort_values(["scheme", "episode"]) if "scheme" in rewards.columns else rewards.sort_values("episode")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})
    group_iter = rewards.groupby("scheme", sort=False) if "scheme" in rewards.columns else [("MADRL", rewards)]
    for scheme, group in group_iter:
        ordered = group.sort_values("episode")
        episodes = ordered["episode"].to_numpy()
        total = ordered["total_reward"].to_numpy(dtype=np.float32)
        axes[0].plot(episodes, total, color="#94a3b8", linewidth=0.9, alpha=0.55, label=f"{scheme} total")
        window = min(10, max(len(ordered), 1))
        rolling = ordered["total_reward"].rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float32)
        axes[0].plot(episodes, rolling, color="#111827", linewidth=1.8, label=f"{scheme} rolling mean")
        multi_scheme = "scheme" in rewards.columns and rewards["scheme"].nunique() > 1
        for key, _, label, color in REWARD_COMPONENT_SPECS:
            axes[1].plot(episodes, ordered[key].to_numpy(dtype=np.float32), linewidth=1.1, alpha=0.82, color=color, label=f"{scheme} {label}" if multi_scheme else label)
    axes[0].set_ylabel("episode reward"); axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.25)
    axes[0].set_title("Episode total reward")
    axes[1].set_title("Signed reward components"); axes[1].set_ylabel("signed reward"); axes[1].set_xlabel("episode")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out = Path(run_dir) / "figures"; out.mkdir(parents=True, exist_ok=True); fig.savefig(out / f"{prefix}_learning_curve.png", dpi=140)
    return fig


def _has_ev_rollout_fields(rollouts: dict[str, Any]) -> bool:
    return any({"ev_soc", "ev_charge_kw"}.issubset(set(rollout.agent_df.columns)) for rollout in rollouts.values())


def _series_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(float) if column in frame.columns else pd.Series(0.0, index=frame.index, dtype=float)


def _ev_charging_cost_weight(cfg) -> float:
    mode = str(getattr(cfg.env, "ev_departure_constraint_mode", "soft")).lower()
    return float(getattr(cfg.reward, "ev_charging_cost_weight", 1.0)) if mode in {"soft", "hard"} else 1.0


def compute_ev_cost_summary(rollout, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return step-level EV experiment costs and a one-row total cost summary.

    PV cost is currently zero unless a PV curtailment price or penalty is added to
    the recorded rollout fields.
    """
    agent = rollout.agent_df.copy()
    if agent.empty:
        cost_ts = pd.DataFrame(columns=["episode_idx", "step", "timestamp", "battery_cost_eur", "ev_cost_eur", "pv_cost_eur", "total_cost_eur", "ev_charge_requested_kwh", "ev_charge_executed_kwh", "ev_charge_clipped_kwh", "ev_overcharge_gap", "ev_overcharge_penalty"])
        return cost_ts, pd.DataFrame([{"battery_total_cost_eur": 0.0, "ev_total_cost_eur": 0.0, "pv_total_cost_eur": 0.0, "total_cost_eur": 0.0, "total_ev_charge_requested_kwh": 0.0, "total_ev_charge_executed_kwh": 0.0, "total_ev_charge_clipped_kwh": 0.0, "ev_overcharge_gap_total": 0.0, "ev_overcharge_penalty_total": 0.0}])
    dt = float(rollout.meta.get("dt_hours", getattr(cfg.env, "dt_hours", 0.25)))
    step_price = rollout.step_df[[c for c in ("episode_idx", "step", "timestamp", "import_price", "wholesale_price") if c in rollout.step_df.columns]].drop_duplicates(["episode_idx", "step"])
    if "import_price" not in agent.columns and "import_price" in step_price.columns:
        agent = agent.merge(step_price[[c for c in ("episode_idx", "step", "import_price") if c in step_price.columns]], on=["episode_idx", "step"], how="left")
    if "timestamp" not in agent.columns and "timestamp" in step_price.columns:
        agent = agent.merge(step_price[["episode_idx", "step", "timestamp"]], on=["episode_idx", "step"], how="left")
    if "storage_profit_eur" in agent.columns:
        battery_cost = -agent["storage_profit_eur"].astype(float)
    elif "madrl_r_inc" in agent.columns:
        battery_cost = -agent["madrl_r_inc"].astype(float)
    elif {"e_bat", "import_price"}.issubset(agent.columns):
        battery_cost = agent["e_bat"].astype(float) * agent["import_price"].astype(float) * dt
    else:
        battery_cost = pd.Series(np.nan, index=agent.index, dtype=float)
    if "ev_charging_cost_eur" in agent.columns:
        ev_cost = agent["ev_charging_cost_eur"].astype(float)
    elif {"ev_charge_kw", "import_price"}.issubset(agent.columns):
        ev_cost = _ev_charging_cost_weight(cfg) * agent["ev_charge_kw"].astype(float).clip(lower=0.0) * agent["import_price"].astype(float) * dt
    else:
        ev_cost = pd.Series(0.0, index=agent.index, dtype=float)
    ev_charge_requested = _series_or_zero(agent, "ev_charge_kw_requested") if "ev_charge_kw_requested" in agent.columns else _series_or_zero(agent, "ev_charge_kw")
    ev_charge_executed = _series_or_zero(agent, "ev_charge_kw_executed") if "ev_charge_kw_executed" in agent.columns else _series_or_zero(agent, "ev_charge_kw")
    ev_charge_clipped = _series_or_zero(agent, "ev_charge_kw_clipped") if "ev_charge_kw_clipped" in agent.columns else (ev_charge_requested - ev_charge_executed).clip(lower=0.0)
    ev_overcharge_gap = _series_or_zero(agent, "ev_overcharge_gap")
    ev_overcharge_penalty = _series_or_zero(agent, "ev_overcharge_penalty")
    pv_cost = _series_or_zero(agent, "pv_cost_eur")
    per_agent = agent[[c for c in ("episode_idx", "step", "timestamp", "agent_id") if c in agent.columns]].copy()
    per_agent["battery_cost_eur"] = battery_cost
    per_agent["ev_cost_eur"] = ev_cost
    per_agent["pv_cost_eur"] = pv_cost
    per_agent["total_cost_eur"] = per_agent[["battery_cost_eur", "ev_cost_eur", "pv_cost_eur"]].sum(axis=1, min_count=1)
    per_agent["ev_charge_requested_kwh"] = ev_charge_requested.clip(lower=0.0) * dt
    per_agent["ev_charge_executed_kwh"] = ev_charge_executed.clip(lower=0.0) * dt
    per_agent["ev_charge_clipped_kwh"] = ev_charge_clipped.clip(lower=0.0) * dt
    per_agent["ev_overcharge_gap"] = ev_overcharge_gap
    per_agent["ev_overcharge_penalty"] = ev_overcharge_penalty
    group_cols = [c for c in ("episode_idx", "step", "timestamp") if c in per_agent.columns]
    cost_columns = ["battery_cost_eur", "ev_cost_eur", "pv_cost_eur", "total_cost_eur", "ev_charge_requested_kwh", "ev_charge_executed_kwh", "ev_charge_clipped_kwh", "ev_overcharge_gap", "ev_overcharge_penalty"]
    cost_ts = per_agent.groupby(group_cols, as_index=False)[cost_columns].sum(min_count=1)
    cost_ts = cost_ts.sort_values([c for c in ("episode_idx", "step") if c in cost_ts.columns]).reset_index(drop=True)
    cost_summary = pd.DataFrame([{
        "battery_total_cost_eur": float(cost_ts["battery_cost_eur"].sum()),
        "ev_total_cost_eur": float(cost_ts["ev_cost_eur"].sum()),
        "pv_total_cost_eur": float(cost_ts["pv_cost_eur"].sum()),
        "total_cost_eur": float(cost_ts["total_cost_eur"].sum()),
        "total_ev_charge_requested_kwh": float(cost_ts["ev_charge_requested_kwh"].sum()),
        "total_ev_charge_executed_kwh": float(cost_ts["ev_charge_executed_kwh"].sum()),
        "total_ev_charge_clipped_kwh": float(cost_ts["ev_charge_clipped_kwh"].sum()),
        "ev_overcharge_gap_total": float(cost_ts["ev_overcharge_gap"].sum()),
        "ev_overcharge_penalty_total": float(cost_ts["ev_overcharge_penalty"].sum()),
        "pv_cost_note": "PV cost currently set to zero unless curtailment price/penalty is configured.",
    }])
    return cost_ts, cost_summary


def compute_ev_agent_cost_summary(rollout, cfg) -> pd.DataFrame:
    agent = rollout.agent_df.copy()
    if agent.empty or "agent_id" not in agent.columns:
        return pd.DataFrame(columns=["agent", "battery_cost_eur", "ev_cost_eur", "pv_cost_eur", "total_cost_eur", "ev_charge_requested_kwh", "ev_charge_executed_kwh", "ev_charge_clipped_kwh", "ev_overcharge_gap", "ev_overcharge_penalty"])
    step_price = rollout.step_df[[c for c in ("episode_idx", "step", "import_price") if c in rollout.step_df.columns]].drop_duplicates(["episode_idx", "step"])
    if "import_price" not in agent.columns and "import_price" in step_price.columns:
        agent = agent.merge(step_price, on=["episode_idx", "step"], how="left")
    dt = float(rollout.meta.get("dt_hours", getattr(cfg.env, "dt_hours", 0.25)))
    if "storage_profit_eur" in agent.columns:
        battery_cost = -agent["storage_profit_eur"].astype(float)
    elif "madrl_r_inc" in agent.columns:
        battery_cost = -agent["madrl_r_inc"].astype(float)
    elif {"e_bat", "import_price"}.issubset(agent.columns):
        battery_cost = agent["e_bat"].astype(float) * agent["import_price"].astype(float) * dt
    else:
        battery_cost = pd.Series(np.nan, index=agent.index, dtype=float)
    ev_cost = agent["ev_charging_cost_eur"].astype(float) if "ev_charging_cost_eur" in agent.columns else _ev_charging_cost_weight(cfg) * _series_or_zero(agent, "ev_charge_kw").clip(lower=0.0) * _series_or_zero(agent, "import_price") * dt
    ev_charge_requested = _series_or_zero(agent, "ev_charge_kw_requested") if "ev_charge_kw_requested" in agent.columns else _series_or_zero(agent, "ev_charge_kw")
    ev_charge_executed = _series_or_zero(agent, "ev_charge_kw_executed") if "ev_charge_kw_executed" in agent.columns else _series_or_zero(agent, "ev_charge_kw")
    ev_charge_clipped = _series_or_zero(agent, "ev_charge_kw_clipped") if "ev_charge_kw_clipped" in agent.columns else (ev_charge_requested - ev_charge_executed).clip(lower=0.0)
    pv_cost = _series_or_zero(agent, "pv_cost_eur")
    frame = pd.DataFrame({
        "agent": agent["agent_id"].astype(int), "battery_cost_eur": battery_cost, "ev_cost_eur": ev_cost, "pv_cost_eur": pv_cost,
        "ev_charge_requested_kwh": ev_charge_requested.clip(lower=0.0) * dt,
        "ev_charge_executed_kwh": ev_charge_executed.clip(lower=0.0) * dt,
        "ev_charge_clipped_kwh": ev_charge_clipped.clip(lower=0.0) * dt,
        "ev_overcharge_gap": _series_or_zero(agent, "ev_overcharge_gap"),
        "ev_overcharge_penalty": _series_or_zero(agent, "ev_overcharge_penalty"),
    })
    frame["total_cost_eur"] = frame[["battery_cost_eur", "ev_cost_eur", "pv_cost_eur"]].sum(axis=1, min_count=1)
    return frame.groupby("agent", as_index=False)[["battery_cost_eur", "ev_cost_eur", "pv_cost_eur", "total_cost_eur", "ev_charge_requested_kwh", "ev_charge_executed_kwh", "ev_charge_clipped_kwh", "ev_overcharge_gap", "ev_overcharge_penalty"]].sum(min_count=1)


def compute_ev_session_summary(rollout, cfg) -> pd.DataFrame:
    return _record_ev_session_summary(rollout, cfg)


def compute_ev_session_cost_summary(rollout, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _record_ev_session_cost_summary(rollout, cfg)


def _plot_total_cost(rollouts: dict[str, Any]):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.6 * len(rollouts), 4.6)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        cost_ts, _ = compute_ev_cost_summary(rollout, Cfg())
        x = cost_ts["timestamp"] if "timestamp" in cost_ts.columns else cost_ts["step"]
        ax.plot(x, cost_ts["total_cost_eur"].fillna(0.0).cumsum(), color="#111827", linewidth=1.8, label="Total cumulative cost" if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - Total cumulative cost"); ax.set_ylabel("Cumulative cost [EUR]"); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_cost_components(rollouts: dict[str, Any], cumulative: bool):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.8)), sharex=True)
    axes = np.atleast_1d(axes)
    specs = [("battery_cost_eur", "Battery cost / revenue", "#2563eb"), ("ev_cost_eur", "EV charging cost", "#0891b2"), ("pv_cost_eur", "PV cost (currently zero)", "#16a34a"), ("total_cost_eur", "Total cost", "#111827")]
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        cost_ts, _ = compute_ev_cost_summary(rollout, Cfg())
        x = cost_ts["timestamp"] if "timestamp" in cost_ts.columns else cost_ts["step"]
        for column, label, color in specs:
            values = cost_ts[column].fillna(0.0).to_numpy(dtype=np.float64)
            if cumulative:
                values = np.cumsum(values)
            ax.plot(x, values, color=color, linewidth=1.7 if column == "total_cost_eur" else 1.3, linestyle="-" if column == "total_cost_eur" else "--", label=label if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - {'Cumulative cost components' if cumulative else 'Cost components per step'}")
        ax.set_ylabel("Cumulative cost [EUR]" if cumulative else "Cost per step [EUR]"); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper left", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _shade_ev_connected(axis, frame: pd.DataFrame) -> None:
    if "ev_available" not in frame.columns:
        return
    connected = frame.groupby(["episode_idx", "step", "timestamp"], as_index=False)["ev_available"].max()
    active = connected["ev_available"].to_numpy(dtype=np.float32) > 0.0
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = connected["timestamp"].iloc[idx]
        if start is not None and (not is_active or idx == len(active) - 1):
            end = connected["timestamp"].iloc[idx] if is_active else connected["timestamp"].iloc[max(idx - 1, 0)]
            axis.axvspan(start, end, color="#e5e7eb", alpha=0.35, linewidth=0.0)
            start = None


def _plot_ev_soc_comparison(rollouts: dict[str, Any], required_soc: float):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.8)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        mode = str(rollout.meta.get("ev_departure_constraint_mode", "soft"))
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            ax.plot(frame["timestamp"], frame["ev_soc"], color=EV_AGENT_COLORS[int(agent_id) % len(EV_AGENT_COLORS)], linewidth=1.7, label=f"Agent {agent_id}" if idx == 0 else None)
        ax.axhline(float(required_soc), color="#dc2626", linestyle="--", linewidth=1.2, label="Required departure SOC" if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - EV SOC - {mode} departure constraint"); ax.set_ylabel("EV SOC"); ax.set_ylim(0.0, 1.0); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_ev_charge_comparison(rollouts: dict[str, Any]):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.8)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        _shade_ev_connected(ax, agent)
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            color = EV_AGENT_COLORS[int(agent_id) % len(EV_AGENT_COLORS)]
            if "ev_charge_kw_rl" in frame.columns:
                rl_values = np.maximum(frame["ev_charge_kw_rl"].to_numpy(dtype=np.float32), 0.0)
                ax.plot(frame["timestamp"], rl_values, color=color, linestyle=":", linewidth=1.1, alpha=0.70, label=f"Agent {agent_id} RL" if idx == 0 else None)
            values = np.maximum(frame["ev_charge_kw"].to_numpy(dtype=np.float32), 0.0)
            ax.plot(frame["timestamp"], values, color=color, linewidth=1.7, label=f"Agent {agent_id} actual" if idx == 0 and "ev_charge_kw_rl" in frame.columns else f"Agent {agent_id}" if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - EV charging power"); ax.set_ylabel("EV charging power [kW]"); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _price_trace(rollout) -> pd.DataFrame:
    step = rollout.step_df.copy()
    price_column = next((column for column in ("import_price", "price", "wholesale_price") if column in step.columns), None)
    if price_column is None:
        return pd.DataFrame(columns=["episode_idx", "step", "timestamp", "price"])
    cols = [column for column in ("episode_idx", "step", "timestamp") if column in step.columns]
    return step.loc[:, cols + [price_column]].drop_duplicates([column for column in ("episode_idx", "step") if column in cols]).rename(columns={price_column: "price"}).sort_values([column for column in ("episode_idx", "step") if column in cols]).reset_index(drop=True)


def _agent_power_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _plot_agent_power_vs_price(rollouts: dict[str, Any], *, power_column_candidates: tuple[str, ...], ylabel: str, title: str, nonnegative: bool = False):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(4.0 * len(rollouts), 5.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        power_column = _agent_power_column(agent, power_column_candidates)
        price = _price_trace(rollout)
        if power_column is None:
            ax.text(0.01, 0.5, f"missing power column: {power_column_candidates}", transform=ax.transAxes, color="#dc2626")
        else:
            for agent_id, frame in agent.groupby("agent_id", sort=True):
                values = frame[power_column].to_numpy(dtype=np.float32)
                if nonnegative:
                    values = np.maximum(values, 0.0)
                ax.plot(frame["timestamp"], values, color=EV_AGENT_COLORS[int(agent_id) % len(EV_AGENT_COLORS)], linewidth=1.6, label=f"Agent {agent_id}" if idx == 0 else None)
        ax.axhline(0.0, color="#64748b", linewidth=0.9)
        ax.set_title(f"{rollout.meta['controller']} - {title}"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25)
        price_ax = ax.twinx()
        if not price.empty:
            x = price["timestamp"] if "timestamp" in price.columns else price["step"]
            price_ax.plot(x, price["price"].to_numpy(dtype=np.float32), color="#111827", linestyle="--", linewidth=1.4, alpha=0.85, label="Electricity price" if idx == 0 else None)
        price_ax.set_ylabel("Electricity price [EUR/kWh]")
        if idx == 0:
            lines, labels = ax.get_legend_handles_labels()
            price_lines, price_labels = price_ax.get_legend_handles_labels()
            ax.legend(lines + price_lines, labels + price_labels, loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_ev_emergency_column(rollouts: dict[str, Any], column: str, ylabel: str, title_suffix: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.8)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        _shade_ev_connected(ax, agent)
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            values = np.maximum(frame[column].to_numpy(dtype=np.float32), 0.0)
            ax.plot(frame["timestamp"], values, color=EV_AGENT_COLORS[int(agent_id) % len(EV_AGENT_COLORS)], linewidth=1.7, label=f"Agent {agent_id}" if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - {title_suffix}"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_ev_hard_projection_comparison(rollouts: dict[str, Any]):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(4.2 * len(rollouts), 5.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        _shade_ev_connected(ax, agent)
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            color = EV_AGENT_COLORS[int(agent_id) % len(EV_AGENT_COLORS)]
            ax.plot(frame["timestamp"], np.maximum(frame["ev_charge_kw_rl"].to_numpy(dtype=np.float32), 0.0), color=color, linestyle=":", linewidth=1.2, alpha=0.75, label=f"Agent {agent_id} RL" if idx == 0 else None)
            ax.plot(frame["timestamp"], np.maximum(frame["ev_charge_kw"].to_numpy(dtype=np.float32), 0.0), color=color, linewidth=1.7, label=f"Agent {agent_id} projected" if idx == 0 else None)
            ax.plot(frame["timestamp"], np.maximum(frame["ev_required_min_charge_kw"].to_numpy(dtype=np.float32), 0.0), color=color, linestyle="--", linewidth=1.0, alpha=0.55, label=f"Agent {agent_id} required min" if idx == 0 else None)
        gap_ax = ax.twinx()
        gap = agent.groupby(["episode_idx", "step", "timestamp"], as_index=False)["ev_projection_gap_kw"].sum()
        gap_ax.fill_between(gap["timestamp"], gap["ev_projection_gap_kw"].to_numpy(dtype=np.float32), color="#f97316", alpha=0.20, label="Projection gap total" if idx == 0 else None)
        gap_ax.set_ylabel("projection gap [kW]")
        ax.set_title(f"{rollout.meta['controller']} - EV hard projection diagnostics"); ax.set_ylabel("EV charging power [kW]"); ax.grid(True, alpha=0.25)
        if idx == 0:
            lines, labels = ax.get_legend_handles_labels()
            gap_lines, gap_labels = gap_ax.get_legend_handles_labels()
            ax.legend(lines + gap_lines, labels + gap_labels, loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_ev_departure_check(rollouts: dict[str, Any], required_soc: float, departure_step: int):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(10, max(3.4 * len(rollouts), 4.0)), sharex=False)
    axes = np.atleast_1d(axes)
    for (_, rollout), ax in zip(rollouts.items(), axes, strict=False):
        agent = rollout.agent_df.copy()
        if "ev_departure_gap" not in agent.columns:
            agent["ev_departure_gap"] = np.maximum(0.0, float(required_soc) - agent["ev_soc"].astype(float))
        departures = agent.loc[agent["step"].astype(int) == int(departure_step)]
        if departures.empty:
            departures = agent.loc[agent["ev_departure_gap"].astype(float) > 0.0]
        if departures.empty:
            departures = agent.sort_values(["episode_idx", "agent_id", "step"]).groupby(["episode_idx", "agent_id"], as_index=False).tail(1)
        summary = departures.groupby("agent_id", as_index=False).agg(ev_departure_gap=("ev_departure_gap", "max"), ev_soc=("ev_soc", "min"))
        colors = ["#16a34a" if float(gap) <= 1e-6 else "#dc2626" for gap in summary["ev_departure_gap"]]
        ax.bar(summary["agent_id"].astype(str), summary["ev_departure_gap"].astype(float), color=colors, alpha=0.82)
        for pos, row in enumerate(summary.itertuples(index=False)):
            ax.text(pos, float(row.ev_departure_gap), f"SOC {float(row.ev_soc):.3f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0.0, color="#111827", linewidth=0.9); ax.set_title(f"{rollout.meta['controller']} - Departure SOC requirement check")
        ax.set_ylabel(f"Gap to {float(required_soc):.2f} SOC"); ax.set_xlabel("Agent"); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); return fig


def _plot_ev_progress_check(rollouts: dict[str, Any], required_soc: float):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(4.2 * len(rollouts), 5.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        _shade_ev_connected(ax, agent)
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            color = EV_AGENT_COLORS[int(agent_id) % len(EV_AGENT_COLORS)]
            ax.plot(frame["timestamp"], frame["ev_soc"].to_numpy(dtype=np.float32), color=color, linewidth=1.7, label=f"Agent {agent_id} EV SOC" if idx == 0 else None)
            ax.plot(frame["timestamp"], frame["ev_progress_target_soc"].to_numpy(dtype=np.float32), color=color, linestyle="--", linewidth=1.2, alpha=0.72, label=f"Agent {agent_id} target" if idx == 0 else None)
        ax.axhline(float(required_soc), color="#111827", linestyle=":", linewidth=1.2, label=f"Departure SOC req {float(required_soc):.2f}" if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - EV SOC progress target check"); ax.set_ylabel("EV SOC"); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _add_ev_madrl_figures(figures: dict[str, Any], rollouts: dict[str, Any], run_dir: str | Path, prefix: str) -> None:
    if not _has_ev_rollout_fields(rollouts):
        return
    out = Path(run_dir) / "figures"; out.mkdir(parents=True, exist_ok=True)
    first = next(iter(rollouts.values()))
    required_soc = float(first.meta.get("ev_departure_soc_req", 0.90))
    departure_step = int(first.meta.get("ev_departure_step", 28))
    ev_specs = {
        f"{prefix}_ev_soc": _plot_ev_soc_comparison(rollouts, required_soc),
        f"{prefix}_ev_charge_kw": _plot_ev_charge_comparison(rollouts),
        f"{prefix}_battery_power_price": _plot_agent_power_vs_price(rollouts, power_column_candidates=("charge_kw", "battery_power_kw", "storage_power_kw", "battery_charge_kw", "p_bat", "e_bat"), ylabel="Battery power [kW]", title="Battery charging/discharging power and electricity price"),
        f"{prefix}_ev_charge_price": _plot_agent_power_vs_price(rollouts, power_column_candidates=("ev_charge_kw",), ylabel="EV charging power [kW]", title="EV charging power and electricity price", nonnegative=True),
        f"{prefix}_ev_departure_check": _plot_ev_departure_check(rollouts, required_soc, departure_step),
        f"{prefix}_total_cost": _plot_total_cost(rollouts),
        f"{prefix}_cost_components": _plot_cost_components(rollouts, cumulative=False),
        f"{prefix}_cumulative_cost_components": _plot_cost_components(rollouts, cumulative=True),
    }
    if any({"ev_charge_kw_rl", "ev_required_min_charge_kw", "ev_projection_gap_kw"}.issubset(set(rollout.agent_df.columns)) for rollout in rollouts.values()):
        ev_specs[f"{prefix}_ev_hard_projection"] = _plot_ev_hard_projection_comparison(rollouts)
    if any({"ev_emergency_added_kw", "ev_emergency_required_kw"}.issubset(set(rollout.agent_df.columns)) for rollout in rollouts.values()):
        ev_specs[f"{prefix}_ev_emergency_added_kw"] = _plot_ev_emergency_column(rollouts, "ev_emergency_added_kw", "Emergency added power [kW]", "EV emergency added charging power")
        ev_specs[f"{prefix}_ev_emergency_required_kw"] = _plot_ev_emergency_column(rollouts, "ev_emergency_required_kw", "Emergency required power [kW]", "EV emergency required charging power")
    if any({"ev_soc", "ev_progress_target_soc"}.issubset(set(rollout.agent_df.columns)) for rollout in rollouts.values()):
        ev_specs[f"{prefix}_ev_progress_check"] = _plot_ev_progress_check(rollouts, required_soc)
    for name, fig in ev_specs.items():
        fig.savefig(out / f"{name}.png", dpi=140)
        figures[name] = fig


def plot_madrl_outputs(result: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    rollouts = {name: saved["rollout"] for name, saved in result["records"].items()}
    prefix = next(iter(rollouts)) if len(rollouts) == 1 else "madrl"
    figures = plot_basic_records(rollouts, run_dir, prefix)
    _add_ev_madrl_figures(figures, rollouts, run_dir, prefix)
    figures[f"{prefix}_learning_curve"] = _plot_reward_curves(result["reward_curves"], run_dir, prefix)
    return figures


SHARED_CRITIC_SPEC = {"scheme": "madrl_base_shared_critic", "controller": "MADRL_BASE_SHARED_CRITIC", "label": "MADRL + Shared Centralized Critic + LSTM Forecast", "algo": "MATD3_SHARED_CRITIC", "projection": False, "reward": (0.0, 0.0, 0.0), "episodes": 50}


def _train_shared_critic_step(cfg: Cfg, actors_target: list[Actor], critic: SharedTwinCritic, critic_target: SharedTwinCritic, optim: torch.optim.Optimizer, batch: dict[str, Any], projector: JointGridSafetyProjector | None) -> float:
    obs, next_obs, action, reward, discount = batch["obs"], batch["next_obs"], batch["action"], batch["reward"], batch["bootstrap_discount"]
    with torch.no_grad():
        clean_next = _stack_actor_outputs(actors_target, next_obs)
        noisy_next = torch.clamp(clean_next + torch.clamp(torch.randn_like(clean_next) * float(cfg.algo.policy_noise), -float(cfg.algo.noise_clip), float(cfg.algo.noise_clip)), -1.0, 1.0)
        next_action = _guard_joint(cfg, next_obs, noisy_next, projector)
        q1_next, q2_next = critic_target(next_obs, next_action)
        if q1_next.shape != reward.shape or q2_next.shape != reward.shape:
            raise RuntimeError(f"Shared target critic expected Q shape {tuple(reward.shape)}, got {tuple(q1_next.shape)} and {tuple(q2_next.shape)}.")
        target_q = reward + discount * torch.minimum(q1_next, q2_next)
    q1, q2 = critic(obs, action)
    if q1.shape != target_q.shape or q2.shape != target_q.shape:
        raise RuntimeError(f"Shared critic expected Q shape {tuple(target_q.shape)}, got {tuple(q1.shape)} and {tuple(q2.shape)}.")
    loss = F.mse_loss(q1.float(), target_q.float()) + F.mse_loss(q2.float(), target_q.float())
    _step_optim(cfg, optim, loss, critic)
    return float(loss.detach().cpu())


def _train_shared_actor_step(cfg: Cfg, actor: Actor, agent_id: int, critic: SharedTwinCritic, optim: torch.optim.Optimizer, batch: dict[str, Any], projector: JointGridSafetyProjector | None) -> float:
    obs, action = batch["obs"], batch["action"]
    candidate = action.clone()
    idx = int(agent_id)
    candidate[:, idx] = map_actor_output_to_soc_feasible_action(cfg, obs["safety_local"][:, idx:idx + 1], actor(obs).unsqueeze(1))[:, 0]
    policy_action = enforce_local_action_feasibility(cfg, obs["safety_local"], projector.project_actions_from_safety_local(obs["safety_local"], candidate) if projector is not None else candidate)
    q1_policy, _ = critic(obs, policy_action)
    if q1_policy.ndim != 2 or q1_policy.shape[1] <= idx:
        raise RuntimeError(f"Shared actor loss expected per-agent Q matrix, got {tuple(q1_policy.shape)} for agent {idx}.")
    loss = -q1_policy[:, idx].float().mean()
    _step_optim(cfg, optim, loss, actor)
    return float(loss.detach().cpu())


def train_madrl_shared_critic_scheme(cfg: Cfg, run_dir: str | Path, share_data: ShareData, *, episodes: int | None = None) -> dict[str, Any]:
    spec = SHARED_CRITIC_SPEC; episodes = int(spec["episodes"] if episodes is None else episodes)
    work_cfg = _scheme_cfg(cfg, spec, episodes); device = resolve_device(work_cfg.runtime.device); set_seed(int(work_cfg.runtime.seed))
    actors = [actor.to(device) for actor in build_actors(work_cfg)]
    actors_target = deepcopy(actors)
    actor_optims = [torch.optim.Adam(actor.parameters(), lr=float(work_cfg.train.actor_lr)) for actor in actors]
    shared_critic = SharedTwinCritic(work_cfg).to(device); shared_critic_target = deepcopy(shared_critic)
    shared_critic_optim = torch.optim.Adam(shared_critic.parameters(), lr=float(work_cfg.train.critic_lr))
    projector = JointGridSafetyProjector(work_cfg).to(device) if bool(spec["projection"]) else None
    rng = np.random.default_rng(int(work_cfg.runtime.seed))
    env = _make_train_env(work_cfg, share_data)
    buffer = ReplayBuffer(work_cfg, int(work_cfg.train.num_envs))
    obs, _ = env.reset(); lane_rewards = np.zeros((int(work_cfg.train.num_envs),), dtype=np.float32)
    lane_components = {key: np.zeros((int(work_cfg.train.num_envs),), dtype=np.float32) for key in REWARD_COMPONENT_COLUMNS}
    completed = total_steps = updates = policy_pointer = 0; rows: list[dict[str, Any]] = []; loss_last = {"critic_loss": np.nan, "actor_loss": np.nan}; started = time.perf_counter()
    timers = {"action_sample_s": 0.0, "env_step_s": 0.0, "replay_add_s": 0.0, "sample_update_s": 0.0}
    progress = tqdm(total=episodes, desc=f"train {spec['scheme']}", unit="episode", ascii=True)
    while completed < episodes:
        tick = time.perf_counter()
        action = _sample_rollout_action(work_cfg, actors, obs, device, rng, total_steps, projector)
        timers["action_sample_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        next_obs, reward, done, _, infos = env.step(action)
        timers["env_step_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        buffer.add_batch(obs, action, reward, next_obs, done)
        timers["replay_add_s"] += time.perf_counter() - tick
        lane_rewards += np.sum(reward, axis=1).astype(np.float32)
        for lane, info in enumerate(infos):
            for key, sign, _, _ in REWARD_COMPONENT_SPECS:
                lane_components[key][lane] += _signed_reward_component(info, key, sign)
        obs = next_obs; total_steps += int(work_cfg.train.num_envs)
        if len(buffer) >= int(work_cfg.train.learning_starts) and total_steps % max(int(work_cfg.train.update_interval), 1) == 0:
            for _ in range(max(int(work_cfg.train.updates_per_step), 1)):
                tick = time.perf_counter()
                batch = _to_batch(buffer.sample(int(work_cfg.train.batch_size), rng), device)
                critic_loss = _train_shared_critic_step(work_cfg, actors_target, shared_critic, shared_critic_target, shared_critic_optim, batch, projector)
                actor_losses: list[float] = []; policy_pointer += 1
                if len(buffer) >= int(work_cfg.train.actor_learning_starts) and policy_pointer % int(work_cfg.algo.policy_update_freq) == 0:
                    actor_losses = [_train_shared_actor_step(work_cfg, actor, agent_id, shared_critic, actor_optims[agent_id], batch, projector) for agent_id, actor in enumerate(actors)]
                    for actor, target in zip(actors, actors_target, strict=True):
                        _soft_update(work_cfg, actor, target)
                _soft_update(work_cfg, shared_critic, shared_critic_target)
                loss_last = {"critic_loss": critic_loss, "actor_loss": _finite_mean(actor_losses)}
                timers["sample_update_s"] += time.perf_counter() - tick
                updates += 1
        if bool(np.all(done)):
            for lane in range(int(work_cfg.train.num_envs)):
                if completed >= episodes:
                    break
                completed += 1
                row = {"scheme": spec["scheme"], "controller": spec["controller"], "episode": completed, "total_reward": float(lane_rewards[lane]), **loss_last}
                row.update({key: float(lane_components[key][lane]) for key in REWARD_COMPONENT_COLUMNS})
                rows.append(row); lane_rewards[lane] = 0.0
                for key in REWARD_COMPONENT_COLUMNS:
                    lane_components[key][lane] = 0.0
            progress.update(min(int(np.sum(done)), episodes - progress.n)); progress.set_postfix(reward=f"{rows[-1]['total_reward']:.2f}", critic=f"{loss_last['critic_loss']:.3e}", refresh=False)
    progress.close(); env.close()
    n = int(work_cfg.env.num_agents)
    elapsed_s = float(time.perf_counter() - started); timed_s = float(sum(timers.values()))
    meta = {
        "scheme": spec["scheme"], "controller": spec["controller"], "label": spec["label"], "algo": spec["algo"],
        "critic_arch": "shared_twin_backbone_per_agent_heads", "projection": bool(spec["projection"]), "train_episodes": int(episodes),
        "steps": int(total_steps), "updates": int(updates), "elapsed_s": elapsed_s, "vec_env_kind": "subproc", "replay_buffer_kind": "array", **timers, "other_s": max(0.0, elapsed_s - timed_s),
        "batch_size": int(work_cfg.train.batch_size), "num_envs": int(work_cfg.train.num_envs),
        "parallel_episode_sampling": str(work_cfg.train.parallel_episode_sampling),
        "actor_param_count": int(sum(param.numel() for actor in actors for param in actor.parameters() if param.requires_grad)),
        "shared_critic_param_count": int(sum(param.numel() for param in shared_critic.parameters() if param.requires_grad)),
        "baseline_critic_param_count_estimate": int(n * sum(param.numel() for param in Critic(work_cfg, twin=True).parameters() if param.requires_grad)),
    }
    table_dir = Path(run_dir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"algo": row["controller"], "episode": row["episode"], "reward": row["total_reward"]} for row in rows]).to_csv(table_dir / f"learning_curve_{spec['scheme']}.csv", index=False)
    pd.DataFrame(rows).to_csv(table_dir / f"madrl_reward_curves_{spec['scheme']}.csv", index=False)
    for actor in actors:
        actor.eval()
    return {"cfg": work_cfg.with_algo(str(spec["controller"])), "actors": actors, "meta": meta, "reward_rows": rows}


def run_madrl_shared_critic_experiment(cfg: Cfg, run_dir: str | Path, share_data: ShareData, *, episodes: int | None = None) -> dict[str, Any]:
    trained = train_madrl_shared_critic_scheme(cfg, run_dir, share_data, episodes=episodes)
    spec = SHARED_CRITIC_SPEC
    controller = MADRLController(trained["cfg"], trained["actors"], name=str(spec["controller"]), projection=bool(spec["projection"]))
    rollout = collect_rollout(trained["cfg"], controller, share_data, forecast_mode="lstm", label=str(spec["label"]))
    saved = save_rollout(trained["cfg"], run_dir, rollout, scheme_name=str(spec["scheme"]))
    summary = pd.DataFrame([{**trained["meta"], "record_dir": str(saved["record_dir"])}])
    summary.to_csv(Path(run_dir) / "tables" / f"madrl_train_summary_{spec['scheme']}.csv", index=False)
    return {"records": {str(spec["scheme"]): saved}, "train_summary": summary, "reward_curves": pd.DataFrame(trained["reward_rows"])}
