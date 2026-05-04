from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
import json, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from REMAKE.configs.cfg import Cfg
from REMAKE.controllers.madrl import MADRLController
from REMAKE.controllers.madrl_safety import JointGridSafetyProjector, enforce_local_action_feasibility, local_bounds_torch, map_actor_output_to_soc_feasible_action
from REMAKE.data.share_data import ShareData
from REMAKE.envs.grid_env import build_env
from REMAKE.envs.vec_env import SyncVecEnv
from REMAKE.models.assembly import Actor, Critic, build_actors, build_critics, to_torch_obs
from REMAKE.utils.records import collect_rollout, load_record, plot_basic_records, save_rollout
from REMAKE.utils.torch_runtime import resolve_device, set_seed

SCHEMES = (
    {"scheme": "madrl_base", "controller": "MADRL_BASE", "label": "MADRL + No Safety + LSTM Forecast", "algo": "MATD3", "projection": False, "reward": (0.0, 0.0, 0.0), "episodes": 500},
    {"scheme": "madrl_base_safe", "controller": "MADRL_PENALTY", "label": "MADRL + Safety Penalty + LSTM Forecast", "algo": "MATD3", "projection": False, "reward": (400.0, 0.0, 10.0), "episodes": 500},
    {"scheme": "madrl_projection_safe", "controller": "MADRL_PROJECTION", "label": "MADRL + Safety Projection + LSTM Forecast", "algo": "MATD3_SAFE_POC", "projection": True, "reward": (400.0, 0.0, 10.0), "episodes": 500},
)
OBS_KEYS = ("madrl_local", "safety_local", "wholesale_price_relative_seq", "wholesale_price_spread_seq", "load_seq", "pv_seq")


def _scheme_cfg(cfg: Cfg, spec: dict[str, Any], episodes: int) -> Cfg:
    w_v, w_l, w_t = spec["reward"]
    return replace(cfg, algo=replace(cfg.algo, name=str(spec["algo"])), train=replace(cfg.train, train_episodes=int(episodes)), reward=replace(cfg.reward, w_voltage_pen=float(w_v), w_line_pen=float(w_l), w_trafo_pen=float(w_t)), safety=replace(cfg.safety, enabled=bool(spec["projection"])))


def _slice_obs(obs: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[idx], dtype=np.float32).copy() for key, value in obs.items()}


def _stack_obs(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([item[key] for item in items]).astype(np.float32) for key in OBS_KEYS}


class ReplayBuffer:
    def __init__(self, cfg: Cfg, num_envs: int) -> None:
        self.capacity, self.gamma, self.n_step = int(cfg.train.buffer_size), float(cfg.algo.gamma), max(int(cfg.train.n_step_return), 1)
        self.items: list[dict[str, Any]] = []; self.pos = 0; self.queues = [deque() for _ in range(int(num_envs))]

    def __len__(self) -> int:
        return len(self.items)

    def _store(self, item: dict[str, Any]) -> None:
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.pos] = item; self.pos = (self.pos + 1) % self.capacity

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
        idxs = rng.integers(0, len(self.items), size=int(batch_size))
        rows = [self.items[int(idx)] for idx in idxs]
        return {
            "obs": _stack_obs([row["obs"] for row in rows]), "next_obs": _stack_obs([row["next_obs"] for row in rows]),
            "action": np.stack([row["action"] for row in rows]).astype(np.float32), "reward": np.stack([row["reward"] for row in rows]).astype(np.float32),
            "terminated": np.stack([row["terminated"] for row in rows]).astype(np.float32), "bootstrap_discount": np.stack([row["bootstrap_discount"] for row in rows]).astype(np.float32),
        }


class Agent:
    def __init__(self, cfg: Cfg, agent_id: int, device: torch.device) -> None:
        self.cfg, self.agent_id, self.device = cfg, int(agent_id), device
        self.actor = Actor(cfg, agent_id).to(device); self.critic = Critic(cfg, twin=True).to(device)
        self.actor_target = deepcopy(self.actor); self.critic_target = deepcopy(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr)); self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))
        self.policy_pointer = 0

    def soft_update(self) -> None:
        tau = float(self.cfg.algo.tau)
        for src, dst in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
            for src_p, dst_p in zip(src.parameters(), dst.parameters(), strict=True):
                dst_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)

    def step_optim(self, optim: torch.optim.Optimizer, loss: torch.Tensor, module: torch.nn.Module) -> None:
        optim.zero_grad(set_to_none=True); loss.backward()
        if bool(self.cfg.model.use_grad_clip):
            torch.nn.utils.clip_grad_norm_(module.parameters(), float(self.cfg.model.grad_clip_norm))
        optim.step()


def _to_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {"obs": to_torch_obs(batch["obs"], device), "next_obs": to_torch_obs(batch["next_obs"], device), **{key: torch.as_tensor(batch[key], dtype=torch.float32, device=device) for key in ("action", "reward", "terminated", "bootstrap_discount")}}


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _stack_actor_outputs(actors: list[Actor], obs: dict[str, torch.Tensor], target: bool = False) -> torch.Tensor:
    return torch.stack([actor(obs) for actor in actors], dim=1)


def _guard_joint(cfg: Cfg, obs: dict[str, torch.Tensor], raw: torch.Tensor, projector: JointGridSafetyProjector | None = None) -> torch.Tensor:
    action = map_actor_output_to_soc_feasible_action(cfg, obs["safety_local"], raw)
    if projector is not None:
        action = projector.project_actions_from_safety_local(obs["safety_local"], action)
    return enforce_local_action_feasibility(cfg, obs["safety_local"], action)


def _sample_rollout_action(cfg: Cfg, agents: list[Agent], obs: dict[str, np.ndarray], device: torch.device, rng: np.random.Generator, step: int, projector: JointGridSafetyProjector | None) -> np.ndarray:
    obs_t = to_torch_obs({key: obs[key] for key in OBS_KEYS}, device)
    with torch.no_grad():
        raw = _stack_actor_outputs([agent.actor for agent in agents], obs_t)
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
    agent.step_optim(agent.critic_optim, critic_loss, agent.critic)
    actor_loss_value = float("nan")
    if allow_actor_update and agent.policy_pointer % int(cfg.algo.policy_update_freq) == 0:
        candidate = action.clone()
        candidate[:, agent.agent_id] = map_actor_output_to_soc_feasible_action(cfg, obs["safety_local"][:, agent.agent_id:agent.agent_id + 1], agent.actor(obs).unsqueeze(1))[:, 0]
        policy_action = enforce_local_action_feasibility(cfg, obs["safety_local"], projector.project_actions_from_safety_local(obs["safety_local"], candidate) if projector is not None else candidate)
        q1_policy, _ = agent.critic(obs, policy_action)
        actor_loss = -q1_policy.float().mean(); agent.step_optim(agent.actor_optim, actor_loss, agent.actor); agent.soft_update()
        actor_loss_value = float(actor_loss.detach().cpu())
    elif not allow_actor_update:
        agent.soft_update()
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
    env = SyncVecEnv(int(work_cfg.train.num_envs), lambda: build_env(work_cfg, "train", forecast_mode="lstm", share_data=share_data), seed=int(work_cfg.runtime.seed), parallel_episode_sampling=str(work_cfg.train.parallel_episode_sampling))
    buffer = ReplayBuffer(work_cfg, int(work_cfg.train.num_envs))
    obs, _ = env.reset(); lane_rewards = np.zeros((int(work_cfg.train.num_envs),), dtype=np.float32)
    completed = total_steps = updates = 0; rows: list[dict[str, Any]] = []; loss_last = {"critic_loss": np.nan, "actor_loss": np.nan}; started = time.perf_counter()
    progress = tqdm(total=episodes, desc=f"train {spec['scheme']}", unit="episode", ascii=True)
    while completed < episodes:
        action = _sample_rollout_action(work_cfg, agents, obs, device, rng, total_steps, projector)
        next_obs, reward, done, _, _ = env.step(action)
        buffer.add_batch(obs, action, reward, next_obs, done)
        lane_rewards += np.sum(reward, axis=1).astype(np.float32); obs = next_obs; total_steps += int(work_cfg.train.num_envs)
        if len(buffer) >= int(work_cfg.train.learning_starts) and total_steps % max(int(work_cfg.train.update_interval), 1) == 0:
            for _ in range(max(int(work_cfg.train.updates_per_step), 1)):
                batch = _to_batch(buffer.sample(int(work_cfg.train.batch_size), rng), device); allow_actor = len(buffer) >= int(work_cfg.train.actor_learning_starts)
                metrics = [_train_agent(work_cfg, agents, agent, batch, projector, allow_actor) for agent in agents]
                loss_last = {"critic_loss": _finite_mean([item["critic_loss"] for item in metrics]), "actor_loss": _finite_mean([item["actor_loss"] for item in metrics])}
                updates += 1
        if bool(np.all(done)):
            for lane in range(int(work_cfg.train.num_envs)):
                if completed >= episodes:
                    break
                completed += 1
                rows.append({"scheme": spec["scheme"], "controller": spec["controller"], "episode": completed, "total_reward": float(lane_rewards[lane]), **loss_last})
                lane_rewards[lane] = 0.0
            progress.update(min(int(np.sum(done)), episodes - progress.n)); progress.set_postfix(reward=f"{rows[-1]['total_reward']:.2f}", critic=f"{loss_last['critic_loss']:.3e}", refresh=False)
    progress.close(); env.close()
    meta = {"scheme": spec["scheme"], "controller": spec["controller"], "label": spec["label"], "algo": spec["algo"], "projection": bool(spec["projection"]), "episodes": int(episodes), "steps": int(total_steps), "updates": int(updates), "elapsed_s": float(time.perf_counter() - started), "batch_size": int(work_cfg.train.batch_size), "num_envs": int(work_cfg.train.num_envs), "parallel_episode_sampling": str(work_cfg.train.parallel_episode_sampling)}
    model_path = _save_model(work_cfg, agents, Path(run_dir) / "models" / "madrl" / str(spec["scheme"]), meta)
    table_dir = Path(run_dir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"algo": row["controller"], "episode": row["episode"], "reward": row["total_reward"]} for row in rows]).to_csv(table_dir / f"learning_curve_{spec['scheme']}.csv", index=False)
    pd.DataFrame(rows).to_csv(table_dir / f"madrl_reward_curves_{spec['scheme']}.csv", index=False)
    return {"cfg": work_cfg.with_algo(str(spec["controller"])), "model_path": model_path, "meta": meta, "reward_rows": rows}


def _refresh_madrl_aggregate_tables(run_dir: str | Path) -> None:
    table_dir = Path(run_dir) / "tables"
    summary_paths = [table_dir / f"madrl_train_summary_{spec['scheme']}.csv" for spec in SCHEMES]
    reward_paths = [table_dir / f"madrl_reward_curves_{spec['scheme']}.csv" for spec in SCHEMES]
    learning_paths = [table_dir / f"learning_curve_{spec['scheme']}.csv" for spec in SCHEMES]
    if all(path.exists() for path in summary_paths):
        pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True).to_csv(table_dir / "madrl_train_summary.csv", index=False)
    if all(path.exists() for path in reward_paths):
        pd.concat([pd.read_csv(path) for path in reward_paths], ignore_index=True).to_csv(table_dir / "madrl_reward_curves.csv", index=False)
    if all(path.exists() for path in learning_paths):
        pd.concat([pd.read_csv(path) for path in learning_paths], ignore_index=True).to_csv(table_dir / "learning_curves.csv", index=False)


def run_madrl_scheme_experiment(cfg: Cfg, run_dir: str | Path, share_data: ShareData, spec: dict[str, Any], *, episodes: int | None = None) -> dict[str, Any]:
    trained = train_madrl_scheme(cfg, run_dir, share_data, spec, episodes=episodes)
    controller = MADRLController.load(trained["cfg"], trained["model_path"])
    rollout = collect_rollout(trained["cfg"], controller, share_data, forecast_mode="lstm", label=str(spec["label"]))
    saved = save_rollout(trained["cfg"], run_dir, rollout, scheme_name=str(spec["scheme"]))
    summary = pd.DataFrame([{**trained["meta"], "model_path": str(trained["model_path"]), "record_dir": str(saved["record_dir"])}])
    summary.to_csv(Path(run_dir) / "tables" / f"madrl_train_summary_{spec['scheme']}.csv", index=False)
    _refresh_madrl_aggregate_tables(run_dir)
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
    _refresh_madrl_aggregate_tables(run_dir)
    return {"records": records, "train_summary": pd.DataFrame(summaries), "reward_curves": pd.DataFrame(rewards)}


def load_madrl_record(run_dir: str | Path, controller: str) -> dict[str, Any]:
    return load_record(run_dir, "lstm", controller)


def plot_madrl_outputs(result: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    rollouts = {name: saved["rollout"] for name, saved in result["records"].items()}
    prefix = next(iter(rollouts)) if len(rollouts) == 1 else "madrl"
    figures = plot_basic_records(rollouts, run_dir, prefix)
    rewards = result.get("reward_curves", pd.DataFrame())
    if not rewards.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        for scheme, group in rewards.groupby("scheme"):
            ax.plot(group["episode"], group["total_reward"], label=scheme)
        ax.legend(fontsize=7); ax.set_ylabel("episode reward"); fig.tight_layout()
        out = Path(run_dir) / "figures"; out.mkdir(parents=True, exist_ok=True); fig.savefig(out / f"{prefix}_learning_curve.png", dpi=140); figures[f"{prefix}_learning_curve"] = fig
    return figures
