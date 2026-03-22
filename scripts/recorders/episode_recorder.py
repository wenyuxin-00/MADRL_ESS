"""Episode 数据记录器。

记录每个 episode 的状态、动作、奖励等轨迹数据。

主要类:
    EpisodeRecorder -- Episode 数据记录器
"""

from __future__ import annotations

import numpy as np


def init_episode_record(n_agents: int, init_soc: float, reward_metas) -> dict:
    """Create an episode history dict compatible with existing plotting code."""
    history = {
        "price": [],
        "base_net_load": [[] for _ in range(int(n_agents))],
        "e_bat_req": [[] for _ in range(int(n_agents))],
        "e_bat_exec": [[] for _ in range(int(n_agents))],
        "soc": [[float(init_soc)] for _ in range(int(n_agents))],
        "r_total_sum": [],
        "r_total_per_agent": [[] for _ in range(int(n_agents))],
    }
    for meta in reward_metas:
        history[f"{meta.key}_sum"] = []
        history[f"{meta.key}_per_agent"] = [[] for _ in range(int(n_agents))]
    return history


def append_step_record(history: dict, info: dict, step_total: float, reward_metas) -> None:
    """Append one env step to an episode history dict."""
    n_agents = len(history["e_bat_req"])
    history["price"].append(float(info.get("price", 0.0)))

    base_net_load = np.asarray(info.get("base_net_load", np.zeros(n_agents)), dtype=np.float32).reshape(
        n_agents
    )
    e_req = np.asarray(info["e_bat_req"], dtype=np.float32).reshape(n_agents)
    e_exec = np.asarray(info["e_bat"], dtype=np.float32).reshape(n_agents)
    soc_next = np.asarray(info["soc_next"], dtype=np.float32).reshape(n_agents)
    reward_per_agent = np.asarray(info.get("reward", np.zeros(n_agents)), dtype=np.float32).reshape(
        n_agents
    )

    for agent_id in range(n_agents):
        history["base_net_load"][agent_id].append(float(base_net_load[agent_id]))
        history["e_bat_req"][agent_id].append(float(e_req[agent_id]))
        history["e_bat_exec"][agent_id].append(float(e_exec[agent_id]))
        history["soc"][agent_id].append(float(soc_next[agent_id]))
        history["r_total_per_agent"][agent_id].append(float(reward_per_agent[agent_id]))

    history["r_total_sum"].append(float(step_total))

    for meta in reward_metas:
        raw_array = np.asarray(info[meta.key], dtype=np.float32).reshape(n_agents)
        raw_value = float(np.sum(raw_array))
        history[f"{meta.key}_sum"].append(meta.sign * raw_value)
        for agent_id in range(n_agents):
            history[f"{meta.key}_per_agent"][agent_id].append(float(meta.sign * raw_array[agent_id]))
