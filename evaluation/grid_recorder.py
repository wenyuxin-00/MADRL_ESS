"""Grid-specific episode history recorder.

Runs **alongside** the standard ``episode_recorder`` (which tracks battery
and reward data).  ``GridRecorder`` captures power-flow fields that are
specific to ``GridEnv`` and are not part of the ComponentMeta auto-flow.

Typical usage (notebook or evaluator)::

    from evaluation.episode_recorder import init_episode_record, append_step_record
    from evaluation.grid_recorder import init_grid_record, append_grid_step_record

    reward_metas = env.reward_fn.component_meta
    history = init_episode_record(n_agents, init_soc, reward_metas)
    grid_history = init_grid_record(n_agents=n_agents)

    obs = env.reset()
    while True:
        action_n = controller.act(obs)
        obs, r_n, done_n, info = env.step(action_n)
        append_step_record(history, info, step_total=sum(r_n), reward_metas=reward_metas)
        append_grid_step_record(grid_history, info)
        if any(d["episode_done"] for d in done_n):
            break
"""

from __future__ import annotations

import numpy as np


def init_grid_record(n_agents: int) -> dict:
    """Create an empty grid history dict for one episode.

    Parameters
    ----------
    n_agents:
        Number of RL agents (= number of agent buses tracked).

    Returns
    -------
    dict with keys:
        ``agent_vm_pu``     — list of per-agent voltage lists, shape (n_agents, T).
        ``line_loading_pct``— list of full line-loading arrays, one per step.
        ``n_v_violations``  — list of violation counts (int) per step.
        ``n_l_violations``  — list of line-overload flags (int, 0 or 1) per step.
        ``pf_converged``    — list of convergence bools per step.
    """
    return {
        "agent_vm_pu": [[] for _ in range(int(n_agents))],
        "line_loading_pct": [],
        "n_v_violations": [],
        "n_l_violations": [],
        "pf_converged": [],
    }


def append_grid_step_record(grid_history: dict, info: dict) -> None:
    """Append one environment step to a grid history dict.

    Silently skips if ``info`` does not contain ``"agent_vm_pu"`` (e.g.,
    when called on a non-grid environment).

    Parameters
    ----------
    grid_history:
        Dict created by :func:`init_grid_record`.
    info:
        The ``info`` dict returned by ``GridEnv.step()``.
    """
    if "agent_vm_pu" not in info:
        return

    vm = np.asarray(info["agent_vm_pu"], dtype=np.float32)
    n_agents = len(grid_history["agent_vm_pu"])
    for i in range(n_agents):
        grid_history["agent_vm_pu"][i].append(float(vm[i]))

    line_loading = info.get("line_loading_pct")
    if line_loading is not None:
        grid_history["line_loading_pct"].append(
            np.asarray(line_loading, dtype=np.float32).copy()
        )
    else:
        grid_history["line_loading_pct"].append(np.zeros(0, dtype=np.float32))

    grid_history["n_v_violations"].append(int(info.get("n_v_violations", 0)))
    grid_history["n_l_violations"].append(int(info.get("n_l_violations", 0)))
    grid_history["pf_converged"].append(bool(info.get("pf_converged", True)))
