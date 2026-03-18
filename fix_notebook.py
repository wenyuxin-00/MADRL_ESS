import json

ipynb_path = "notebooks/madrl/train_madrl_grid.ipynb"

with open(ipynb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("id") == "grid-madrl-10":
        new_source = [
            "from scripts.builder import build_env\n",
            "eval_env     = build_env(cfg, mode=\"test\")\n",
            "controller   = MADRLController(runner.agent_n, noise_std=0.0)\n",
            "reward_metas = eval_env.reward_fn.component_meta\n",
            "\n",
            "all_histories      = []\n",
            "all_grid_histories = []\n",
            "\n",
            "try:\n",
            "    for ep_i in range(n_eval_episodes):\n",
            "        obs_n = eval_env.reset()\n",
            "        controller.reset()\n",
            "\n",
            "        history      = init_episode_record(\n",
            "            n_agents=eval_env.n,\n",
            "            init_soc=float(eval_env.init_soc),\n",
            "            reward_metas=reward_metas,\n",
            "        )\n",
            "        grid_history = init_grid_record(n_agents=eval_env.n)\n",
            "\n",
            "        done = False\n",
            "        while not done:\n",
            "            action_n = controller.act(obs_n, deterministic=True)\n",
            "            obs_n, r_n, done_n, info = eval_env.step(action_n)\n",
            "\n",
            "            step_total = float(np.sum(np.asarray(r_n, dtype=np.float32)))\n",
            "            append_step_record(history, info, step_total=step_total, reward_metas=reward_metas)\n",
            "            append_grid_step_record(grid_history, info)\n",
            "\n",
            "            done = bool(info.get(\"episode_done\", False))\n",
            "\n",
            "        all_histories.append(history)\n",
            "        all_grid_histories.append(grid_history)\n",
            "\n",
            "        n_steps = len(grid_history[\"pf_converged\"])\n",
            "        n_conv  = sum(grid_history[\"pf_converged\"])\n",
            "        n_vviol = sum(grid_history[\"n_v_violations\"])\n",
            "        n_lviol = sum(grid_history[\"n_l_violations\"])\n",
            "        print(f\"Ep {ep_i+1:2d}: 步数={n_steps:3d}  潮流收敛={n_conv}/{n_steps}  \"\n",
            "              f\"电压违规={n_vviol:3d} 步  线路违规={n_lviol:3d} 步\")\n",
            "finally:\n",
            "    eval_env.close()\n",
            "\n",
            "total_v = sum(sum(gh[\"n_v_violations\"]) for gh in all_grid_histories)\n",
            "total_l = sum(sum(gh[\"n_l_violations\"]) for gh in all_grid_histories)\n",
            "print(f\"\\n汇总 — 电压违规总步数: {total_v},  线路违规总步数: {total_l}\")"
        ]
        cell["source"] = new_source
        break

with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
