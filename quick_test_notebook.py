import json

ipynb_path = "notebooks/madrl/train_madrl_grid.ipynb"

with open(ipynb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("id") == "grid-madrl-03":
        # Modify train episodes for quick test
        cell["source"] = [
            "# ── 训练规模 ──────────────────────────────────────────────────\n",
            "algorithm          = \"MADDPG\"   # MADDPG | MATD3\n",
            "train_episodes     = 2          # Quick test\n",
            "n_eval_episodes    = 1          # Quick test\n",
            "reward_plot_window = 20         # 奖励移动平均窗口\n",
            "n_recent_to_plot   = 2          # 训练轨迹图展示的最近 episode 数\n",
            "\n",
            "# ── 运行时 ────────────────────────────────────────────────────\n",
            "seed           = 0\n",
            "runtime_mode   = \"performance\"  # performance | strict_reproducibility\n",
            "device_request = None           # None = 自动选 CUDA\n",
            "require_cuda   = False"
        ]
        break

with open("notebooks/madrl/train_madrl_grid_quick_test.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
