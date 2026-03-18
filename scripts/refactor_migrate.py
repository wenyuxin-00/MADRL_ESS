"""Full project refactoring migration script.

Executes Phases 0-7 of the MADRL_ESS restructuring plan:
  Phase 0: Merge grid_profiles.py into profiles.py (configs cleanup)
  Phase 1: Merge data/ + datasets/ -> data/
  Phase 2: Merge grid/ -> envs/grid/, common/rewards -> envs/rewards/
  Phase 3: Merge algorithms/ -> controllers/madrl/
  Phase 4: Rename forecast/ -> predictors/
  Phase 5: Consolidate core/ + runners/ + evaluation/ + common/ -> scripts/
  Phase 6: Global import path updates
  Phase 7: Cleanup

Usage::
    python scripts/refactor_migrate.py
"""
import os
import sys
import shutil
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path):
    """Move a file, creating parent dirs as needed."""
    if not src.exists():
        print(f"  [SKIP] {src} does not exist")
        return
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    print(f"  [MOVE] {src} -> {dst}")


def move_dir(src: Path, dst: Path):
    """Move a directory, creating parent dirs as needed."""
    if not src.exists():
        print(f"  [SKIP] {src} does not exist")
        return
    ensure_dir(dst.parent)
    if dst.exists():
        # Merge into existing dir
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                move_dir(item, target)
            else:
                move_file(item, target)
        # Remove empty source
        try:
            src.rmdir()
        except OSError:
            pass
    else:
        shutil.move(str(src), str(dst))
    print(f"  [MOVE_DIR] {src} -> {dst}")


def copy_file(src: Path, dst: Path):
    """Copy a file, creating parent dirs as needed."""
    if not src.exists():
        print(f"  [SKIP] {src} does not exist")
        return
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))
    print(f"  [COPY] {src} -> {dst}")


def rm_tree(p: Path):
    """Remove a directory tree."""
    if p.exists():
        shutil.rmtree(str(p))
        print(f"  [DELETE] {p}")


def rm_file(p: Path):
    """Remove a single file."""
    if p.exists():
        p.unlink()
        print(f"  [DELETE] {p}")


def write_init(p: Path, content: str = ""):
    """Create/overwrite an __init__.py."""
    ensure_dir(p.parent)
    p.write_text(content, encoding="utf-8")
    print(f"  [WRITE] {p}")


# ─── IMPORT REPLACEMENT MAP ──────────────────────────────────────────────
# Each entry: (old_pattern, new_string)
# Applied to all .py files (including tests) in Phase 6.
IMPORT_REPLACEMENTS = [
    # Phase 0 - grid_profiles
    ("from configs.profiles import", "from configs.profiles import"),
    ("from configs.profiles ", "from configs.profiles "),
    ("import configs.profiles", "import configs.profiles"),

    # Phase 1 - datasets -> data.loaders
    ("from data.loaders.registry import", "from data.loaders.registry import"),
    ("from data.loaders.registry ", "from data.loaders.registry "),
    ("from data.loaders.base import", "from data.loaders.base import"),
    ("from data.loaders.base ", "from data.loaders.base "),
    ("from data.loaders.csv_price_load import", "from data.loaders.csv_price_load import"),
    ("from data.loaders.csv_price_load ", "from data.loaders.csv_price_load "),
    ("from data.loaders.csv_prosumer import", "from data.loaders.csv_prosumer import"),
    ("from data.loaders.csv_prosumer ", "from data.loaders.csv_prosumer "),
    ("from data.loaders.simbench_export import", "from data.loaders.simbench_export import"),
    ("from data.loaders.simbench_export ", "from data.loaders.simbench_export "),
    ("from data.loaders import", "from data.loaders import"),
    ("import data.loaders.", "import data.loaders."),
    ("import data.loaders", "import data.loaders"),

    # Phase 2 - grid -> envs.grid
    ("from envs.grid.core.grid_core import", "from envs.grid.core.grid_core import"),
    ("from envs.grid.core.grid_core ", "from envs.grid.core.grid_core "),
    ("from envs.grid.core.grid_types import", "from envs.grid.core.grid_types import"),
    ("from envs.grid.core.grid_types ", "from envs.grid.core.grid_types "),
    ("from envs.grid.core.net_builder import", "from envs.grid.core.net_builder import"),
    ("from envs.grid.core.net_builder ", "from envs.grid.core.net_builder "),
    ("from envs.grid.core import", "from envs.grid.core import"),
    ("from envs.grid.config.grid_config import", "from envs.grid.config.grid_config import"),
    ("from envs.grid.config.grid_config ", "from envs.grid.config.grid_config "),
    ("from envs.grid.config import", "from envs.grid.config import"),
    ("from envs.grid.topology.rural1_fixed import", "from envs.grid.topology.rural1_fixed import"),
    ("from envs.grid.topology.rural1_fixed ", "from envs.grid.topology.rural1_fixed "),
    ("from envs.grid.topology import", "from envs.grid.topology import"),
    ("from envs.grid import", "from envs.grid import"),
    ("import envs.grid.", "import envs.grid."),
    ("import envs.grid", "import envs.grid"),

    # Phase 2 - common.rewards -> envs.rewards
    ("from envs.rewards import", "from envs.rewards import"),
    ("from envs.rewards.", "from envs.rewards."),
    ("import envs.rewards", "import envs.rewards"),

    # Phase 2 - common.vec_env / subproc_vec_env -> envs.*
    ("from envs.vec_env import", "from envs.vec_env import"),
    ("from envs.vec_env ", "from envs.vec_env "),
    ("from envs.subproc_vec_env import", "from envs.subproc_vec_env import"),
    ("from envs.subproc_vec_env ", "from envs.subproc_vec_env "),
    ("import envs.vec_env", "import envs.vec_env"),
    ("import envs.subproc_vec_env", "import envs.subproc_vec_env"),

    # Phase 3 - algorithms -> controllers.madrl
    ("from controllers.madrl.registry import", "from controllers.madrl.registry import"),
    ("from controllers.madrl.registry ", "from controllers.madrl.registry "),
    ("from controllers.madrl.base_agent import", "from controllers.madrl.base_agent import"),
    ("from controllers.madrl.base_agent ", "from controllers.madrl.base_agent "),
    ("from controllers.madrl.maddpg import", "from controllers.madrl.maddpg import"),
    ("from controllers.madrl.maddpg ", "from controllers.madrl.maddpg "),
    ("from controllers.madrl.matd3 import", "from controllers.madrl.matd3 import"),
    ("from controllers.madrl.matd3 ", "from controllers.madrl.matd3 "),
    ("from controllers.madrl import", "from controllers.madrl import"),
    ("import controllers.madrl.", "import controllers.madrl."),
    ("import controllers.madrl", "import controllers.madrl"),

    # Phase 3 - controllers.mpc_controller -> controllers.mpc.mpc_controller
    ("from controllers.mpc.mpc_controller import", "from controllers.mpc.mpc_controller import"),
    ("from controllers.mpc.mpc_controller ", "from controllers.mpc.mpc_controller "),
    ("import controllers.mpc.mpc_controller", "import controllers.mpc.mpc_controller"),

    # Phase 3 - controllers.classic_drl_controller -> controllers.drl.classic_drl_controller
    ("from controllers.drl.classic_drl_controller import", "from controllers.drl.classic_drl_controller import"),
    ("from controllers.drl.classic_drl_controller ", "from controllers.drl.classic_drl_controller "),
    ("import controllers.drl.classic_drl_controller", "import controllers.drl.classic_drl_controller"),

    # Phase 4 - forecast -> predictors
    ("from predictors.registry import", "from predictors.registry import"),
    ("from predictors.registry ", "from predictors.registry "),
    ("from predictors.training import", "from predictors.training import"),
    ("from predictors.training ", "from predictors.training "),
    ("from predictors.base import", "from predictors.base import"),
    ("from predictors.base ", "from predictors.base "),
    ("from predictors.naive import", "from predictors.naive import"),
    ("from predictors.naive ", "from predictors.naive "),
    ("from predictors.oracle import", "from predictors.oracle import"),
    ("from predictors.oracle ", "from predictors.oracle "),
    ("from predictors.lstm_forecaster import", "from predictors.lstm_forecaster import"),
    ("from predictors.lstm_forecaster ", "from predictors.lstm_forecaster "),
    ("from predictors.lstm_model import", "from predictors.lstm_model import"),
    ("from predictors.lstm_model ", "from predictors.lstm_model "),
    ("from predictors.artifacts import", "from predictors.artifacts import"),
    ("from predictors.artifacts ", "from predictors.artifacts "),
    ("from predictors import", "from predictors import"),
    ("import predictors.", "import predictors."),
    ("import predictors", "import predictors"),

    # Phase 5 - core.builder -> scripts.builder
    ("from scripts.builder import", "from scripts.builder import"),
    ("from scripts.builder ", "from scripts.builder "),
    ("from scripts import", "from scripts import"),
    ("import scripts.builder", "import scripts.builder"),
    ("import scripts", "import scripts"),

    # Phase 5 - runners -> scripts
    ("from scripts.train import", "from scripts.train import"),
    ("from scripts.train ", "from scripts.train "),
    ("from scripts.checkpoints import", "from scripts.checkpoints import"),
    ("from scripts.checkpoints ", "from scripts.checkpoints "),
    ("from scripts import", "from scripts import"),
    ("import scripts.train", "import scripts.train"),
    ("import scripts.checkpoints", "import scripts.checkpoints"),
    ("import scripts", "import scripts"),

    # Phase 5 - evaluation -> scripts
    ("from scripts.evaluate import", "from scripts.evaluate import"),
    ("from scripts.evaluate ", "from scripts.evaluate "),
    ("from scripts.comparison import", "from scripts.comparison import"),
    ("from scripts.comparison ", "from scripts.comparison "),
    ("from scripts.plots.plots import", "from scripts.plots.plots import"),
    ("from scripts.plots.plots ", "from scripts.plots.plots "),
    ("from scripts.plots.grid_plots import", "from scripts.plots.grid_plots import"),
    ("from scripts.plots.grid_plots ", "from scripts.plots.grid_plots "),
    ("from scripts.plots.reward_plots import", "from scripts.plots.reward_plots import"),
    ("from scripts.plots.reward_plots ", "from scripts.plots.reward_plots "),
    ("from scripts.recorders.episode_recorder import", "from scripts.recorders.episode_recorder import"),
    ("from scripts.recorders.episode_recorder ", "from scripts.recorders.episode_recorder "),
    ("from scripts.recorders.grid_recorder import", "from scripts.recorders.grid_recorder import"),
    ("from scripts.recorders.grid_recorder ", "from scripts.recorders.grid_recorder "),
    ("from scripts import", "from scripts import"),
    ("import scripts.", "import scripts."),
    ("import scripts", "import scripts"),

    # Phase 5 - common misc -> scripts.utils
    ("from scripts.utils.project_paths import", "from scripts.utils.project_paths import"),
    ("from scripts.utils.project_paths ", "from scripts.utils.project_paths "),
    ("from scripts.utils.nested import", "from scripts.utils.nested import"),
    ("from scripts.utils.nested ", "from scripts.utils.nested "),
    ("from scripts.utils.torch_runtime import", "from scripts.utils.torch_runtime import"),
    ("from scripts.utils.torch_runtime ", "from scripts.utils.torch_runtime "),
    ("from scripts.utils.replay_buffer import", "from scripts.utils.replay_buffer import"),
    ("from scripts.utils.replay_buffer ", "from scripts.utils.replay_buffer "),
    ("from scripts.utils.experiment_notebook_utils import", "from scripts.utils.experiment_notebook_utils import"),
    ("from scripts.utils.experiment_notebook_utils ", "from scripts.utils.experiment_notebook_utils "),
    ("from scripts.utils import", "from scripts.utils import"),
    ("import scripts.utils.", "import scripts.utils."),
    ("import scripts.utils", "import scripts.utils"),
]


def apply_import_replacements(filepath: Path):
    """Apply all import replacements to a single Python file."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return False

    original = text
    for old, new in IMPORT_REPLACEMENTS:
        text = text.replace(old, new)

    if text != original:
        filepath.write_text(text, encoding="utf-8")
        print(f"  [IMPORT] Updated imports in {filepath}")
        return True
    return False


def collect_py_files(root: Path, exclude_dirs=None) -> list[Path]:
    """Collect all .py files under root, excluding specified directories."""
    if exclude_dirs is None:
        exclude_dirs = {".git", ".gitnexus", "__pycache__", ".claude", ".vscode",
                        "artifacts", ".jupyter_runtime", "node_modules"}
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            if f.endswith(".py"):
                result.append(Path(dirpath) / f)
    return result


# ═════════════════════════════════════════════════════════════════════════
# PHASE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════

def phase_0():
    """Phase 0: Merge grid_profiles.py into profiles.py in configs/."""
    print("\n" + "="*60)
    print("PHASE 0: Merge grid_profiles.py -> profiles.py")
    print("="*60)

    # profiles.py was already edited (grid functions appended).
    # Now we need to add `from typing import Any` if not present.
    profiles = ROOT / "configs" / "profiles.py"
    text = profiles.read_text(encoding="utf-8")
    if "from typing import Any" not in text:
        text = text.replace(
            "import os\r\n",
            "import os\r\nfrom typing import Any\r\n",
            1
        )
        if "from typing import Any" not in text:
            # try unix line endings
            text = text.replace(
                "import os\n",
                "import os\nfrom typing import Any\n",
                1
            )
        profiles.write_text(text, encoding="utf-8")
        print("  [FIX] Added 'from typing import Any' to profiles.py")

    # Update __init__.py to export apply_grid_profile
    init_path = ROOT / "configs" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    if "apply_grid_profile" not in init_text:
        # Add to imports from profiles
        init_text = init_text.replace(
            "from configs.profiles import (\n",
            "from configs.profiles import (\n    apply_grid_profile,\n"
        )
        init_text = init_text.replace(
            "from configs.profiles import (\r\n",
            "from configs.profiles import (\r\n    apply_grid_profile,\r\n"
        )
        # Add to __all__
        init_text = init_text.replace(
            '    "apply_forecast_profile",\n',
            '    "apply_forecast_profile",\n    "apply_grid_profile",\n'
        )
        init_text = init_text.replace(
            '    "apply_forecast_profile",\r\n',
            '    "apply_forecast_profile",\r\n    "apply_grid_profile",\r\n'
        )
        init_path.write_text(init_text, encoding="utf-8")
        print("  [UPDATE] configs/__init__.py: added apply_grid_profile")

    # Delete grid_profiles.py
    rm_file(ROOT / "configs" / "grid_profiles.py")

    print("Phase 0 DONE\n")


def phase_1():
    """Phase 1: Merge data/ + datasets/ -> data/."""
    print("\n" + "="*60)
    print("PHASE 1: Merge data/ + datasets/ -> data/")
    print("="*60)

    data_dir = ROOT / "data"
    raw_dir = data_dir / "raw"
    loaders_dir = data_dir / "loaders"

    ensure_dir(raw_dir)
    ensure_dir(loaders_dir)

    # Move raw data files into data/raw/
    data_files = [f for f in data_dir.iterdir()
                  if f.is_file() and f.suffix in ('.csv', '.xlsx', '.json')
                  and f.name != '__init__.py']
    for f in data_files:
        move_file(f, raw_dir / f.name)

    # Move data_process.ipynb to notebooks/data/
    nb_data = ROOT / "notebooks" / "data"
    ensure_dir(nb_data)
    ipynb = data_dir / "data_process.ipynb"
    if ipynb.exists():
        move_file(ipynb, nb_data / "data_process.ipynb")

    # Move datasets/ Python files to data/loaders/
    datasets_dir = ROOT / "datasets"
    if datasets_dir.exists():
        for f in datasets_dir.iterdir():
            if f.is_file() and f.suffix == ".py":
                move_file(f, loaders_dir / f.name)
        # Remove __pycache__ and empty dir
        pycache = datasets_dir / "__pycache__"
        rm_tree(pycache)
        try:
            datasets_dir.rmdir()
        except OSError:
            rm_tree(datasets_dir)

    # Create __init__.py files
    write_init(data_dir / "__init__.py",
               '"""数据模块：原始数据与数据加载器。\n\n'
               '子模块:\n'
               '    raw/     -- CSV / XLSX 等原始数据文件\n'
               '    loaders/ -- 数据集加载器（CsvProsumerDataset 等）\n'
               '"""\n')

    # Update loaders/__init__.py content (rewrite from old datasets/__init__.py)
    loaders_init = loaders_dir / "__init__.py"
    if loaders_init.exists():
        old = loaders_init.read_text(encoding="utf-8")
        # Update internal references: datasets. -> data.loaders.
        new = old.replace("from datasets.", "from data.loaders.")
        new = new.replace("import data.loaders.", "import data.loaders.")
        if "数据加载器" not in new and '"""' in new:
            new = new.replace('"""', '"""数据加载器注册与公共 API。\n\n', 1)
        loaders_init.write_text(new, encoding="utf-8")
        print("  [UPDATE] data/loaders/__init__.py internal imports")

    print("Phase 1 DONE\n")


def phase_2():
    """Phase 2: Merge grid/ -> envs/grid/, common/rewards -> envs/rewards/."""
    print("\n" + "="*60)
    print("PHASE 2: Merge grid/ -> envs/grid/, common/rewards -> envs/rewards/")
    print("="*60)

    envs_dir = ROOT / "envs"
    grid_src = ROOT / "grid"
    rewards_src = ROOT / "common" / "rewards"

    # Move grid/ -> envs/grid/
    if grid_src.exists():
        # Remove __pycache__ first
        for pc in grid_src.rglob("__pycache__"):
            rm_tree(pc)
        move_dir(grid_src, envs_dir / "grid")

    # Move common/rewards -> envs/rewards/
    if rewards_src.exists():
        for pc in rewards_src.rglob("__pycache__"):
            rm_tree(pc)
        move_dir(rewards_src, envs_dir / "rewards")

    # Move common/vec_env.py, common/subproc_vec_env.py -> envs/
    for fname in ("vec_env.py", "subproc_vec_env.py"):
        src = ROOT / "common" / fname
        if src.exists():
            move_file(src, envs_dir / fname)

    # Clean up grid/ if still exists
    if grid_src.exists():
        rm_tree(grid_src)

    print("Phase 2 DONE\n")


def phase_3():
    """Phase 3: Merge algorithms/ -> controllers/madrl/."""
    print("\n" + "="*60)
    print("PHASE 3: Merge algorithms/ -> controllers/madrl/")
    print("="*60)

    controllers_dir = ROOT / "controllers"
    madrl_dir = controllers_dir / "madrl"
    mpc_dir = controllers_dir / "mpc"
    drl_dir = controllers_dir / "drl"
    algorithms_dir = ROOT / "algorithms"

    ensure_dir(madrl_dir)
    ensure_dir(mpc_dir)
    ensure_dir(drl_dir)

    # Move algorithms/ files to controllers/madrl/
    if algorithms_dir.exists():
        for pc in algorithms_dir.rglob("__pycache__"):
            rm_tree(pc)
        for f in algorithms_dir.iterdir():
            if f.is_file() and f.suffix == ".py":
                move_file(f, madrl_dir / f.name)
        try:
            algorithms_dir.rmdir()
        except OSError:
            rm_tree(algorithms_dir)

    # Move mpc_controller.py -> controllers/mpc/
    mpc_src = controllers_dir / "mpc_controller.py"
    if mpc_src.exists():
        move_file(mpc_src, mpc_dir / "mpc_controller.py")

    # Move classic_drl_controller.py -> controllers/drl/
    drl_src = controllers_dir / "classic_drl_controller.py"
    if drl_src.exists():
        move_file(drl_src, drl_dir / "classic_drl_controller.py")

    # Create __init__.py for new subdirs
    write_init(mpc_dir / "__init__.py",
               '"""MPC 控制器子模块。\n\n'
               '包含基于模型预测控制的控制器实现。\n'
               '"""\n')
    write_init(drl_dir / "__init__.py",
               '"""单智能体 DRL 控制器子模块。\n\n'
               '包含经典深度强化学习控制器实现。\n'
               '"""\n')

    print("Phase 3 DONE\n")


def phase_4():
    """Phase 4: Rename forecast/ -> predictors/."""
    print("\n" + "="*60)
    print("PHASE 4: Rename forecast/ -> predictors/")
    print("="*60)

    forecast_dir = ROOT / "forecast"
    predictors_dir = ROOT / "predictors"

    if forecast_dir.exists():
        # Remove __pycache__ first
        for pc in forecast_dir.rglob("__pycache__"):
            rm_tree(pc)
        shutil.move(str(forecast_dir), str(predictors_dir))
        print(f"  [RENAME] {forecast_dir} -> {predictors_dir}")

    print("Phase 4 DONE\n")


def phase_5():
    """Phase 5: Consolidate core/ + runners/ + evaluation/ + common/ -> scripts/."""
    print("\n" + "="*60)
    print("PHASE 5: Consolidate core/runners/evaluation/common -> scripts/")
    print("="*60)

    scripts_dir = ROOT / "scripts"
    utils_dir = scripts_dir / "utils"
    plots_dir = scripts_dir / "plots"
    recorders_dir = scripts_dir / "recorders"

    ensure_dir(utils_dir)
    ensure_dir(plots_dir)
    ensure_dir(recorders_dir)

    # core/builder.py -> scripts/builder.py
    move_file(ROOT / "core" / "builder.py", scripts_dir / "builder.py")

    # runners/train_runner.py -> scripts/train.py
    move_file(ROOT / "runners" / "train_runner.py", scripts_dir / "train.py")
    move_file(ROOT / "runners" / "checkpoints.py", scripts_dir / "checkpoints.py")

    # evaluation/ files
    eval_dir = ROOT / "evaluation"
    move_file(eval_dir / "evaluator.py", scripts_dir / "evaluate.py")
    move_file(eval_dir / "comparison.py", scripts_dir / "comparison.py")
    move_file(eval_dir / "plots.py", plots_dir / "plots.py")
    move_file(eval_dir / "grid_plots.py", plots_dir / "grid_plots.py")
    move_file(eval_dir / "reward_plots.py", plots_dir / "reward_plots.py")
    move_file(eval_dir / "episode_recorder.py", recorders_dir / "episode_recorder.py")
    move_file(eval_dir / "grid_recorder.py", recorders_dir / "grid_recorder.py")

    # common/ remaining files -> scripts/utils/
    common_dir = ROOT / "common"
    for fname in ("project_paths.py", "nested.py", "torch_runtime.py",
                  "replay_buffer.py", "experiment_notebook_utils.py"):
        src = common_dir / fname
        if src.exists():
            move_file(src, utils_dir / fname)

    # Create __init__.py files for new subdirs
    write_init(plots_dir / "__init__.py",
               '"""训练与评估可视化绘图工具。\n\n'
               '子模块:\n'
               '    plots        -- 通用训练曲线绘制\n'
               '    grid_plots   -- 电网潮流结果可视化\n'
               '    reward_plots -- 奖励分量分析图\n'
               '"""\n')
    write_init(recorders_dir / "__init__.py",
               '"""Episode 与电网数据录制器。\n\n'
               '子模块:\n'
               '    episode_recorder -- 通用 episode 数据记录\n'
               '    grid_recorder    -- 电网潮流详细数据记录\n'
               '"""\n')
    write_init(utils_dir / "__init__.py",
               '"""通用工具与基础设施。\n\n'
               '子模块:\n'
               '    project_paths              -- 项目根路径与数据目录定位\n'
               '    nested                     -- 嵌套 tensor 操作工具\n'
               '    torch_runtime              -- PyTorch 运行时配置\n'
               '    replay_buffer              -- 经验回放缓冲区\n'
               '    experiment_notebook_utils  -- notebook 辅助工具\n'
               '"""\n')

    # Cleanup: remove emptied directories
    for d in (ROOT / "core", ROOT / "runners", eval_dir, common_dir):
        # Remove __pycache__ first
        if d.exists():
            for pc in d.rglob("__pycache__"):
                rm_tree(pc)
            # Remove __init__.py
            init = d / "__init__.py"
            rm_file(init)
            try:
                d.rmdir()
            except OSError:
                # Still has files?
                remaining = list(d.iterdir())
                if remaining:
                    print(f"  [WARN] {d} still has: {[f.name for f in remaining]}")
                else:
                    d.rmdir()

    print("Phase 5 DONE\n")


def phase_6():
    """Phase 6: Global import path updates."""
    print("\n" + "="*60)
    print("PHASE 6: Global import path updates")
    print("="*60)

    py_files = collect_py_files(ROOT)
    updated = 0
    for f in py_files:
        if apply_import_replacements(f):
            updated += 1
    print(f"  Updated imports in {updated} files")

    # Also update notebooks (.ipynb files)
    for nb_file in ROOT.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in str(nb_file):
            continue
        try:
            text = nb_file.read_text(encoding="utf-8")
            original = text
            for old, new in IMPORT_REPLACEMENTS:
                text = text.replace(old, new)
            if text != original:
                nb_file.write_text(text, encoding="utf-8")
                print(f"  [IMPORT] Updated notebook: {nb_file}")
        except (UnicodeDecodeError, PermissionError):
            pass

    print("Phase 6 DONE\n")


def phase_7():
    """Phase 7: Cleanup."""
    print("\n" + "="*60)
    print("PHASE 7: Cleanup")
    print("="*60)

    # Remove all __pycache__ directories
    for pc in ROOT.rglob("__pycache__"):
        rm_tree(pc)

    # Remove empty directories left behind
    for d in sorted(ROOT.rglob("*"), reverse=True):
        if d.is_dir() and not list(d.iterdir()):
            try:
                d.rmdir()
                print(f"  [RMDIR] {d}")
            except OSError:
                pass

    print("Phase 7 DONE\n")


if __name__ == "__main__":
    print("MADRL_ESS Project Restructuring")
    print(f"Root: {ROOT}")
    print()

    phase_0()
    phase_1()
    phase_2()
    phase_3()
    phase_4()
    phase_5()
    phase_6()
    phase_7()

    print("\n" + "="*60)
    print("ALL PHASES COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review moved files and import changes")
    print("  2. Run: python -m pytest tests/ -v --tb=short")
    print("  3. Run: npx gitnexus analyze --embeddings")
