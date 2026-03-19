"""Checkpoint 保存与加载。

管理训练过程中模型权重、优化器状态、训练进度的持久化。

主要函数:
    save_checkpoint -- 保存训练 checkpoint
    load_checkpoint -- 加载训练 checkpoint
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# 最新 checkpoint 清单文件名
LATEST_CHECKPOINT_MANIFEST = "latest_checkpoint.json"
# 按 episode 标签命名的 checkpoint 清单文件名模板
TAGGED_CHECKPOINT_MANIFEST = "checkpoint_ep_{episode_tag}.json"
# 用于从文件名中提取 episode 标签的正则表达式
ACTOR_TAG_PATTERN = re.compile(r"actor_agent_\d+_ep_(\d+)\.pth$")
CRITIC_TAG_PATTERN = re.compile(r"critic_agent_\d+_ep_(\d+)\.pth$")


def get_algorithm_checkpoint_dir(model_dir, algorithm: str) -> Path:
    """返回指定算法的 checkpoint 目录路径。

    参数:
        model_dir: 模型根目录。
        algorithm: 算法名称（如 "MADDPG"）。

    返回:
        算法专属的 checkpoint 目录 Path。
    """
    return Path(model_dir) / algorithm


def checkpoint_tag_exists(algo_dir, episode_tag: int) -> bool:
    """检查指定 episode 标签的完整 checkpoint 是否存在。

    完整 checkpoint 要求同时存在 actor 和 critic 的权重文件。

    参数:
        algo_dir: 算法 checkpoint 目录。
        episode_tag: episode 标签编号。

    返回:
        如果 actor 和 critic 文件都存在则返回 True。
    """
    algo_dir = Path(algo_dir)
    actor_files = list(algo_dir.glob(f"actor_agent_*_ep_{episode_tag}.pth"))
    critic_files = list(algo_dir.glob(f"critic_agent_*_ep_{episode_tag}.pth"))
    return bool(actor_files) and bool(critic_files)


def infer_latest_checkpoint_tag(model_dir, algorithm: str) -> int:
    """通过扫描权重文件名推断最新的 checkpoint 标签。

    遍历目录下所有 actor 和 critic 文件，提取 episode 标签，
    取两者的交集后返回最大值。

    参数:
        model_dir: 模型根目录。
        algorithm: 算法名称。

    返回:
        最新的完整 checkpoint 的 episode 标签。

    异常:
        FileNotFoundError: 目录不存在或找不到完整的 checkpoint。
    """
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    if not algo_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: '{algo_dir}'")

    # 从 actor 文件名中提取 episode 标签集合
    actor_tags = {
        int(match.group(1))
        for path in algo_dir.glob("actor_agent_*_ep_*.pth")
        if (match := ACTOR_TAG_PATTERN.match(path.name)) is not None
    }
    # 从 critic 文件名中提取 episode 标签集合
    critic_tags = {
        int(match.group(1))
        for path in algo_dir.glob("critic_agent_*_ep_*.pth")
        if (match := CRITIC_TAG_PATTERN.match(path.name)) is not None
    }
    # 取交集确保 actor 和 critic 都存在，然后取最大标签
    common_tags = sorted(actor_tags & critic_tags)
    if not common_tags:
        raise FileNotFoundError(
            f"No complete checkpoint tags were found under '{algo_dir}'."
        )
    return common_tags[-1]


def build_checkpoint_manifest(
    *,
    algorithm: str,
    saved_episode_tag: int,
    episodes_completed: int,
    total_steps: int,
    num_envs: int,
    episode_limit: int,
    save_dir,
) -> dict:
    """构建与 checkpoint 文件一同保存的轻量级清单字典。

    参数:
        algorithm: 算法名称。
        saved_episode_tag: 保存时的 episode 标签。
        episodes_completed: 已完成的总 episode 数。
        total_steps: 已执行的总步数。
        num_envs: 并行环境数量。
        episode_limit: 每个 episode 的最大步数。
        save_dir: 保存目录。

    返回:
        包含训练进度元数据的字典。
    """
    return {
        "algorithm": algorithm,
        "saved_episode_tag": int(saved_episode_tag),
        "episodes_completed": int(episodes_completed),
        "total_steps": int(total_steps),
        "num_envs": int(num_envs),
        "episode_limit": int(episode_limit),
        "save_dir": str(Path(save_dir).resolve()),
    }


def write_checkpoint_manifest(algo_dir, manifest: dict) -> dict:
    """同时写入最新清单和按标签命名的清单文件。

    参数:
        algo_dir: 算法 checkpoint 目录。
        manifest: 由 build_checkpoint_manifest 构建的清单字典。

    返回:
        实际写入的清单内容（字典副本）。
    """
    algo_dir = Path(algo_dir)
    algo_dir.mkdir(parents=True, exist_ok=True)

    # 写入 "latest" 清单（总是覆盖）
    latest_path = algo_dir / LATEST_CHECKPOINT_MANIFEST
    # 写入按 episode 标签命名的清单（用于历史追溯）
    tagged_path = algo_dir / TAGGED_CHECKPOINT_MANIFEST.format(
        episode_tag=manifest["saved_episode_tag"]
    )
    payload = dict(manifest)

    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tagged_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_latest_checkpoint_manifest(model_dir, algorithm: str) -> dict:
    """加载指定算法的最新 checkpoint 清单。

    参数:
        model_dir: 模型根目录。
        algorithm: 算法名称。

    返回:
        解析后的清单字典。

    异常:
        FileNotFoundError: 清单文件不存在。
    """
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    manifest_path = algo_dir / LATEST_CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Latest checkpoint manifest not found: '{manifest_path}'"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_checkpoint_to_load(model_dir, algorithm: str, episode_tag: int | None = None) -> dict:
    """解析应当加载的 checkpoint 标签。

    解析优先级:
      1. 若显式指定了 episode_tag，直接校验该标签。
      2. 否则尝试读取最新清单文件。
      3. 回退到扫描权重文件名推断最新完整 checkpoint。

    参数:
        model_dir: 模型根目录。
        algorithm: 算法名称。
        episode_tag: 指定的 episode 标签，为 None 时自动推断。

    返回:
        包含 algorithm、algo_dir、saved_episode_tag 等键的解析结果字典。

    异常:
        FileNotFoundError: 无法找到有效的 checkpoint。
    """
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    manifest = None

    # 未指定标签时，优先尝试从最新清单获取
    if episode_tag is None:
        try:
            manifest = load_latest_checkpoint_manifest(model_dir, algorithm)
            candidate_tag = int(manifest["saved_episode_tag"])
            if checkpoint_tag_exists(algo_dir, candidate_tag):
                episode_tag = candidate_tag
        except FileNotFoundError:
            manifest = None

    # 清单也未能提供时，回退到文件名扫描
    if episode_tag is None:
        episode_tag = infer_latest_checkpoint_tag(model_dir, algorithm)

    # 最终校验选定标签的文件是否真实存在
    if not checkpoint_tag_exists(algo_dir, int(episode_tag)):
        raise FileNotFoundError(
            f"Checkpoint tag {episode_tag} was not found under '{algo_dir}'."
        )

    resolved = {
        "algorithm": algorithm,
        "algo_dir": str(algo_dir),
        "saved_episode_tag": int(episode_tag),
    }
    # 如果有清单元数据，附加训练进度信息
    if manifest is not None:
        resolved.update(
            {
                "episodes_completed": int(manifest.get("episodes_completed", episode_tag)),
                "total_steps": int(manifest.get("total_steps", 0)),
                "num_envs": int(manifest.get("num_envs", 0)),
                "episode_limit": int(manifest.get("episode_limit", 0)),
                "save_dir": manifest.get("save_dir", str(algo_dir)),
            }
        )
    return resolved
