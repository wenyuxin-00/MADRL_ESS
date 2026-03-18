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


LATEST_CHECKPOINT_MANIFEST = "latest_checkpoint.json"
TAGGED_CHECKPOINT_MANIFEST = "checkpoint_ep_{episode_tag}.json"
ACTOR_TAG_PATTERN = re.compile(r"actor_agent_\d+_ep_(\d+)\.pth$")
CRITIC_TAG_PATTERN = re.compile(r"critic_agent_\d+_ep_(\d+)\.pth$")


def get_algorithm_checkpoint_dir(model_dir, algorithm: str) -> Path:
    """Return the algorithm-specific checkpoint directory."""
    return Path(model_dir) / algorithm


def checkpoint_tag_exists(algo_dir, episode_tag: int) -> bool:
    """Check whether a full actor+critic checkpoint exists for the given tag."""
    algo_dir = Path(algo_dir)
    actor_files = list(algo_dir.glob(f"actor_agent_*_ep_{episode_tag}.pth"))
    critic_files = list(algo_dir.glob(f"critic_agent_*_ep_{episode_tag}.pth"))
    return bool(actor_files) and bool(critic_files)


def infer_latest_checkpoint_tag(model_dir, algorithm: str) -> int:
    """Infer the latest checkpoint tag by scanning saved weight filenames."""
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    if not algo_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: '{algo_dir}'")

    actor_tags = {
        int(match.group(1))
        for path in algo_dir.glob("actor_agent_*_ep_*.pth")
        if (match := ACTOR_TAG_PATTERN.match(path.name)) is not None
    }
    critic_tags = {
        int(match.group(1))
        for path in algo_dir.glob("critic_agent_*_ep_*.pth")
        if (match := CRITIC_TAG_PATTERN.match(path.name)) is not None
    }
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
    """Build the lightweight manifest stored next to saved checkpoints."""
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
    """Write both a latest manifest and a tag-specific manifest."""
    algo_dir = Path(algo_dir)
    algo_dir.mkdir(parents=True, exist_ok=True)

    latest_path = algo_dir / LATEST_CHECKPOINT_MANIFEST
    tagged_path = algo_dir / TAGGED_CHECKPOINT_MANIFEST.format(
        episode_tag=manifest["saved_episode_tag"]
    )
    payload = dict(manifest)

    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tagged_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_latest_checkpoint_manifest(model_dir, algorithm: str) -> dict:
    """Load the latest checkpoint manifest for the given algorithm."""
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    manifest_path = algo_dir / LATEST_CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Latest checkpoint manifest not found: '{manifest_path}'"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_checkpoint_to_load(model_dir, algorithm: str, episode_tag: int | None = None) -> dict:
    """Resolve which checkpoint tag should be loaded.

    Resolution order:
      1. If episode_tag is provided, validate it directly.
      2. Otherwise try the latest manifest.
      3. Fall back to scanning weight filenames for the latest full checkpoint.
    """
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    manifest = None

    if episode_tag is None:
        try:
            manifest = load_latest_checkpoint_manifest(model_dir, algorithm)
            candidate_tag = int(manifest["saved_episode_tag"])
            if checkpoint_tag_exists(algo_dir, candidate_tag):
                episode_tag = candidate_tag
        except FileNotFoundError:
            manifest = None

    if episode_tag is None:
        episode_tag = infer_latest_checkpoint_tag(model_dir, algorithm)

    if not checkpoint_tag_exists(algo_dir, int(episode_tag)):
        raise FileNotFoundError(
            f"Checkpoint tag {episode_tag} was not found under '{algo_dir}'."
        )

    resolved = {
        "algorithm": algorithm,
        "algo_dir": str(algo_dir),
        "saved_episode_tag": int(episode_tag),
    }
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
