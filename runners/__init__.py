"""Training runners and checkpoint management.
训练循环 Runner 与 Checkpoint 管理。

Key components / 主要组件:
    - TrainRunner               -- main training loop orchestrator
    - checkpoints module        -- save/load/manifest utilities

Typical usage::

    from core import build_train_runner
    runner = build_train_runner(cfg)
    runner.run()
    runner.save_model(model_dir, episode_tag)
"""

from runners.checkpoints import (
    build_checkpoint_manifest,
    infer_latest_checkpoint_tag,
    load_latest_checkpoint_manifest,
    resolve_checkpoint_to_load,
    write_checkpoint_manifest,
)
from runners.train_runner import TrainRunner

__all__ = [
    "TrainRunner",
    "build_checkpoint_manifest",
    "infer_latest_checkpoint_tag",
    "load_latest_checkpoint_manifest",
    "resolve_checkpoint_to_load",
    "write_checkpoint_manifest",
]
