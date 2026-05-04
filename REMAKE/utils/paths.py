from __future__ import annotations

from pathlib import Path

REMAKE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REMAKE_ROOT.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

