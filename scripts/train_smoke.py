"""Compatibility smoke-training entry point.

`run_debug_training.py` is the canonical implementation. This wrapper keeps the
historical script name available without maintaining a second copy of the same
logic.
"""

from __future__ import annotations

try:
    from scripts.run_debug_training import main
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_debug_training import main


if __name__ == "__main__":
    completed = main()
    print(f"Completed episodes: {completed}")
