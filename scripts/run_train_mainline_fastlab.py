"""Compatibility alias for the fast-lab-backed mainline training entrypoint."""

from __future__ import annotations

from scripts.run_train_mainline import main


if __name__ == "__main__":
    raise SystemExit(main())
