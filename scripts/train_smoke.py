"""Compatibility wrapper around scripts.run_debug_training."""

from scripts.run_debug_training import main

__all__ = ["main"]


if __name__ == "__main__":
    completed = main()
    print(f"Completed episodes: {completed}")
