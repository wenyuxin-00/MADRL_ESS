"""Shared pytest fixtures for repository-local temporary files."""

from __future__ import annotations

import shutil
import uuid
import multiprocessing as mp
from pathlib import Path

import pytest


TEST_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@pytest.fixture
def tmp_path():
    """Provide a temporary directory that stays inside ``tests/``."""
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    case_dir = TEST_TMP_ROOT / f"case_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    """Remove the shared temporary root after the test session finishes."""
    _ = session, exitstatus
    if mp.current_process().name != "MainProcess":
        return
    shutil.rmtree(TEST_TMP_ROOT, ignore_errors=True)
