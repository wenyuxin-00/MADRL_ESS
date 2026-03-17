"""pytest 公共配置。

当前运行环境的系统临时目录存在权限问题，因此这里显式提供一个仓库内可控、可清理的
`tmp_path` fixture，保证测试在不同机器上都不依赖本机目录结构。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "test_tmp_cases"


@pytest.fixture
def tmp_path():
    """提供一个仓库内的临时目录，并在测试后清理。"""
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    case_dir = TEST_TMP_ROOT / f"case_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    """测试结束后清理统一临时目录。"""
    shutil.rmtree(TEST_TMP_ROOT, ignore_errors=True)
