from __future__ import annotations

import json
from pathlib import Path


RUN_ID = "20260504_234339_01e73bd1"


def _notebook_text(name: str) -> str:
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )


def test_notebooks_use_promoted_imports_and_root_marker() -> None:
    for path in sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")):
        text = _notebook_text(path.name)
        assert ("from " + "REMAKE") not in text
        assert ("import " + "REMAKE") not in text
        assert 'path / "REMAKE" / "configs" / "cfg.py"' not in text
        assert ('path / "configs" / "cfg.py"' in text) or ("path / 'configs' / 'cfg.py'" in text)


def test_cached_notebooks_pin_complete_run() -> None:
    for name in ("madrl.ipynb", "madrl_base.ipynb", "madrl_base_safe.ipynb", "madrl_projection_safe.ipynb", "misocp.ipynb", "mpc.ipynb", "compare.ipynb"):
        assert RUN_ID in _notebook_text(name)


def test_madrl_base_safe_defaults_to_cached_penalty_run() -> None:
    text = _notebook_text("madrl_base_safe.ipynb")
    required = [
        f'run_dir = Path("artifacts/runs/{RUN_ID}")',
        'scheme_name = "madrl_base_safe"',
        "retrain = False",
        "train_episodes = 500",
        'display(Markdown("# MADRL + Safety Penalty + LSTM Forecast"))',
        '"w_voltage_pen": float(w_voltage)',
        '"w_line_pen": float(w_line)',
        '"w_trafo_pen": float(w_trafo)',
    ]
    for token in required:
        assert token in text
