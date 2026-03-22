import json
from pathlib import Path


def test_python_files_are_utf8_without_bom():
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in repo_root.rglob("*.py"):
        if path.read_bytes().startswith(b"\xef\xbb\xbf"):
            offenders.append(path)

    assert offenders == []


def test_key_notebooks_are_utf8_without_bom():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "data" / "prepare_simbench_data.ipynb",
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_madrl.ipynb",
        repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb",
    ]

    offenders = [path for path in notebook_paths if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert offenders == []


def test_key_notebooks_do_not_contain_placeholder_text():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "data" / "prepare_simbench_data.ipynb",
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_madrl.ipynb",
        repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb",
    ]

    for path in notebook_paths:
        text = path.read_text(encoding="utf-8")
        assert "???" not in text
        assert "?? notebook" not in text


def test_notebook_defaults_stay_portable():
    repo_root = Path(__file__).resolve().parents[1]
    forecast_nb = json.loads(
        (repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb").read_text(encoding="utf-8")
    )
    madrl_nb = json.loads(
        (repo_root / "notebooks" / "madrl" / "train_madrl.ipynb").read_text(encoding="utf-8")
    )

    forecast_root_cell = "".join(forecast_nb["cells"][1]["source"])
    forecast_runtime_cell = "".join(forecast_nb["cells"][4]["source"])
    madrl_runtime_cell = "".join(madrl_nb["cells"][3]["source"])

    assert 'while project_root != project_root.parent and not (project_root / "configs").exists()' in forecast_root_cell
    assert 'device_request = None' in forecast_runtime_cell
    assert 'require_cuda = False' in forecast_runtime_cell

    assert 'profile = "debug"' in madrl_runtime_cell
    assert 'vec_env_type = "dummy"' in madrl_runtime_cell
    assert 'device_request = None' in madrl_runtime_cell
    assert 'require_cuda = False' in madrl_runtime_cell
