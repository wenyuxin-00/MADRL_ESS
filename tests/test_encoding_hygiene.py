import json
from pathlib import Path


def _load_notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )


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
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb",
        repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb",
    ]

    offenders = [path for path in notebook_paths if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert offenders == []


def test_key_notebooks_do_not_contain_placeholder_text():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb",
        repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb",
    ]

    for path in notebook_paths:
        text = path.read_text(encoding="utf-8")
        assert "???" not in text
        assert "?? notebook" not in text


def test_notebook_defaults_stay_portable():
    repo_root = Path(__file__).resolve().parents[1]
    forecast_text = _load_notebook_text(repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb")
    madrl_text = _load_notebook_text(repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb")
    grid_text = _load_notebook_text(repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb")

    assert 'while project_root != project_root.parent and not (project_root / "configs").exists()' in forecast_text
    assert 'device_request = None' in forecast_text
    assert 'require_cuda = False' in forecast_text
    assert 'cfg.data.agent_profiles = ["SFH12", "SFH14", "SFH16"]' in forecast_text

    assert 'experiment_controls = {' in madrl_text
    assert 'data_controls = {' in madrl_text
    assert 'train_controls = {' in madrl_text
    assert '"vec_env_type": "subproc"' in madrl_text
    assert '"prediction_mode": "normal"' in madrl_text
    assert '"launch_mode": "external"' in madrl_text
    assert 'recommended_gpu_fast_num_envs()' in madrl_text
    assert 'bool(torch.cuda.is_available())' in madrl_text

    assert 'ProsumerDataset(' in grid_text
    assert 'build_simbench_net(cfg.grid.sb_code)' in grid_text
    assert 'simbench_2016_full.csv' not in grid_text
    assert 'observation_profile=' not in grid_text


def test_removed_legacy_notebooks_are_gone():
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "notebooks" / "data" / "data_process.ipynb").exists()
    assert not (repo_root / "notebooks" / "data" / "prepare_simbench_data.ipynb").exists()
    assert not (repo_root / "notebooks" / "madrl" / "train_madrl.ipynb").exists()
