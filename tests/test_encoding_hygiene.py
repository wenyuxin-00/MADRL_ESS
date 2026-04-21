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
        repo_root / "notebooks" / "madrl" / "train_base.ipynb",
        repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb",
    ]

    offenders = [path for path in notebook_paths if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert offenders == []


def test_key_notebooks_do_not_contain_placeholder_text():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_base.ipynb",
        repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb",
    ]

    for path in notebook_paths:
        text = path.read_text(encoding="utf-8")
        assert "???" not in text
        assert "?? notebook" not in text


def test_notebook_defaults_stay_portable():
    repo_root = Path(__file__).resolve().parents[1]
    forecast_text = _load_notebook_text(repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb")
    madrl_text = _load_notebook_text(repo_root / "notebooks" / "madrl" / "train_base.ipynb")
    grid_text = _load_notebook_text(repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb")

    assert 'while project_root != project_root.parent and not (project_root / "configs").exists()' in forecast_text
    assert 'device_request = None' in forecast_text
    assert 'require_cuda = False' in forecast_text
    assert 'cfg.data.agent_profiles = ["SFH12", "SFH14", "SFH16", "SFH18", "SFH20"]' in forecast_text
    assert 'cfg.env.num_agents = len(cfg.data.agent_profiles)' in forecast_text
    assert 'signals_to_train = [signal_name for signal_name in signal_order if signal_name not in results]' in forecast_text
    assert 'retraining missing or incompatible signals' in forecast_text
    assert 'full_test_view_signals = ["wholesale_price"]' in forecast_text
    assert 'legacy_aliases = {"price": "wholesale_price"}' in forecast_text
    assert 'set `full_test_view_signals = ["wholesale_price"]`' in forecast_text
    assert 'confirm only `wholesale_price` is plotted for that date window' in forecast_text

    assert 'experiment_controls = {' in madrl_text
    assert 'data_controls = {' in madrl_text
    assert 'train_controls = {' in madrl_text
    assert 'prediction_mode = "normal"' in madrl_text
    assert '"launch_mode": "external"' in madrl_text
    assert '"vec_env_type": "subproc" if int(num_envs) > 1 else "dummy"' in madrl_text
    assert 'recommended_gpu_fast_num_envs' in madrl_text
    assert '"agent_profiles": notebook_controls["agent_profiles"]' in madrl_text
    assert '"agent_bus_ids": notebook_controls["agent_bus_ids"]' in madrl_text
    assert 'battery_controls = deepcopy(notebook_controls["battery"])' in madrl_text
    assert 'battery_capacity_kwh = 20.0' in madrl_text
    assert 'notebook_battery_controls = {"battery_capacity": battery_capacity_kwh, "max_charge_rate": battery_max_charge_rate}' in madrl_text

    assert 'ProsumerDataset(' in grid_text
    assert 'build_simbench_net(cfg.grid.sb_code)' in grid_text
    assert 'simbench_2016_full.csv' not in grid_text
    assert 'observation_profile=' not in grid_text


def test_forecast_lstm_section3_signal_defaults_match_price_protocol():
    repo_root = Path(__file__).resolve().parents[1]
    forecast_text = _load_notebook_text(repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb")

    assert 'full_test_view_signals = ["wholesale_price"]' in forecast_text
    assert 'legacy_aliases = {"price": "wholesale_price"}' in forecast_text
    assert 'set `full_test_view_signals = ["wholesale_price"]`' in forecast_text
    assert 'confirm only `wholesale_price` is plotted for that date window' in forecast_text

