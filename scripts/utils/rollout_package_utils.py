from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY, PRICE_PROTOCOL_VERSION
ROLLOUT_PACKAGE_VERSION = 2
_CFG_FLOAT_RTOL = _CFG_FLOAT_ATOL = 1e-06

def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value

def _raise_mismatch(prefix: str, field_name: str, expected: Any, actual: Any) -> None:
    raise ValueError(f"{prefix} mismatch at '{field_name}': expected={expected!r}, actual={actual!r}.")

def _assert_strict_match(prefix: str, field_name: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        _raise_mismatch(prefix, field_name, expected, actual)

def _assert_float_match(prefix: str, field_name: str, expected: float, actual: float) -> None:
    if not math.isclose(float(expected), float(actual), rel_tol=_CFG_FLOAT_RTOL, abs_tol=_CFG_FLOAT_ATOL):
        _raise_mismatch(prefix, field_name, expected, actual)

def _assert_float_sequence_match(prefix: str, field_name: str, expected: list[float], actual: list[float]) -> None:
    expected_array, actual_array = (np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64))
    if expected_array.shape != actual_array.shape or not np.allclose(expected_array, actual_array, rtol=_CFG_FLOAT_RTOL, atol=_CFG_FLOAT_ATOL):
        _raise_mismatch(prefix, field_name, expected, actual)

def _assert_int_sequence_match(prefix: str, field_name: str, expected: list[int], actual: list[int]) -> None:
    if [int(value) for value in expected] != [int(value) for value in actual]:
        _raise_mismatch(prefix, field_name, expected, actual)

def _assert_str_sequence_match(prefix: str, field_name: str, expected: list[str], actual: list[str]) -> None:
    if [str(value) for value in expected] != [str(value) for value in actual]:
        _raise_mismatch(prefix, field_name, expected, actual)

def _normalize_battery_controls_from_cfg(cfg: Any) -> dict[str, Any]:
    raw = getattr(getattr(cfg, 'env', None), 'battery_capacity', [])
    if isinstance(raw, np.ndarray):
        capacity = raw.reshape(-1).tolist()
    elif isinstance(raw, (list, tuple)):
        capacity = list(raw)
    else:
        capacity = [] if raw in (None, '') else [raw]
    env = getattr(cfg, 'env', None)
    return {'battery_capacity': [float(value) for value in capacity], 'max_charge_rate': float(getattr(env, 'max_charge_rate', np.nan)), 'efficiency': float(getattr(env, 'efficiency', np.nan)), 'init_soc': float(getattr(env, 'init_soc', np.nan)), 'soc_min': float(getattr(env, 'soc_min', np.nan)), 'soc_max': float(getattr(env, 'soc_max', np.nan)), 'soc_target': float(getattr(env, 'soc_target', np.nan))}

def build_rollout_cfg_snapshot_from_cfg(cfg: Any, *, prediction_mode: str | None=None) -> dict[str, Any]:
    forecast_backend = str(getattr(getattr(cfg, 'forecast', None), 'type', 'perfect'))
    resolved_mode = grid_nb.normalize_prediction_mode(prediction_mode) if prediction_mode is not None else grid_nb.resolve_prediction_mode_from_forecast_backend(forecast_backend)
    return {'test_start_date': str(getattr(getattr(cfg, 'data', None), 'test_start_date', '')), 'test_end_date': str(getattr(getattr(cfg, 'data', None), 'test_end_date', '')), 'prediction_mode': str(resolved_mode), 'forecast_backend': forecast_backend, 'agent_profiles': [str(value) for value in list(getattr(getattr(cfg, 'data', None), 'agent_profiles', []))], 'agent_bus_ids': [int(value) for value in list(getattr(getattr(cfg, 'grid', None), 'agent_bus_ids', []))], 'load_scale': [float(value) for value in list(getattr(getattr(cfg, 'data', None), 'load_scale', []))], 'pv_scale': [float(value) for value in list(getattr(getattr(cfg, 'data', None), 'pv_scale', []))], 'battery_controls': _normalize_battery_controls_from_cfg(cfg), 'future_horizon': int(getattr(getattr(cfg, 'env', None), 'future_horizon', 0)), 'episode_limit': int(getattr(getattr(cfg, 'env', None), 'episode_limit', 0)), 'v_min_pu': float(getattr(getattr(cfg, 'grid', None), 'v_min_pu', np.nan)), 'v_max_pu': float(getattr(getattr(cfg, 'grid', None), 'v_max_pu', np.nan)), 'price_protocol_version': int(PRICE_PROTOCOL_VERSION), IMPORT_PRICE_MARKUP_KEY: float(getattr(getattr(cfg, 'reward', None), IMPORT_PRICE_MARKUP_KEY, 0.0)), 'export_subsidy_eur_per_kwh': float(getattr(getattr(cfg, 'reward', None), 'export_subsidy_eur_per_kwh', np.nan))}

def assert_rollout_cfg_snapshot_matches(expected_snapshot: dict[str, Any], actual_snapshot: dict[str, Any], *, mismatch_prefix: str='Rollout package') -> None:
    for field_name in ('test_start_date', 'test_end_date', 'prediction_mode', 'forecast_backend', 'future_horizon', 'episode_limit', 'price_protocol_version'):
        _assert_strict_match(mismatch_prefix, field_name, expected_snapshot[field_name], actual_snapshot[field_name])
    _assert_str_sequence_match(mismatch_prefix, 'agent_profiles', expected_snapshot['agent_profiles'], actual_snapshot['agent_profiles'])
    _assert_int_sequence_match(mismatch_prefix, 'agent_bus_ids', expected_snapshot['agent_bus_ids'], actual_snapshot['agent_bus_ids'])
    for field_name in ('load_scale', 'pv_scale'):
        _assert_float_sequence_match(mismatch_prefix, field_name, expected_snapshot[field_name], actual_snapshot[field_name])
    _assert_float_sequence_match(mismatch_prefix, 'battery_controls.battery_capacity', expected_snapshot['battery_controls']['battery_capacity'], actual_snapshot['battery_controls']['battery_capacity'])
    for field_name in ('max_charge_rate', 'efficiency', 'init_soc', 'soc_min', 'soc_max', 'soc_target'):
        _assert_float_match(mismatch_prefix, f'battery_controls.{field_name}', expected_snapshot['battery_controls'][field_name], actual_snapshot['battery_controls'][field_name])
    for field_name in ('v_min_pu', 'v_max_pu', IMPORT_PRICE_MARKUP_KEY, 'export_subsidy_eur_per_kwh'):
        _assert_float_match(mismatch_prefix, field_name, expected_snapshot[field_name], actual_snapshot[field_name])

def assert_solver_fingerprint_matches(expected: dict[str, Any], actual: dict[str, Any], *, name: str) -> None:
    if set(expected) != set(actual):
        _raise_mismatch(name, 'keys', sorted(expected), sorted(actual))
    for field_name, expected_value in sorted(expected.items()):
        actual_value = actual[field_name]
        if isinstance(expected_value, (float, int, np.floating, np.integer)) and isinstance(actual_value, (float, int, np.floating, np.integer)):
            _assert_float_match(name, field_name, float(expected_value), float(actual_value))
        else:
            _assert_strict_match(name, field_name, expected_value, actual_value)

def table_required_files() -> tuple[str, ...]:
    return ('manifest.json', 'diagnostics.json', 'step_df.npz', 'agent_df.npz', 'grid_df.npz', 'summary_df.npz')

def save_rollout_package(package: dict[str, Any], target_dir: str | Path) -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    manifest = dict(package['manifest'])
    manifest['tables'] = {name: save_dataframe_npz(pd.DataFrame(package[name]).copy(), target_path / f'{name}.npz') for name in ('step_df', 'agent_df', 'grid_df', 'summary_df')}
    (target_path / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_default), encoding='utf-8')
    (target_path / 'diagnostics.json').write_text(json.dumps(dict(package.get('diagnostics', {})), indent=2, default=json_default), encoding='utf-8')
    return target_path

def load_rollout_package(target_dir: str | Path, *, package_label: str, version: int, regenerate_hint: str) -> dict[str, Any]:
    target_path = Path(target_dir).resolve()
    paths = {name: target_path / f'{name}.json' for name in ('manifest', 'diagnostics')}
    paths |= {name: target_path / f'{name}.npz' for name in ('step_df', 'agent_df', 'grid_df', 'summary_df')}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{package_label} is incomplete at {target_path}: missing {', '.join(missing)}.")
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    if int(manifest.get('rollout_package_version', -1)) != int(version):
        raise ValueError(f"Unsupported {package_label} version at {target_path}: expected={version}, actual={manifest.get('rollout_package_version')!r}. {regenerate_hint}")
    tables = dict(manifest.get('tables', {}))
    missing_tables = sorted({'step_df', 'agent_df', 'grid_df', 'summary_df'}.difference(tables))
    if missing_tables:
        raise ValueError(f"{package_label} manifest at {target_path} is missing table metadata for {', '.join(missing_tables)}.")
    return {'manifest': manifest, 'diagnostics': json.loads(paths['diagnostics'].read_text(encoding='utf-8')), **{name: load_dataframe_npz(paths[name], metadata=tables[name]) for name in ('step_df', 'agent_df', 'grid_df', 'summary_df')}}

def resolve_exact_rollout_package_dir(target_dir: str | Path, *, package_label: str, missing_dir_hint: str, validate_package_dir) -> Path:
    target_path = Path(target_dir).expanduser().resolve()
    if target_path.exists():
        if not target_path.is_dir():
            raise NotADirectoryError(f'Expected a {package_label.lower()} directory, got file: {target_path}')
        validate_package_dir(target_path)
        return target_path
    parent_dir, prefix = (target_path.parent, target_path.name)
    candidates = []
    if parent_dir.exists():
        for candidate_dir in parent_dir.iterdir():
            if candidate_dir.is_dir() and (candidate_dir.name == prefix or candidate_dir.name.startswith(f'{prefix}_')):
                candidates.append(candidate_dir.name)
    discovered_text = ', '.join(sorted(candidates)) if candidates else 'none'
    raise FileNotFoundError(f"{package_label} resolution now requires one exact package directory. Requested path does not exist: {target_path}. Matching candidates under '{parent_dir}': {discovered_text}. {missing_dir_hint}")

def _serialize_dataframe_for_npz(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    schema: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for column_index, column_name in enumerate(frame.columns):
        series, key = (frame[column_name], f'col_{column_index}')
        if pd.api.types.is_datetime64_any_dtype(series):
            storage, values = ('datetime', pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S.%f').to_numpy(dtype=np.str_))
        elif pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            storage, values = ('native', np.asarray(series.to_numpy(copy=True)))
        else:
            storage, values = ('string', series.fillna('').astype(str).to_numpy(dtype=np.str_))
        arrays[key] = values
        schema.append({'name': str(column_name), 'dtype': str(series.dtype), 'storage': storage, 'key': key})
    return ({'columns': schema, 'row_count': int(len(frame))}, arrays)

def save_dataframe_npz(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    metadata, arrays = _serialize_dataframe_for_npz(frame)
    np.savez_compressed(path, **arrays)
    return metadata

def load_dataframe_npz(path: Path, *, metadata: dict[str, Any]) -> pd.DataFrame:
    columns = list(metadata.get('columns', []))
    if not columns:
        return pd.DataFrame()
    data: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for column_meta in columns:
            name = str(column_meta['name'])
            raw = np.asarray(archive[str(column_meta['key'])])
            storage = str(column_meta.get('storage', 'native'))
            data[name] = pd.to_datetime(raw.astype(str), errors='coerce') if storage == 'datetime' else raw.astype(str if storage == 'string' else np.dtype(str(column_meta['dtype'])), copy=False)
    return pd.DataFrame(data, columns=[str(column_meta['name']) for column_meta in columns])

def relabel_rollout_dataframe(frame: pd.DataFrame, controller_label: str) -> pd.DataFrame:
    relabeled = frame.copy()
    if 'controller' in relabeled.columns:
        relabeled['controller'] = str(controller_label)
    return relabeled
