from __future__ import annotations
from pathlib import Path
from data.loaders.prosumer import ProsumerDataset
DATASET_REGISTRY: dict[str, type] = {'prosumer': ProsumerDataset}
_USE_CFG_VALUE = object()

def register_dataset(name: str, dataset_cls: type) -> None:
    DATASET_REGISTRY[name] = dataset_cls

def get_dataset_cls(name: str='prosumer') -> type:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}', available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]

def _normalized_optional_date(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text == '' else text

def _same_year_has_explicit_train_range(cfg) -> bool:
    return bool(_normalized_optional_date(cfg.data.train_start_date) or _normalized_optional_date(cfg.data.train_end_date))

def _resolve_split_dates(cfg, mode: str, *, override_start_date: str | None | object=_USE_CFG_VALUE, override_end_date: str | None | object=_USE_CFG_VALUE, override_exclude_start_date: str | None | object=_USE_CFG_VALUE, override_exclude_end_date: str | None | object=_USE_CFG_VALUE) -> tuple[int, str | None, str | None, str | None, str | None]:
    selected_year = int(cfg.data.train_year if mode == 'train' else cfg.data.test_year)
    start_date = _normalized_optional_date(cfg.data.train_start_date if mode == 'train' else cfg.data.test_start_date)
    end_date = _normalized_optional_date(cfg.data.train_end_date if mode == 'train' else cfg.data.test_end_date)
    exclude_start_date = None
    exclude_end_date = None
    if mode == 'train' and int(cfg.data.train_year) == int(cfg.data.test_year) and (not _same_year_has_explicit_train_range(cfg)) and (cfg.data.test_start_date or cfg.data.test_end_date):
        exclude_start_date = _normalized_optional_date(cfg.data.test_start_date)
        exclude_end_date = _normalized_optional_date(cfg.data.test_end_date)
    if override_start_date is not _USE_CFG_VALUE:
        start_date = _normalized_optional_date(override_start_date)
    if override_end_date is not _USE_CFG_VALUE:
        end_date = _normalized_optional_date(override_end_date)
    if override_exclude_start_date is not _USE_CFG_VALUE:
        exclude_start_date = _normalized_optional_date(override_exclude_start_date)
    if override_exclude_end_date is not _USE_CFG_VALUE:
        exclude_end_date = _normalized_optional_date(override_exclude_end_date)
    return (selected_year, start_date, end_date, exclude_start_date, exclude_end_date)

def build_dataset(cfg, mode: str='train', *, override_start_date: str | None | object=_USE_CFG_VALUE, override_end_date: str | None | object=_USE_CFG_VALUE, override_exclude_start_date: str | None | object=_USE_CFG_VALUE, override_exclude_end_date: str | None | object=_USE_CFG_VALUE):
    data_dir = Path(cfg.data.data_dir or Path(__file__).resolve().parents[2] / 'data')
    selected_year, start_date, end_date, exclude_start_date, exclude_end_date = _resolve_split_dates(cfg, mode, override_start_date=override_start_date, override_end_date=override_end_date, override_exclude_start_date=override_exclude_start_date, override_exclude_end_date=override_exclude_end_date)
    history_warmup_steps = int(getattr(cfg.forecast, 'history_window', 0)) if str(getattr(cfg.forecast, 'type', '')).strip().lower() == 'lstm' else 0
    return ProsumerDataset(data_dir=data_dir, episode_length=cfg.env.episode_limit, history_warmup_steps=history_warmup_steps, n_agents=cfg.env.num_agents, agent_profiles=list(cfg.data.agent_profiles), year=selected_year, start_date=start_date, end_date=end_date, exclude_start_date=exclude_start_date, exclude_end_date=exclude_end_date, load_components=list(cfg.data.load_components), pv_reference=str(cfg.data.pv_reference), pv_capacity_kw=list(cfg.data.pv_capacity_kw), load_scale=list(cfg.data.load_scale), pv_scale=list(cfg.data.pv_scale), node_ids=list(cfg.grid.agent_bus_ids))
