"""Dataset registry and factory.
数据集注册表与工厂函数。

How to add a new dataset / 如何添加新数据集:
    1. Create ``datasets/your_dataset.py`` implementing ``BaseEpisodeDataset``
       - Must provide ``num_episodes()`` and ``get_episode(idx)``
       - ``get_episode()`` returns ``{"signals": {"price": ..., "load": ...}, "meta": {...}}``
    2. Register here::

           register_dataset("your_type", YourDataset)

    3. Update ``build_dataset()`` below if your dataset needs different construction args
    4. Use in config: ``cfg.data.dataset_type = "your_type"``
"""

from __future__ import annotations

from pathlib import Path

from datasets.csv_price_load import CsvPriceLoadDataset

DATASET_REGISTRY: dict[str, type] = {
    "csv_price_load": CsvPriceLoadDataset,
}


def register_dataset(name: str, dataset_cls: type) -> None:
    """注册一个数据集类。"""
    DATASET_REGISTRY[name] = dataset_cls


def get_dataset_cls(name: str) -> type:
    """按名称读取数据集类。"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown data.dataset_type '{name}', available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def build_dataset(cfg, mode: str = "train"):
    """按配置构建数据集。"""
    dataset_cls = get_dataset_cls(cfg.data.dataset_type)
    data_dir = Path(cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"))
    csv_name = "train_prices.csv" if mode == "train" else "test_prices.csv"
    data_path = data_dir / csv_name

    return dataset_cls(
        data_path=data_path,
        episode_length=cfg.env.episode_limit,
        n_agents=cfg.env.num_agents,
    )
