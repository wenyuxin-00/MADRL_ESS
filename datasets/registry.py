"""数据集 registry。

当前只注册一个最小可用数据集实现，但保留最小 registry 骨架，
方便未来按 `dataset_type` 扩展新的数据来源。
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
