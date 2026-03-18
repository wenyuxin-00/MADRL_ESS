"""数据集注册表。

根据数据集类型名称（如 "csv_price_load"、"csv_prosumer"）
返回对应的数据集类或实例。

主要函数:
    build_dataset -- 根据配置构建数据集实例
"""

from __future__ import annotations

from pathlib import Path

from data.loaders.csv_price_load import CsvPriceLoadDataset
from data.loaders.csv_prosumer import CsvProsumerDataset

DATASET_REGISTRY: dict[str, type] = {
    "csv_price_load": CsvPriceLoadDataset,
    "csv_prosumer": CsvProsumerDataset,
}


def register_dataset(name: str, dataset_cls: type) -> None:
    """Register one dataset class."""
    DATASET_REGISTRY[name] = dataset_cls


def get_dataset_cls(name: str) -> type:
    """Return a dataset class by ``data.dataset_type``."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown data.dataset_type '{name}', available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def build_dataset(cfg, mode: str = "train"):
    """Build the configured dataset for train/test mode."""
    dataset_type = str(cfg.data.dataset_type)
    dataset_cls = get_dataset_cls(dataset_type)
    data_dir = Path(cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"))
    if dataset_type == "csv_prosumer":
        csv_name = "simbench_2016_train.csv" if mode == "train" else "simbench_2016_test.csv"
    else:
        csv_name = "train_prices.csv" if mode == "train" else "test_prices.csv"
    data_path = data_dir / csv_name

    build_kwargs = {
        "data_path": data_path,
        "episode_length": cfg.env.episode_limit,
        "n_agents": cfg.env.num_agents,
    }
    if dataset_type == "csv_prosumer":
        build_kwargs["metadata_path"] = data_dir / "simbench_2016_metadata.json"

    return dataset_cls(**build_kwargs)
