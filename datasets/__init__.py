"""Dataset loading and episode slicing.
数据集加载与 episode 切分。

Current implementations / 已实现数据集:
    - CsvPriceLoadDataset  -- loads price + load signals from CSV files

How to add a new dataset / 如何添加新数据集:
    1. Create ``datasets/your_dataset.py``, implementing ``BaseEpisodeDataset``
    2. Register in ``datasets/registry.py``::

           register_dataset("your_type", YourDataset)

    3. Use in config: ``cfg.data.dataset_type = "your_type"``

See ``datasets/base.py`` for the interface contract.
"""

from datasets.base import BaseEpisodeDataset
from datasets.csv_price_load import CsvPriceLoadDataset
from datasets.registry import DATASET_REGISTRY, build_dataset, get_dataset_cls, register_dataset

__all__ = [
    "BaseEpisodeDataset",
    "CsvPriceLoadDataset",
    "DATASET_REGISTRY",
    "build_dataset",
    "get_dataset_cls",
    "register_dataset",
]