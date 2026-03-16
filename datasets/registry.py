"""
datasets/registry.py
职责：数据集注册表与工厂函数。

用法：
    from datasets.registry import build_dataset
    train_ds = build_dataset(args, mode='train')
    test_ds  = build_dataset(args, mode='test')

后续扩展：注册新数据集类型只需在 DATASET_REGISTRY 中添加条目。
"""

from pathlib import Path
from datasets.csv_price_load import CsvPriceLoadDataset

DATASET_REGISTRY: dict = {
    "csv_price_load": CsvPriceLoadDataset,
}


def build_dataset(args, mode: str = "train"):
    """根据 args.dataset_type 构建数据集实例。

    Parameters
    ----------
    args : Config
        超参数对象。须包含 episode_limit, num_agents。
        可选：dataset_type（默认 "csv_price_load"）、data_dir。
    mode : str
        "train" 或 "test"，决定使用哪个 CSV 文件。

    Returns
    -------
    BaseEpisodeDataset
    """
    dataset_type = getattr(args, "dataset_type", "csv_price_load")
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"未知 dataset_type '{dataset_type}'，可选: {list(DATASET_REGISTRY)}")

    # 默认数据路径：项目根目录 data/
    data_dir = Path(getattr(args, "data_dir", None) or
                    Path(__file__).resolve().parent.parent / "data")
    csv_name = "train_prices.csv" if mode == "train" else "test_prices.csv"
    data_path = data_dir / csv_name

    return DATASET_REGISTRY[dataset_type](
        data_path=data_path,
        episode_length=int(args.episode_limit),
        n_agents=int(args.num_agents),
    )
