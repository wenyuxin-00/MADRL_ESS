from typing import Callable

from .hems_env import EnergyStorageEnv


class MultiEnergyStorageEnv(EnergyStorageEnv):
    def __init__(self, args, reward_fn: Callable, data_path: str = None, mode: str = "train"):
        super().__init__(args=args, reward_fn=reward_fn, data_path=data_path, mode=mode)
