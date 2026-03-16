import copy
from typing import Callable

from .hems_env import EnergyStorageEnv


class SingleEnergyStorageEnv(EnergyStorageEnv):
    def __init__(self, args, reward_fn: Callable = None, data_path: str = None, mode: str = "train"):
        single_args = copy.deepcopy(args)
        single_args.num_agents = 1
        super().__init__(args=single_args, mode=mode, data_path=data_path, reward_fn=reward_fn)
