from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    return torch.device(str(name))
