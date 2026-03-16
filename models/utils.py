"""Shared model utilities."""

import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    """Apply orthogonal initialization to a module in the legacy way."""
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)
