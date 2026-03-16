"""Optional base classes for actor/critic models.

These classes keep the model package structure explicit without introducing a
heavy abstraction layer. Concrete models can subclass them or plain
``nn.Module`` in future extensions.
"""

import torch.nn as nn


class ActorBase(nn.Module):
    """Base class marker for actor models."""


class CriticBase(nn.Module):
    """Base class marker for critic models."""
