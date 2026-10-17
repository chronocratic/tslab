"""The metric interface: score predictions against targets."""

from typing import Protocol

import torch

__all__ = ['Metric']


class Metric(Protocol):
    """A named score of predictions against targets.

    Pass metrics to ``evaluate`` as frozen dataclasses (or other
    fingerprintable objects), so their settings take part in caching.
    """

    @property
    def name(self) -> str:
        """Return the key under which the score is stored and logged."""
        ...

    def __call__(self, *, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Return the score."""
        ...
