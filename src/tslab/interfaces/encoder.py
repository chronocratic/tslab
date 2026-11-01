"""The encoder interfaces: what a trained model offers to produce representations."""

from typing import Protocol, runtime_checkable

import torch

__all__ = ['Encoder', 'SequenceEncoder']


@runtime_checkable
class Encoder(Protocol):
    """Turns a batch of inputs into one representation vector per sample."""

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode ``inputs`` of shape ``(N, ...)`` into representations ``(N, D)``."""
        ...


@runtime_checkable
class SequenceEncoder(Protocol):
    """Turns series into one representation per timestep, causally.

    The representation at step ``t`` may only depend on steps ``<= t``. The
    TS2Vec forecasting protocol relies on this to encode a whole series once
    and read off every forecast position.
    """

    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode series ``(N, T, C)`` into representations ``(N, T, D)``."""
        ...
