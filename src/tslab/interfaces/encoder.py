"""The encoder interfaces: what a trained model offers to produce representations."""

from dataclasses import dataclass
from typing import Annotated, cast, Protocol, runtime_checkable

import torch

from tslab.core.fingerprint import NotFingerprinted

__all__ = ['Encoder', 'ModelAsEncoder', 'SequenceEncoder', 'resolve_device']


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
    TS2Vec forecasting protocol relies on this to
    encode a whole series once and read off every forecast position.
    """

    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode series ``(N, T, C)`` into representations ``(N, T, D)``."""
        ...


class _EncoderWithSequence(Encoder, SequenceEncoder, Protocol):
    """An object offering both encodings."""


def resolve_device(*, device: str) -> torch.device:
    """Turn a device name into a ``torch.device``; ``'auto'`` prefers CUDA.

    Args:
        device: ``'auto'``, or any name ``torch.device`` accepts (``'cpu'``,
            ``'cuda'``, ``'cuda:1'``).

    Returns:
        The device.
    """
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


@dataclass(frozen=True, kw_only=True)
class ModelAsEncoder:
    """Encoder factory for models that already have ``encode(*, inputs)``.

    Pass it as ``encoder=`` to the ready encode stages. It also offers
    ``encode_sequence`` when the model has it. On first use, the model is
    moved to ``device`` and switched to evaluation mode; inputs move to the
    device in chunks of ``batch_size`` rows, and the results come back to the
    CPU.

    Attributes:
        device: ``'auto'`` (CUDA when available, else the CPU) or a device
            name. It does not affect the fingerprint, so changing it reuses
            cached representations. ``'cpu'`` reproduces representations
            computed on the CPU bit for bit.
        batch_size: Rows per call to the model; ``None`` encodes each split
            in one call.
    """

    device: Annotated[str, NotFingerprinted] = 'auto'
    batch_size: int | None = None

    def __call__(self, model: Encoder) -> Encoder:
        """Wrap ``model``; nothing moves until the first encode call.

        Raises:
            TypeError: If ``model`` has no ``encode`` method.
        """
        if not isinstance(model, Encoder):
            message = (
                f'{type(model).__qualname__} has no encode(*, inputs); write that method, or '
                'pass an encoder factory for this model library as encoder='
            )
            raise TypeError(message)
        device = resolve_device(device=self.device)
        if isinstance(model, SequenceEncoder):
            sequence_model = cast('_EncoderWithSequence', model)
            return _DeviceSequenceEncoder(
                model=sequence_model, device=device, batch_size=self.batch_size
            )
        return _DeviceEncoder(model=model, device=device, batch_size=self.batch_size)


class _DeviceEncoder:
    """Runs a model ``encode`` on a device, in chunks, and returns CPU tensors."""

    def __init__(self, *, model: Encoder, device: torch.device, batch_size: int | None) -> None:
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._prepared = False

    def _prepare(self) -> None:
        if self._prepared:
            return
        if isinstance(self._model, torch.nn.Module):
            self._model.to(device=self._device)
            self._model.eval()
        self._prepared = True

    def _chunks(self, *, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._prepare()
        size = len(inputs) if self._batch_size is None else self._batch_size
        return cast('tuple[torch.Tensor, ...]', inputs.split(split_size=max(size, 1)))

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode ``inputs`` with the model ``encode``."""
        return torch.cat(
            tensors=[
                self._model.encode(inputs=chunk.to(device=self._device)).cpu()
                for chunk in self._chunks(inputs=inputs)
            ]
        )


class _DeviceSequenceEncoder(_DeviceEncoder):
    """``_DeviceEncoder`` that also runs the ``encode_sequence`` of the model."""

    def __init__(
        self, *, model: _EncoderWithSequence, device: torch.device, batch_size: int | None
    ) -> None:
        super().__init__(model=model, device=device, batch_size=batch_size)
        self._sequence_model = model

    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode ``inputs`` with the ``encode_sequence`` of the model."""
        return torch.cat(
            tensors=[
                self._sequence_model.encode_sequence(inputs=chunk.to(device=self._device)).cpu()
                for chunk in self._chunks(inputs=inputs)
            ]
        )
