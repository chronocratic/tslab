"""``ChronocraticEncoder``: the TSLab encode calls, mapped to the ``encode`` of chronocratic."""

from dataclasses import dataclass
from typing import Annotated

from chronocratic.models._mixin.encoding import BasicEncodingMixin
from chronocratic.models.convolutional.dilated._mixin.encoding import (
    DecompositionEncodingMixin,
    PoolingEncodingMixin,
)
from chronocratic.models.enums import EncodingOutputShape
import torch

from tslab.core.fingerprint import NotFingerprinted
from tslab.interfaces import Encoder
from tslab.interfaces.encoder import resolve_device

__all__ = ['ChronocraticEncoder']


@dataclass(frozen=True, kw_only=True)
class ChronocraticEncoder:
    """Encoder factory for chronocratic models; pass it as ``encoder=`` to the encode stages.

    Dilated models (TS2Vec, AutoTCL, CoST) also offer ``encode_sequence``: one
    causal representation per timestep, computed with a sliding window, as in
    the TS2Vec forecasting protocol. Other chronocratic
    models offer one vector per sample.

    Attributes:
        device: ``'auto'`` (CUDA when available, else the CPU) or a device
            name. It does not affect the fingerprint; ``'cpu'`` reproduces
            representations computed on the CPU bit for bit.
        num_workers: Data-loading workers inside the ``encode`` of chronocratic.
            Not fingerprinted.
        batch_size: Samples per forward pass.
        sliding_padding: Past timesteps each timestep sees when a dilated
            model encodes per timestep.
    """

    device: Annotated[str, NotFingerprinted] = 'auto'
    num_workers: Annotated[int, NotFingerprinted] = 0
    batch_size: int = 256
    sliding_padding: int = 200

    def __call__(self, model: torch.nn.Module) -> Encoder:
        """Wrap ``model``; nothing moves until the first encode call.

        Raises:
            TypeError: If ``model`` has no chronocratic encoding mixin.
        """
        if isinstance(model, PoolingEncodingMixin | DecompositionEncodingMixin):
            return _SequenceEncoder(encoding=model, module=model, settings=self)
        if isinstance(model, BasicEncodingMixin):
            return _VectorEncoder(encoding=model, module=model, settings=self)
        message = (
            f'{type(model).__qualname__} has no chronocratic encoding mixin; '
            'ChronocraticEncoder only serves chronocratic models'
        )
        raise TypeError(message)


type _Encoding = BasicEncodingMixin | PoolingEncodingMixin | DecompositionEncodingMixin
type _DilatedEncoding = PoolingEncodingMixin | DecompositionEncodingMixin


class _VectorEncoder:
    """One vector per sample through the ``encode`` of chronocratic.

    The model is held twice, under two types: as its encoding mixin, whose
    ``encode`` signature ty checks, and as a ``torch.nn.Module``, for moving it.
    """

    def __init__(
        self, *, encoding: _Encoding, module: torch.nn.Module, settings: ChronocraticEncoder
    ) -> None:
        self._encoding = encoding
        self._module = module
        self._settings = settings
        self._prepared = False

    def _prepare(self) -> None:
        if not self._prepared:
            self._module.to(device=resolve_device(device=self._settings.device))
            self._module.eval()
            self._prepared = True

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode ``inputs`` ``(N, T, C)`` into ``(N, D)``."""
        self._prepare()
        return self._encoding.encode(
            data=inputs,
            batch_size=self._settings.batch_size,
            num_workers=self._settings.num_workers,
            output=EncodingOutputShape.VECTOR,
        )


class _SequenceEncoder(_VectorEncoder):
    """``_VectorEncoder`` plus one causal vector per timestep."""

    def __init__(
        self, *, encoding: _DilatedEncoding, module: torch.nn.Module, settings: ChronocraticEncoder
    ) -> None:
        super().__init__(encoding=encoding, module=module, settings=settings)
        self._dilated = encoding

    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        """Encode ``inputs`` ``(N, T, C)`` into ``(N, T, D)``, causally."""
        self._prepare()
        return self._dilated.encode(
            data=inputs,
            batch_size=self._settings.batch_size,
            num_workers=self._settings.num_workers,
            encoding_window=None,
            causal=True,
            sliding_length=1,
            sliding_padding=self._settings.sliding_padding,
            output=EncodingOutputShape.SEQUENCE,
        )
