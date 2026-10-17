from typing import override

import pytest
import torch

from tslab.core.fingerprint import canonicalize
from tslab.interfaces import DownstreamModel, Encoder, SequenceEncoder
from tslab.interfaces.encoder import ModelAsEncoder, resolve_device


class VectorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(in_features=3, out_features=2)
        self.calls = 0

    def encode(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.layer(inputs.mean(dim=1))  # ty: ignore[unsound-return-statement]


class SequenceModel(VectorModel):
    def encode_sequence(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs).cumsum(dim=1)  # ty: ignore[unsound-return-statement]


def series(*, count: int = 5) -> torch.Tensor:
    return torch.randn(count, 4, 3, generator=torch.Generator().manual_seed(0))


def test_a_vector_model_gives_a_vector_encoder() -> None:
    encoder = ModelAsEncoder(device='cpu')(VectorModel())
    assert isinstance(encoder, Encoder)
    assert not isinstance(encoder, SequenceEncoder)
    assert encoder.encode(inputs=series()).shape == (5, 2)


def test_a_sequence_model_also_gives_sequence_encodings() -> None:
    model = SequenceModel()
    encoder = ModelAsEncoder(device='cpu')(model)
    assert isinstance(encoder, SequenceEncoder)
    assert torch.equal(
        encoder.encode_sequence(inputs=series()), model.encode_sequence(inputs=series())
    )


def test_a_model_without_encode_is_rejected() -> None:
    linear = torch.nn.Linear(in_features=2, out_features=2)
    with pytest.raises(TypeError, match='encode'):
        ModelAsEncoder()(linear)  # ty: ignore[invalid-argument-type]


def test_the_model_is_prepared_on_first_use_only() -> None:
    model = VectorModel()
    encoder = ModelAsEncoder(device='cpu')(model)
    assert model.training
    encoder.encode(inputs=series())
    assert not model.training


def test_batches_give_the_same_result_as_one_call() -> None:
    model = VectorModel()
    whole = ModelAsEncoder(device='cpu')(model).encode(inputs=series())
    calls_before = model.calls
    chunked = ModelAsEncoder(device='cpu', batch_size=2)(model).encode(inputs=series())
    assert model.calls - calls_before == 3
    torch.testing.assert_close(actual=chunked, expected=whole)


def test_results_come_back_on_the_cpu() -> None:
    encoded = ModelAsEncoder(device='auto')(VectorModel()).encode(inputs=series())
    assert encoded.device == torch.device('cpu')


def test_device_does_not_change_the_fingerprint() -> None:
    assert canonicalize(value=ModelAsEncoder(device='cpu')) == canonicalize(
        value=ModelAsEncoder(device='cuda')
    )
    assert canonicalize(value=ModelAsEncoder(batch_size=8)) != canonicalize(
        value=ModelAsEncoder(batch_size=16)
    )


def test_auto_device_prefers_cuda() -> None:
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert resolve_device(device='auto').type == expected
    assert resolve_device(device='cpu') == torch.device('cpu')


class Constant(DownstreamModel):
    @override
    def fit(
        self,
        *,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        val_inputs: torch.Tensor | None,
        val_targets: torch.Tensor | None,
    ) -> None:
        pass

    @override
    def predict(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(inputs))


def test_default_metrics_explain_what_to_do() -> None:
    with pytest.raises(NotImplementedError, match='metrics='):
        Constant().default_metrics(predictions=torch.zeros(2), targets=torch.zeros(2))
