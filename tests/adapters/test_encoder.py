import typing
from typing import Any

import pytest
import torch

from tslab.adapters.chronocratic_models import (
    build_chronocratic_model,
    ChronocraticEncoder,
    ChronocraticModelChoice,
)
from tslab.core.fingerprint import canonicalize
from tslab.interfaces import Encoder, SequenceEncoder
from tslab.tasks import TSClassificationDataInfo

pytestmark = pytest.mark.chronocratic

INFO = TSClassificationDataInfo(num_channels=2, length=48, num_classes=2)
INPUTS = torch.randn(5, 48, 2)


def model(*, name: str) -> Any:  # noqa: ANN401 # why: generated classes have no static type
    for member in typing.get_args(ChronocraticModelChoice):
        settings_class, subcommand = typing.get_args(member)[:2]
        if subcommand.name == name:
            return build_chronocratic_model(config=settings_class(), data_info=INFO)
    raise KeyError(name)


@pytest.mark.parametrize('name', ['ts2vec', 'autotcl', 'cost'])
def test_dilated_models_offer_both_encodings(name: str) -> None:
    encoder = ChronocraticEncoder(device='cpu', batch_size=2)(model(name=name))
    assert isinstance(encoder, SequenceEncoder)
    vectors = encoder.encode(inputs=INPUTS)
    sequences = encoder.encode_sequence(inputs=INPUTS)
    assert vectors.shape[0] == 5
    assert vectors.ndim == 2
    assert sequences.shape[:2] == (5, 48)
    assert sequences.ndim == 3
    assert vectors.device.type == 'cpu'


@pytest.mark.parametrize('name', ['tstcc', 'simclr', 'timenet'])
def test_basic_models_offer_vectors_only(name: str) -> None:
    encoder = ChronocraticEncoder(device='cpu')(model(name=name))
    assert isinstance(encoder, Encoder)
    assert not isinstance(encoder, SequenceEncoder)
    assert encoder.encode(inputs=INPUTS).shape[0] == 5


def test_sequence_encoding_is_causal() -> None:
    encoder = ChronocraticEncoder(device='cpu')(model(name='ts2vec'))
    assert isinstance(encoder, SequenceEncoder)
    changed = INPUTS.clone()
    changed[:, 30:, :] += 5.0
    before = encoder.encode_sequence(inputs=INPUTS)
    after = encoder.encode_sequence(inputs=changed)
    torch.testing.assert_close(actual=after[:, :30], expected=before[:, :30])
    assert not torch.allclose(after[:, 30:], before[:, 30:])


def test_unknown_models_are_rejected() -> None:
    with pytest.raises(TypeError, match='Linear'):
        ChronocraticEncoder()(torch.nn.Linear(in_features=2, out_features=2))


def test_creating_the_encoder_moves_nothing() -> None:
    built = model(name='ts2vec')
    built.train()
    ChronocraticEncoder(device='cpu')(built)
    assert built.training


def test_only_result_changing_settings_are_fingerprinted() -> None:
    base = canonicalize(value=ChronocraticEncoder())
    assert canonicalize(value=ChronocraticEncoder(device='cpu', num_workers=4)) == base
    assert canonicalize(value=ChronocraticEncoder(sliding_padding=100)) != base
    assert canonicalize(value=ChronocraticEncoder(batch_size=32)) != base
