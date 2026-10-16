import json
import math
from pathlib import Path

import pytest
import torch

from tslab.artifact import (
    is_available,
    Json,
    Metrics,
    Tensors,
    UnavailableArtifact,
    UnavailableArtifactError,
    UnavailableReason,
)
from tslab.fingerprint import canonicalize


def test_json_round_trip(tmp_path: Path) -> None:
    Json(data={'b': 1, 'a': [1, 2]}).save(directory=tmp_path)
    assert Json.load(directory=tmp_path).data == {'a': [1, 2], 'b': 1}


def test_json_writes_non_finite_as_null(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    Json(data={'x': math.nan, 'y': [math.inf]}).save(directory=tmp_path)
    assert json.loads((tmp_path / 'data.json').read_text(encoding='utf-8')) == {
        'x': None,
        'y': [None],
    }
    assert 'non-finite' in caplog.text


def test_metrics_round_trip_nan(tmp_path: Path) -> None:
    Metrics(data={'accuracy': 0.5, 'loss': math.nan}).save(directory=tmp_path)
    values = Metrics.load(directory=tmp_path).values
    assert values['accuracy'] == 0.5
    assert math.isnan(values['loss'])


def test_metrics_reject_non_numbers() -> None:
    with pytest.raises(TypeError, match='must be int or float'):
        Metrics(data={'flag': True})


def test_tensors_round_trip_shared_and_noncontiguous(tmp_path: Path) -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    Tensors(tensors={'base': base, 'transposed': base.T, 'row': base[0]}, metadata={'k': 'v'}).save(
        directory=tmp_path
    )
    loaded = Tensors.load(directory=tmp_path)
    assert torch.equal(loaded.tensors['transposed'], base.T)
    assert torch.equal(loaded.tensors['row'], base[0])
    assert loaded.metadata == {'k': 'v'}


def test_external_fingerprint_follows_content(tmp_path: Path) -> None:
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    first.write_text('{"a": 1}', encoding='utf-8')
    second.write_text('{"a": 1}', encoding='utf-8')
    fingerprint = Json.external(path=first).tslab_fingerprint
    assert fingerprint == Json.external(path=second).tslab_fingerprint
    second.write_text('{"a": 2}', encoding='utf-8')
    assert fingerprint != Json.external(path=second).tslab_fingerprint


def test_external_type_is_part_of_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / 'data.json'
    path.write_text('{"a": 1}', encoding='utf-8')
    assert (
        Json.external(path=path).tslab_fingerprint != Metrics.external(path=path).tslab_fingerprint
    )


def test_external_missing_and_str(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Json.external(path=tmp_path / 'missing.json')
    with pytest.raises(TypeError):
        Json.external(path='x.json')  # ty: ignore[invalid-argument-type]


def test_fresh_artifact_cannot_be_fingerprinted() -> None:
    with pytest.raises(TypeError, match='not produced by a stage'):
        canonicalize(value=Json(data={}))


def test_unavailable_artifact_raises_on_use() -> None:
    placeholder = UnavailableArtifact(
        instance_id='transform[sma]', reason=UnavailableReason.FAILED, detail='boom'
    )
    assert not is_available(artifact=placeholder)
    with pytest.raises(UnavailableArtifactError, match=r'transform\[sma\] is unavailable'):
        _ = placeholder.tensors
