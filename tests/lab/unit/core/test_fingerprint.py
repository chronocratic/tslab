from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Annotated, ClassVar

import pytest

from tslab.lab.core.fingerprint import (
    canonicalize,
    compute_fingerprint,
    hash_path_contents,
    NotFingerprinted,
)


class Color(Enum):
    RED = 1


class Mask(StrEnum):
    BINOMIAL = 'binomial'


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    hidden: int = 8
    dropout: float = 0.1
    workers: Annotated[int, NotFingerprinted] = 0


@dataclass(frozen=True, kw_only=True)
class OtherConfig:
    hidden: int = 8
    dropout: float = 0.1


@dataclass(frozen=True, kw_only=True)
class Nested:
    model: ModelConfig = field(default_factory=ModelConfig)
    tags: tuple[str, ...] = ('a',)


@dataclass(frozen=True, kw_only=True)
class PinnedName:
    _tslab_canonical_name: ClassVar[str] = 'legacy.module.PinnedName'
    value: int = 1


class Describes:
    def fingerprint_settings(self) -> object:
        return ModelConfig(hidden=3)


def fingerprint_of(value: object) -> str:
    return compute_fingerprint(document={'value': canonicalize(value=value)})


def test_same_value_same_fingerprint() -> None:
    assert fingerprint_of(Nested()) == fingerprint_of(Nested())


def test_fingerprint_is_16_hex_chars() -> None:
    fingerprint = fingerprint_of(1)
    assert len(fingerprint) == 16
    int(fingerprint, 16)


@pytest.mark.parametrize(
    'changed',
    [
        Nested(model=ModelConfig(hidden=9)),
        Nested(model=ModelConfig(dropout=0.2)),
        Nested(tags=('b',)),
    ],
)
def test_any_setting_change_changes_fingerprint(changed: Nested) -> None:
    assert fingerprint_of(changed) != fingerprint_of(Nested())


def test_type_name_is_part_of_fingerprint() -> None:
    assert fingerprint_of(ModelConfig()) != fingerprint_of(OtherConfig())


def test_fingerprint_ignores_a_canonical_name_attribute() -> None:
    expected = f'{PinnedName.__module__}.{PinnedName.__qualname__}'
    assert canonicalize(value=PinnedName()) == {'__type__': expected, 'value': 1}


def test_not_fingerprinted_field_is_ignored() -> None:
    assert fingerprint_of(ModelConfig(workers=8)) == fingerprint_of(ModelConfig(workers=0))


def test_scalars_paths_enums() -> None:
    assert canonicalize(value=0.1) == '0.1'
    assert canonicalize(value=Path('a') / 'b') == 'a/b'
    assert canonicalize(value=Mask.BINOMIAL) == 'binomial'
    assert canonicalize(value=(1, [2])) == [1, [2]]


def test_int_and_float_differ() -> None:
    assert fingerprint_of(1) != fingerprint_of(1.0)


def test_plain_floats_keep_their_repr() -> None:
    assert canonicalize(value=0.1) == '0.1'
    assert canonicalize(value=1e-07) == '1e-07'


def test_a_numpy_float_has_the_fingerprint_of_the_same_float() -> None:
    numpy = pytest.importorskip('numpy')
    assert canonicalize(value=numpy.float64(0.1)) == canonicalize(value=0.1)
    assert fingerprint_of(numpy.float64(0.1)) == fingerprint_of(0.1)


def test_numpy_scalars_have_the_fingerprint_of_the_same_python_number() -> None:
    numpy = pytest.importorskip('numpy')
    assert canonicalize(value=numpy.int64(3)) == 3
    assert isinstance(canonicalize(value=numpy.int64(3)), int)
    assert fingerprint_of(numpy.int32(3)) == fingerprint_of(3)
    assert canonicalize(value=numpy.float32(0.5)) == canonicalize(value=0.5)
    assert fingerprint_of(numpy.float32(0.5)) == fingerprint_of(0.5)


def test_mapping_order_does_not_matter() -> None:
    assert fingerprint_of({'a': 1, 'b': 2}) == fingerprint_of({'b': 2, 'a': 1})


def test_fingerprintable_uses_settings() -> None:
    assert canonicalize(value=Describes()) == canonicalize(value=ModelConfig(hidden=3))


class Opaque:
    pass


def test_unsupported_values_raise() -> None:
    with pytest.raises(TypeError, match='cannot fingerprint value of type Opaque'):
        canonicalize(value=Opaque())
    with pytest.raises(TypeError, match='non-str key'):
        canonicalize(value={1: 'a'})


def test_content_hash_follows_content_not_path(tmp_path: Path) -> None:
    first = tmp_path / 'a.bin'
    second = tmp_path / 'b.bin'
    first.write_bytes(b'same')
    second.write_bytes(b'same')
    assert hash_path_contents(path=first) == hash_path_contents(path=second)
    second.write_bytes(b'different!')
    assert hash_path_contents(path=first) != hash_path_contents(path=second)


def test_directory_hash_includes_relative_names(tmp_path: Path) -> None:
    (tmp_path / 'one').mkdir()
    (tmp_path / 'two').mkdir()
    (tmp_path / 'one' / 'x.txt').write_text('1', encoding='utf-8')
    (tmp_path / 'two' / 'y.txt').write_text('1', encoding='utf-8')
    assert hash_path_contents(path=tmp_path / 'one') != hash_path_contents(path=tmp_path / 'two')


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_path_contents(path=tmp_path / 'missing')


def test_plain_enum_cannot_be_fingerprinted() -> None:
    with pytest.raises(TypeError, match='configs hold choices as StrEnum'):
        canonicalize(value=Color.RED)


def test_enum_fingerprint_follows_the_value_not_the_class() -> None:
    class Other(StrEnum):
        BINOMIAL = 'binomial'
        CONTINUOUS = 'continuous'

    class Same(StrEnum):
        RENAMED = 'binomial'

    assert canonicalize(value=Other.BINOMIAL) == canonicalize(value=Mask.BINOMIAL)
    assert canonicalize(value=Same.RENAMED) == canonicalize(value=Mask.BINOMIAL)
    assert canonicalize(value=Other.CONTINUOUS) != canonicalize(value=Mask.BINOMIAL)
