"""``decode_dataclass``: strict decoding that reports every mismatch as ``ConfigFileError``."""

from dataclasses import dataclass, field
from typing import Annotated

import pytest

from tslab.core.config import (
    config_to_data,
    ConfigFileError,
    decode_dataclass,
    strip_annotated,
    union_members,
)
from tslab.core.fingerprint import NotFingerprinted


@dataclass(frozen=True, kw_only=True)
class Needs:
    name: str
    count: int
    optional: float = 1.0
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, kw_only=True)
class Checked:
    low: int = 0
    high: int = 1

    def __post_init__(self) -> None:
        """Reject an empty range."""
        if self.low > self.high:
            message = 'low must not exceed high'
            raise ValueError(message)


def test_decode_dataclass_builds_the_instance() -> None:
    assert decode_dataclass(data={'name': 'a', 'count': 2}, cls=Needs, location='top') == Needs(
        name='a', count=2
    )


def test_missing_required_fields_are_named() -> None:
    with pytest.raises(ConfigFileError, match=r'top: missing required field top\.count'):
        decode_dataclass(data={'name': 'a'}, cls=Needs, location='top')
    with pytest.raises(ConfigFileError, match=r'top\.name, top\.count'):
        decode_dataclass(data={}, cls=Needs, location='top')


def test_post_init_errors_become_config_file_errors() -> None:
    with pytest.raises(ConfigFileError, match='top: Checked rejected the values: low must not'):
        decode_dataclass(data={'low': 5, 'high': 1}, cls=Checked, location='top')


def test_unknown_keys_and_non_mapping_are_config_file_errors() -> None:
    with pytest.raises(ConfigFileError, match='unknown keys'):
        decode_dataclass(data={'name': 'a', 'count': 1, 'extra': 1}, cls=Needs, location='top')
    with pytest.raises(ConfigFileError, match='expected mapping for Needs'):
        decode_dataclass(data=[1], cls=Needs, location='top')


def test_non_dataclass_is_a_type_error() -> None:
    with pytest.raises(TypeError, match='not a dataclass'):
        decode_dataclass(data={}, cls=int, location='top')


def test_wrong_value_type_names_the_field() -> None:
    with pytest.raises(ConfigFileError, match=r'top\.count: expected int'):
        decode_dataclass(data={'name': 'a', 'count': 'x'}, cls=Needs, location='top')


@dataclass(frozen=True, kw_only=True)
class IntKeys:
    table: dict[int, int] = field(default_factory=dict)


def test_non_str_mapping_keys_are_rejected_on_save() -> None:
    with pytest.raises(TypeError, match='mapping keys must be str'):
        config_to_data(config=IntKeys(table={1: 2}))


def test_strip_annotated_and_union_members_are_public() -> None:
    assert strip_annotated(annotation=Annotated[int, NotFingerprinted]) is int
    assert strip_annotated(annotation=str) is str
    assert union_members(annotation=int | None) == (int, type(None))
    assert union_members(annotation=int) is None
