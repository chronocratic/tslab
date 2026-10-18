from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import pytest
import tyro

import tslab
import tslab.core.config


@dataclass(frozen=True, kw_only=True)
class Small:
    size: int = 1


@dataclass(frozen=True, kw_only=True)
class Large:
    size: int = 9
    depth: int = 2


@dataclass(frozen=True, kw_only=True)
class NamedChoices:
    choice: (
        Annotated[Small, tyro.conf.subcommand(name='small')]
        | Annotated[Large, tyro.conf.subcommand(name='large')]
    ) = field(default_factory=Small)
    optional: (
        Annotated[Small, tyro.conf.subcommand(name='small')]
        | Annotated[None, tyro.conf.subcommand(name='default')]
    ) = None


def test_named_union_members_round_trip(tmp_path: Path) -> None:
    config = tslab.parse_config(
        config_class=NamedChoices,
        args=['choice:large', '--choice.depth', '5', 'optional:small', '--optional.size', '3'],
    )
    assert config == NamedChoices(choice=Large(depth=5), optional=Small(size=3))
    path = tmp_path / 'config.yaml'
    tslab.save_config(config=config, path=path)
    assert tslab.load_config(config_class=NamedChoices, path=path) == config


def test_named_none_member_round_trips(tmp_path: Path) -> None:
    config = tslab.parse_config(config_class=NamedChoices, args=[])
    assert config.optional is None
    path = tmp_path / 'config.yaml'
    tslab.save_config(config=config, path=path)
    assert tslab.load_config(config_class=NamedChoices, path=path) == config


def _same_named_members() -> tuple[type, type]:
    @dataclass(frozen=True, kw_only=True)
    class Twin:
        size: int = 1

    first = Twin

    @dataclass(frozen=True, kw_only=True)
    class Twin:  # why: a second class with the same name, as in two modules
        size: int = 2

    return first, Twin


def test_same_named_union_members_are_rejected_on_save() -> None:
    first, second = _same_named_members()

    @dataclass(frozen=True, kw_only=True)
    class Holder:
        choice: first | second = field(default_factory=first)  # ty: ignore[invalid-type-form]

    with pytest.raises(TypeError, match=r"share the class name \['Twin'\]"):
        tslab.core.config.config_to_data(config=Holder())
