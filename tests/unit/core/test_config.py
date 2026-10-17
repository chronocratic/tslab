from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
import math
from pathlib import Path
from typing import Annotated, Literal

import pytest
import yaml

from tslab.core.config import (
    config_from_yaml,
    config_to_yaml,
    ConfigFileError,
    load_config,
    parse_config,
    save_config,
)
from tslab.core.fingerprint import NotFingerprinted


class Mode(Enum):
    FAST = 'fast'
    SLOW = 'slow'


class MaskMode(StrEnum):  # the enums of chronocratic are StrEnums: str subclasses
    BINOMIAL = 'binomial'
    CONTINUOUS = 'continuous'


@dataclass(frozen=True, kw_only=True)
class TS2VecLike:
    hidden: int = 64
    lr: float = 1e-3
    augmentation: Callable[[int], int] | None = None  # like the augmentation fields of chronocratic


@dataclass(frozen=True, kw_only=True)
class TSTCCLike:  # the same fields as TS2VecLike on purpose: the union member must not be guessed
    hidden: int = 64
    lr: float = 1e-3
    augmentation: Callable[[int], int] | None = None


@dataclass(frozen=True, kw_only=True)
class RunOptions:  # a stand-in with the shapes of tslab.RunOptions (a tuple and a Literal)
    force_all: bool = False
    rerun: tuple[str, ...] = ()
    log_level: Literal['DEBUG', 'INFO'] = 'INFO'


@dataclass(frozen=True, kw_only=True)
class Config:
    model: TS2VecLike | TSTCCLike = field(default_factory=TS2VecLike)
    optional_model: TS2VecLike | TSTCCLike | None = None
    run_path: Path = Path('a/b')
    scales: tuple[float, ...] = (1.0, 2.0)
    pair: tuple[int, str] = (1, 'x')
    names: list[str] = field(default_factory=list)
    mode: Mode = Mode.FAST
    mask: MaskMode = MaskMode.BINOMIAL
    level: Literal['low', 'high'] = 'low'
    extra: dict[str, int] = field(default_factory=dict)
    workers: Annotated[int, NotFingerprinted] = 0
    run: RunOptions = field(default_factory=RunOptions)
    seed: int = 0


SAMPLE = Config(
    model=TSTCCLike(hidden=8),
    optional_model=TSTCCLike(lr=0.5),
    run_path=Path('x') / 'y',
    scales=(0.5, math.inf),
    pair=(2, 'y'),
    names=['a'],
    mode=Mode.SLOW,
    mask=MaskMode.CONTINUOUS,
    level='high',
    extra={'k': 1},
    workers=4,
    run=RunOptions(force_all=True, rerun=('a[b]',), log_level='DEBUG'),
    seed=3,
)


def round_trip(config: Config) -> Config:
    return config_from_yaml(config_class=Config, text=config_to_yaml(config=config))


def test_round_trip_keeps_every_value_and_union_member() -> None:
    assert round_trip(SAMPLE) == SAMPLE
    assert type(round_trip(SAMPLE).model) is TSTCCLike


def test_round_trip_defaults() -> None:
    assert round_trip(Config()) == Config()


def test_file_is_plain_yaml() -> None:
    document = yaml.safe_load(config_to_yaml(config=SAMPLE))
    assert document['format_version'] == 1
    assert document['config_type'] == 'test_config.Config'
    assert document['config']['model'] == {
        '__type__': 'TSTCCLike',
        'hidden': 8,
        'lr': 0.001,
        'augmentation': None,
    }
    assert document['config']['run_path'] == 'x/y'
    assert document['config']['mode'] == 'SLOW'
    assert document['config']['mask'] == 'CONTINUOUS'


def test_non_none_unsupported_value_raises_on_save() -> None:
    with pytest.raises(
        TypeError, match=r'config\.model\.augmentation: cannot save a value of type function'
    ):
        config_to_yaml(config=Config(model=TS2VecLike(augmentation=lambda value: value)))


def test_non_dataclass_config_raises() -> None:
    with pytest.raises(TypeError, match='dataclass instance'):
        config_to_yaml(config={'seed': 1})


def edited(**changes: object) -> str:
    document = yaml.safe_load(config_to_yaml(config=Config()))
    document['config'].update(changes)
    return yaml.safe_dump(document)


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        ({'sede': 1}, r"config: unknown keys \['sede'\]"),
        ({'seed': 'one'}, r"config\.seed: expected int, got 'one'"),
        ({'seed': True}, r'config\.seed: expected int, got True'),
        ({'scales': ['abc']}, r"config\.scales\[0\]: expected float, got 'abc'"),
        ({'mode': 'MEDIUM'}, r"config\.mode: expected one of \['FAST', 'SLOW'\]"),
        ({'level': 'mid'}, r"config\.level: expected one of \['low', 'high'\]"),
        ({'model': {'__type__': 'Nope'}}, r"config\.model: __type__ 'Nope' is not one of"),
        ({'model': {'hidden': 1}}, r'config\.model: missing __type__'),
        ({'optional_model': 3}, r'config\.optional_model: 3 matches no member of the union'),
    ],
)
def test_strict_loading(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigFileError, match=message):
        config_from_yaml(config_class=Config, text=edited(**changes))


def test_missing_keys_take_defaults() -> None:
    document = yaml.safe_load(config_to_yaml(config=Config()))
    document['config'] = {'seed': 5}
    assert config_from_yaml(config_class=Config, text=yaml.safe_dump(document)) == Config(seed=5)


def test_python_tags_are_refused() -> None:
    text = 'format_version: 1\nconfig: !!python/object/apply:os.system ["echo pwned"]\n'
    with pytest.raises(yaml.YAMLError):
        config_from_yaml(config_class=Config, text=text)


def test_not_a_config_file() -> None:
    with pytest.raises(ConfigFileError, match='not a TSLab config file'):
        config_from_yaml(config_class=Config, text='a: 1\n')


def test_other_config_type_warns(caplog: pytest.LogCaptureFixture) -> None:
    @dataclass(frozen=True, kw_only=True)
    class Old:
        seed: int = 0

    @dataclass(frozen=True, kw_only=True)
    class Renamed:
        seed: int = 0

    assert config_from_yaml(
        config_class=Renamed, text=config_to_yaml(config=Old(seed=2))
    ) == Renamed(seed=2)
    assert 'loading it as test_config.test_other_config_type_warns.<locals>.Renamed' in caplog.text


def test_save_and_load(tmp_path: Path) -> None:
    save_config(config=SAMPLE, path=tmp_path / 'config.yaml')
    assert load_config(config_class=Config, path=tmp_path / 'config.yaml') == SAMPLE
    with pytest.raises(TypeError):
        load_config(
            config_class=Config,
            path=str(tmp_path / 'config.yaml'),  # ty: ignore[invalid-argument-type]
        )


def test_parse_config_accepts_flags_after_subcommands() -> None:
    config = parse_config(
        config_class=Config,
        args=[
            'model:tstcc-like',
            '--model.hidden',
            '3',
            '--seed',
            '7',
            '--run.rerun',
            'a[x]',
            'b[y]',
            '--run.force-all',
        ],
    )
    assert config.model == TSTCCLike(hidden=3)
    assert config.seed == 7
    assert config.run == RunOptions(force_all=True, rerun=('a[x]', 'b[y]'))


def test_parse_config_with_default_overrides_loaded_values() -> None:
    config = parse_config(config_class=Config, default=SAMPLE, args=['--run.no-force-all'])
    assert config.model == SAMPLE.model
    assert config.run.force_all is False


def test_preset_fixes_its_union_choices() -> None:
    presets = {'tstcc': ('TSTCC, small', Config(model=TSTCCLike(hidden=8)))}
    config = parse_config(config_class=Config, presets=presets, args=['tstcc', '--model.lr', '0.5'])
    assert config.model == TSTCCLike(hidden=8, lr=0.5)
    with pytest.raises(SystemExit):
        parse_config(config_class=Config, presets=presets, args=['tstcc', 'model:ts2-vec-like'])
    free = parse_config(
        config_class=Config, presets=presets, args=['no-preset', 'model:ts2-vec-like']
    )
    assert free.model == TS2VecLike()


def test_no_preset_is_a_reserved_name() -> None:
    with pytest.raises(ValueError, match='reserved'):
        parse_config(config_class=Config, presets={'no-preset': ('x', Config())}, args=[])


def test_parse_config_presets() -> None:
    presets = {
        'small': ('A small run', Config(seed=1)),
        'big': ('A big run', Config(seed=2, names=['b'])),
    }
    config = parse_config(config_class=Config, presets=presets, args=['big', '--seed', '9'])
    assert config == Config(seed=9, names=['b'])
    config = parse_config(config_class=Config, presets=presets, args=['no-preset', '--seed', '3'])
    assert config == Config(seed=3)
    with pytest.raises(ValueError, match='either presets= or default='):
        parse_config(config_class=Config, presets=presets, default=Config(), args=[])
