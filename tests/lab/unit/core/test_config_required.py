"""``tslab.REQUIRED``: the optional preset value the user must give."""

from dataclasses import dataclass
from pathlib import Path

import pytest

import tslab
from tslab.core import codec as core_codec


@dataclass(frozen=True, kw_only=True)
class Data:
    folder: Path
    batch_size: int = 8


@dataclass(frozen=True, kw_only=True)
class Config:
    data: Data
    seed: int = 0


PRESETS = {
    'any': ('give the folder', Config(data=Data(folder=tslab.REQUIRED, batch_size=4))),
    'fixed': ('fixed', Config(data=Data(folder=Path('d')))),
}


def test_required_preset_value_must_be_given(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        tslab.parse_config(config_class=Config, presets=PRESETS, args=['any'])
    assert error.value.code == 2
    assert '--data.folder' in capsys.readouterr().err


def test_required_preset_value_given_keeps_the_rest() -> None:
    config = tslab.parse_config(
        config_class=Config, presets=PRESETS, args=['any', '--data.folder', 'x']
    )
    assert config == Config(data=Data(folder=Path('x'), batch_size=4))


def test_presets_without_the_marker_are_unchanged() -> None:
    config = tslab.parse_config(config_class=Config, presets=PRESETS, args=['fixed'])
    assert config == Config(data=Data(folder=Path('d')))


def test_saving_an_unparsed_preset_with_the_marker_raises() -> None:
    with pytest.raises(TypeError):
        core_codec.config_to_data(config=PRESETS['any'][1])
