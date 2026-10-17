"""A small config shaped like the template's ``ClassifyConfig``, for the study tests.

Every union member is named with ``tyro.conf.subcommand(name=...)``, as Q-3
requires of a choice union. ``PRESETS`` mirrors the shape of the template's
preset registry, including a ``tslab.REQUIRED`` value (``ts2vec-any``).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

import tyro

import tslab


class SplitMode(Enum):
    """The splitting strategies of the mock data settings."""

    ARCHIVE = 'archive'
    MANUAL = 'manual'


@dataclass(frozen=True, kw_only=True)
class Jitter:
    """Mock augmentation: jitter."""

    sigma: float = 0.1


@dataclass(frozen=True, kw_only=True)
class Crop:
    """Mock augmentation: crop."""

    ratio: float = 0.5


Augmentations = (
    Annotated[Jitter, tyro.conf.subcommand(name='jitter')]
    | Annotated[Crop, tyro.conf.subcommand(name='crop')]
)


@dataclass(frozen=True, kw_only=True)
class TS2VecSettings:
    """Mock TS2Vec model settings."""

    depth: int = 10
    hidden_dims: int = 64
    lr: float = 1e-3
    use_mask: bool = True
    dims: tuple[int, ...] = (64, 128)
    augmentation: Augmentations = field(default_factory=Jitter)


@dataclass(frozen=True, kw_only=True)
class CoSTSettings:
    """Mock CoST model settings."""

    alpha: float = 0.05
    lr: float = 1e-3


Models = (
    Annotated[TS2VecSettings, tyro.conf.subcommand(name='ts2vec')]
    | Annotated[CoSTSettings, tyro.conf.subcommand(name='cost')]
)


@dataclass(frozen=True, kw_only=True)
class UCRSettings:
    """Mock UCR data settings."""

    dataset_folder_path: Path
    target_column_name: str = 'target'
    batch_size: int = 8
    splitting_strategy: SplitMode = SplitMode.ARCHIVE


@dataclass(frozen=True, kw_only=True)
class UEASettings:
    """Mock UEA data settings."""

    dataset_folder_path: Path
    batch_size: int = 16


DataChoice = (
    Annotated[UCRSettings, tyro.conf.subcommand(name='ucr')]
    | Annotated[UEASettings, tyro.conf.subcommand(name='uea')]
)


@dataclass(frozen=True, kw_only=True)
class Trainer:
    """Mock trainer settings."""

    max_epochs: int = 100
    deterministic: bool = False
    max_training_time: str | None = None


@dataclass(frozen=True, kw_only=True)
class ClassifyConfig:
    """Mock of the template's classification config."""

    model: Models
    data: DataChoice
    trainer: Trainer = field(default_factory=Trainer)
    noise_levels: tuple[float, ...] = (0.1, 0.5)
    seed: int = 0
    run_path: Path = Path('classification')
    output_root: Path = Path('runs')


PRESETS = {
    'ts2vec-paper': (
        'TS2Vec with non-default values of every kind the round trip must cover',
        ClassifyConfig(
            model=TS2VecSettings(
                depth=12, use_mask=False, dims=(32,), augmentation=Crop(ratio=0.3)
            ),
            data=UCRSettings(
                dataset_folder_path=Path('data/UCR/Coffee'), splitting_strategy=SplitMode.MANUAL
            ),
            trainer=Trainer(max_epochs=600, deterministic=True),
            noise_levels=(0.2,),
        ),
    ),
    'cost-paper': (
        'CoST with one non-default model value',
        ClassifyConfig(
            model=CoSTSettings(alpha=0.1),
            data=UCRSettings(dataset_folder_path=Path('data/UCR/Coffee')),
        ),
    ),
    'ts2vec-any': (
        'TS2Vec whose dataset folder the user gives on the command line',
        ClassifyConfig(
            model=TS2VecSettings(),
            data=UCRSettings(dataset_folder_path=tslab.REQUIRED, batch_size=4),
        ),
    ),
}
