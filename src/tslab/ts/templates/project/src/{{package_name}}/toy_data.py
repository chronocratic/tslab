"""Write small synthetic datasets, so both scripts run before you have real data.

The files are generated from fixed rules (no real data is involved):

- ``ToySpectra/``: a two-class dataset in the UCR archive format, for ``classify``;
- ``toy_hourly.csv.gz``: an hourly multivariate series in the ETT format, for
  ``forecast`` (``--data.variant ETTh1``).

Usage: ``python -m {{package_name}}.toy_data <folder>``.
"""

import gzip
from pathlib import Path
import sys

import numpy as np

CLASSIFICATION_NAME = 'ToySpectra'
FORECASTING_FILE = 'toy_hourly.csv.gz'

_LENGTH = 128
_SAMPLES_PER_CLASS = 12
_CLASS_BUMP_CENTERS = (0.35, 0.65)
# The ETTh1 variant splits at fixed hours (8640 / 11520 / 14400), so the series needs 14400.
_HOURS = 14_400
_START = np.datetime64('2016-07-01T00:00')
_COLUMNS = ('HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT')


def _spectrum(*, rng: np.random.Generator, label: int) -> np.ndarray:
    position = np.linspace(start=0.0, stop=1.0, num=_LENGTH)
    center = _CLASS_BUMP_CENTERS[label] + rng.normal(loc=0.0, scale=0.01)
    row = np.exp(-0.5 * ((position - 0.15) / 0.05) ** 2)
    row += 1.2 * np.exp(-0.5 * ((position - center) / 0.05) ** 2)
    row += rng.normal(loc=0.0, scale=0.05, size=_LENGTH)
    return (row - row.mean()) / row.std()


def write_classification(*, directory: Path) -> Path:
    """Write ``<directory>/ToySpectra/ToySpectra_{TRAIN,TEST}.arff`` and return the folder."""
    rng = np.random.default_rng(seed=0)
    folder = directory / CLASSIFICATION_NAME
    folder.mkdir(parents=True, exist_ok=True)
    header = [f'@relation {CLASSIFICATION_NAME}', '']
    header += [f'@attribute att{index + 1} numeric' for index in range(_LENGTH)]
    header += ['@attribute target {0,1}', '', '@data']
    for part in ('TRAIN', 'TEST'):
        lines = [
            ','.join([*(f'{value:.6g}' for value in _spectrum(rng=rng, label=label)), str(label)])
            for label in (0, 1)
            for _ in range(_SAMPLES_PER_CLASS)
        ]
        path = folder / f'{CLASSIFICATION_NAME}_{part}.arff'
        path.write_text('\n'.join([*header, *lines]) + '\n', encoding='utf-8')
    return folder


def write_forecasting(*, directory: Path) -> Path:
    """Write ``<directory>/toy_hourly.csv.gz`` and return its path."""
    rng = np.random.default_rng(seed=1)
    hours = np.arange(_HOURS)
    columns = [
        5.0
        + 3.0 * np.sin(2 * np.pi * hours / 24 + rng.uniform(low=0.0, high=2 * np.pi))
        + np.sin(2 * np.pi * hours / (24 * 7))
        + rng.normal(loc=0.0, scale=0.3, size=_HOURS)
        for _ in _COLUMNS
    ]
    dates = _START + hours.astype('timedelta64[h]')
    lines = [
        ','.join(['date', *_COLUMNS]),
        *(
            ','.join(
                [
                    np.datetime_as_string(dates[row], unit='s').replace('T', ' '),
                    *(f'{column[row]:.3f}' for column in columns),
                ]
            )
            for row in range(_HOURS)
        ),
    ]
    path = directory / FORECASTING_FILE
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=path, mode='wb', mtime=0) as file:
        file.write(('\n'.join(lines) + '\n').encode('utf-8'))
    return path


def main(*, directory: Path) -> None:
    """Write both datasets into ``directory``."""
    write_classification(directory=directory)
    write_forecasting(directory=directory)


if __name__ == '__main__':
    main(directory=Path(sys.argv[1] if len(sys.argv) > 1 else 'data'))
