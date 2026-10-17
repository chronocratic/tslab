"""Synthetic stand-ins for a UCR classification dataset and an ETT forecasting dataset.

The files only resemble the real ones in format and rough statistics; they are
generated from the constants below and never derived from real data. Run this
file to regenerate the committed copies in ``tests/adapters/data``.
"""

import gzip
from pathlib import Path

import numpy as np

DATA_DIRECTORY = Path(__file__).parent / 'data'
CLASSIFICATION_NAME = 'ToySpectra'
FORECASTING_FILE = 'toy_hourly.csv.gz'

_LENGTH = 286
_TRAIN_COUNTS = (14, 14)
_TEST_COUNTS = (15, 13)
_SHARED_BUMPS = ((0.15, 0.04, 1.0), (0.5, 0.08, 0.6), (0.85, 0.05, 0.8))  # centre, width, height
_CLASS_BUMP_CENTRES = (0.35, 0.65)

_HOURS = 14_400
_START = np.datetime64('2016-07-01T00:00')
# column: (level, scale), close to the real ETTh1 means and standard deviations
_COLUMNS = {
    'HUFL': (7.4, 7.1),
    'HULL': (2.2, 2.0),
    'MUFL': (4.3, 6.8),
    'MULL': (0.9, 1.8),
    'LUFL': (3.1, 1.2),
    'LULL': (0.9, 0.6),
}
_OIL_TEMPERATURE = (13.3, 8.6)


def _spectrum(*, rng: np.random.Generator, label: int) -> np.ndarray:
    position = np.linspace(start=0.0, stop=1.0, num=_LENGTH)
    bumps = [*_SHARED_BUMPS, (_CLASS_BUMP_CENTRES[label], 0.05, 1.2)]
    row = np.zeros(_LENGTH)
    for centre, width, height in bumps:
        shifted = centre + rng.normal(loc=0.0, scale=0.01)
        wide = width * rng.uniform(low=0.9, high=1.1)
        row += height * np.exp(-0.5 * ((position - shifted) / wide) ** 2)
    row += rng.normal(loc=0.0, scale=0.02, size=_LENGTH)
    return (row - row.mean()) / row.std()


def _write_arff(*, path: Path, rows: list[np.ndarray], labels: list[int]) -> None:
    header = [f'@relation {CLASSIFICATION_NAME}', '']
    header += [f'@attribute att{index + 1} numeric' for index in range(_LENGTH)]
    header += ['@attribute target {0,1}', '', '@data']
    lines = [
        ','.join([*(f'{value:.7g}' for value in row), str(label)])
        for row, label in zip(rows, labels, strict=True)
    ]
    path.write_text('\n'.join([*header, *lines]) + '\n', encoding='utf-8')


def write_classification(*, directory: Path) -> None:
    """Write ``<directory>/ToySpectra/ToySpectra_{TRAIN,TEST}.arff``."""
    rng = np.random.default_rng(seed=0)
    folder = directory / CLASSIFICATION_NAME
    folder.mkdir(parents=True, exist_ok=True)
    for part, counts in (('TRAIN', _TRAIN_COUNTS), ('TEST', _TEST_COUNTS)):
        labels = [label for label, count in enumerate(counts) for _ in range(count)]
        rows = [_spectrum(rng=rng, label=label) for label in labels]
        _write_arff(path=folder / f'{CLASSIFICATION_NAME}_{part}.arff', rows=rows, labels=labels)


def _ar1(*, rng: np.random.Generator, size: int, coefficient: float) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=1.0, size=size)
    values = np.empty(size)
    values[0] = noise[0]
    for index in range(1, size):
        values[index] = coefficient * values[index - 1] + noise[index]
    return values * np.sqrt(1.0 - coefficient**2)


def write_forecasting(*, directory: Path) -> None:
    """Write ``<directory>/toy_hourly.csv.gz``."""
    rng = np.random.default_rng(seed=1)
    hours = np.arange(_HOURS)
    daily = np.sin(2 * np.pi * hours / 24)
    weekly = np.sin(2 * np.pi * hours / (24 * 7))
    seasonal = np.sin(2 * np.pi * hours / (24 * 365))
    columns: dict[str, np.ndarray] = {}
    for name, (level, scale) in _COLUMNS.items():
        phase = rng.uniform(low=0.0, high=2 * np.pi)
        shape = (
            0.5 * np.sin(2 * np.pi * hours / 24 + phase)
            + 0.3 * weekly
            + 0.3 * seasonal
            + 0.4 * _ar1(rng=rng, size=_HOURS, coefficient=0.9)
        )
        columns[name] = level + scale * shape
    load = np.mean([columns[name] for name in _COLUMNS], axis=0)
    lagged = np.concatenate([np.full(6, load[0]), load[:-6]])
    level, scale = _OIL_TEMPERATURE
    oil = 0.4 * (lagged - load.mean()) / load.std() + 0.5 * seasonal + 0.2 * daily
    columns['OT'] = level + scale * (oil + 0.2 * _ar1(rng=rng, size=_HOURS, coefficient=0.95))
    dates = _START + hours.astype('timedelta64[h]')
    header = ','.join(['date', *columns])
    lines = [
        ','.join(
            [np.datetime_as_string(dates[row], unit='s').replace('T', ' ')]
            + [f'{columns[name][row]:.3f}' for name in columns]
        )
        for row in range(_HOURS)
    ]
    text = '\n'.join([header, *lines]) + '\n'
    with gzip.GzipFile(filename=directory / FORECASTING_FILE, mode='wb', mtime=0) as file:
        file.write(text.encode('utf-8'))


if __name__ == '__main__':
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_classification(directory=DATA_DIRECTORY)
    write_forecasting(directory=DATA_DIRECTORY)
