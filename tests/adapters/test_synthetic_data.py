import gzip
from pathlib import Path

import numpy as np
import pytest
from synthetic_data import (
    CLASSIFICATION_NAME,
    DATA_DIRECTORY,
    FORECASTING_FILE,
    write_classification,
    write_forecasting,
)

pytestmark = pytest.mark.chronocratic


def arff_values(*, path: Path) -> np.ndarray:
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = lines[lines.index('@data') + 1 :]
    return np.array([[float(value) for value in row.split(',')] for row in rows])


def csv_values(*, path: Path) -> tuple[list[str], np.ndarray]:
    with gzip.open(filename=path, mode='rt', encoding='utf-8') as file:
        lines = file.read().splitlines()
    dates = [line.split(',', maxsplit=1)[0] for line in lines[1:]]
    values = np.array([[float(value) for value in line.split(',')[1:]] for line in lines[1:]])
    return dates, values


def test_committed_files_match_the_generator(tmp_path: Path) -> None:
    write_classification(directory=tmp_path)
    write_forecasting(directory=tmp_path)
    for part in ('TRAIN', 'TEST'):
        name = f'{CLASSIFICATION_NAME}/{CLASSIFICATION_NAME}_{part}.arff'
        np.testing.assert_allclose(
            actual=arff_values(path=tmp_path / name),
            desired=arff_values(path=DATA_DIRECTORY / name),
            atol=1e-6,
        )
    fresh_dates, fresh = csv_values(path=tmp_path / FORECASTING_FILE)
    committed_dates, committed = csv_values(path=DATA_DIRECTORY / FORECASTING_FILE)
    assert fresh_dates == committed_dates
    np.testing.assert_allclose(actual=fresh, desired=committed, atol=1e-3)


def test_classification_file_has_the_promised_shape() -> None:
    train = arff_values(
        path=DATA_DIRECTORY / CLASSIFICATION_NAME / f'{CLASSIFICATION_NAME}_TRAIN.arff'
    )
    assert train.shape == (28, 287)
    assert sorted(np.unique(train[:, -1]).tolist()) == [0.0, 1.0]
    np.testing.assert_allclose(actual=train[:, :-1].mean(axis=1), desired=0.0, atol=1e-5)


def test_forecasting_file_covers_the_hourly_splits() -> None:
    dates, values = csv_values(path=DATA_DIRECTORY / FORECASTING_FILE)
    assert values.shape == (14_400, 7)
    assert dates[0] == '2016-07-01 00:00:00'
    assert dates[1] == '2016-07-01 01:00:00'
    assert np.isfinite(values).all()
