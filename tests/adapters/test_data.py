import dataclasses
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import chronocratic.datasets
import pytest
from synthetic_data import CLASSIFICATION_NAME, DATA_DIRECTORY, FORECASTING_FILE
import torch

import tslab
from tslab.adapters.chronocratic_datasets import (
    build_chronocratic_data_source,
    check_label_range,
    ChronocraticClassificationData,
    ChronocraticDataChoice,
    ChronocraticForecastingData,
    data_settings,
    DATAMODULES,
    EXCLUDED_PARAMETERS,
)
from tslab.core.fingerprint import canonicalize
from tslab.interfaces import SPLITS
from tslab.lightning import LightningDataSource

pytestmark = pytest.mark.chronocratic

CLASSIFICATION_ARGS = (
    'data:ucr',
    '--data.dataset-folder-path',
    str(DATA_DIRECTORY / CLASSIFICATION_NAME),
    '--data.target-column-name',
    'target',
)
FORECASTING_ARGS = (
    'data:ett',
    '--data.dataset-file-path',
    str(DATA_DIRECTORY / FORECASTING_FILE),
    '--data.variant',
    'ETTh1',
)


@dataclass(frozen=True)
class Config:
    data: ChronocraticDataChoice


def parse(*args: str) -> Config:
    return tslab.parse_config(config_class=Config, args=list(args))


def classification() -> ChronocraticClassificationData:
    data = build_chronocratic_data_source(config=parse(*CLASSIFICATION_ARGS).data)
    assert isinstance(data, ChronocraticClassificationData)
    return data


def forecasting() -> ChronocraticForecastingData:
    data = build_chronocratic_data_source(config=parse(*FORECASTING_ARGS).data)
    assert isinstance(data, ChronocraticForecastingData)
    return data


def test_every_datamodule_is_discovered() -> None:
    assert {'ucr', 'uea', 'ett', 'electricity', 'weather'} <= set(DATAMODULES)
    for entry in DATAMODULES.values():
        assert entry.datamodule_class.__name__ in chronocratic.datasets.__all__


def test_settings_are_the_constructor_without_loader_parameters() -> None:
    for name, entry in DATAMODULES.items():
        parameters = set(inspect.signature(entry.datamodule_class.__init__).parameters) - {'self'}
        fields = {field.name for field in dataclasses.fields(entry.settings_class)}
        assert fields == parameters - EXCLUDED_PARAMETERS, name


def test_only_result_changing_settings_are_fingerprinted() -> None:
    settings: Any = parse(*CLASSIFICATION_ARGS).data
    base = canonicalize(value=settings)
    assert canonicalize(value=dataclasses.replace(settings, num_workers=4)) == base
    assert canonicalize(value=dataclasses.replace(settings, batch_size=8)) != base
    assert (
        canonicalize(value=dataclasses.replace(settings, dataset_folder_path=Path('elsewhere')))
        != base
    )


def test_data_settings_round_trip_through_a_config_file(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    for args in (CLASSIFICATION_ARGS, FORECASTING_ARGS):
        config = parse(*args, '--data.batch-size', '16')
        tslab.save_config(config=config, path=path)
        assert tslab.load_config(config_class=Config, path=path) == config


def test_classification_info() -> None:
    info = classification().info
    assert info.num_channels == 1
    assert info.length == 286
    assert info.num_classes == 2


def test_labeled_splits_cover_every_sample_once() -> None:
    data = classification()
    splits = {split: data.labeled_split(split=split) for split in SPLITS}
    samples = torch.cat([inputs for inputs, _ in splits.values()])
    labels = torch.cat([targets for _, targets in splits.values()])
    assert samples.shape == (56, 286, 1)
    assert torch.unique(samples.flatten(start_dim=1), dim=0).shape[0] == 56
    assert set(labels.tolist()) == {0, 1}
    assert splits['test'][0].shape[0] == 28


def test_label_range_check() -> None:
    check_label_range(labels=torch.tensor([0, 1, 1]), num_classes=2, split='train')
    with pytest.raises(ValueError, match=r'outside \[0, 2\)'):
        check_label_range(labels=torch.tensor([1, 2]), num_classes=2, split='test')


def test_lightning_receives_the_datamodule() -> None:
    data = classification()
    assert isinstance(data, LightningDataSource)
    assert isinstance(data.datamodule, chronocratic.datasets.UCRClassificationDataModule)


def test_setup_again_gives_the_same_data() -> None:
    data = classification()
    before = data.labeled_split(split='train')
    data.datamodule.setup(stage='fit')
    after = data.labeled_split(split='train')
    torch.testing.assert_close(actual=after, expected=before)
    series = forecasting()
    first = series.full_series()
    series.datamodule.setup(stage='fit')
    torch.testing.assert_close(actual=series.full_series(), expected=first)


def test_forecasting_channels_and_splits() -> None:
    data = forecasting()
    info = data.info
    time_features = len(info.covariate_channels)
    assert time_features > 0
    assert info.covariate_channels == tuple(range(time_features))
    assert info.target_channels == (time_features,)
    assert info.num_channels == time_features + 1
    assert info.length is None
    assert data.full_series().shape == (1, 14_400, info.num_channels)
    assert data.full_series().dtype == torch.float32
    slices = data.split_slices()
    assert slices['train'].stop == 8_640
    assert (slices['val'].start, slices['val'].stop) == (8_640, 11_520)
    assert (slices['test'].start, slices['test'].stop) == (11_520, 14_400)


def test_foreign_settings_are_rejected() -> None:
    with pytest.raises(TypeError, match='ChronocraticDataChoice'):
        build_chronocratic_data_source(config=Config)


def test_data_settings_builds_a_member_of_the_choice() -> None:
    path = DATA_DIRECTORY / FORECASTING_FILE
    settings = data_settings(name='ett', dataset_file_path=path, variant='ETTh1', batch_size=1)
    assert type(settings) is DATAMODULES['ett'].settings_class
    assert settings.batch_size == 1
    with pytest.raises(ValueError, match="no dataset 'etth'"):
        data_settings(name='etth')
    with pytest.raises(ValueError, match="has no settings \\['seq_len'\\]"):
        data_settings(name='ett', dataset_file_path=path, variant='ETTh1', seq_len=128)
