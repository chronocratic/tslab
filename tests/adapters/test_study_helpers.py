"""``ucr_datasets`` and ``uea_datasets``: dataset axes for the study layer."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from tslab.adapters.chronocratic.datasets import ucr_datasets, uea_datasets
from tslab.adapters.chronocratic.datasets.settings import ChronocraticDataChoice
from tslab.study import check_paths, Settings

pytestmark = pytest.mark.chronocratic


@dataclass(frozen=True, kw_only=True)
class Config:
    """The smallest config the helpers' settings must type-check against."""

    data: ChronocraticDataChoice
    seed: int = 0


def test_helpers_build_settings_the_real_classes_accept() -> None:
    axis = ucr_datasets(root=Path('${UCR_ROOT}'), names=('Coffee',)) | uea_datasets(
        root=Path('/d/UEA'), names=('BasicMotions',), settings={'data.batch_size': 2}
    )
    assert axis['Coffee'] == {
        'data': 'ucr',
        'data.dataset_folder_path': '${UCR_ROOT}/Coffee',
        'data.target_column_name': 'target',
    }
    assert axis['BasicMotions']['data.batch_size'] == 2
    for settings in axis.values():
        assert check_paths(config_class=Config, settings=cast('Settings', settings)) == []


def test_names_are_required() -> None:
    with pytest.raises(ValueError, match='dataset name'):
        ucr_datasets(root=Path('/d'), names=())


def test_a_string_root_is_rejected() -> None:
    with pytest.raises(TypeError, match=r'root must be pathlib.Path'):
        ucr_datasets(root='/d', names=('Coffee',))  # ty: ignore[invalid-argument-type]
