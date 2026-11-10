"""Toy dataset axes shared by the template studies."""

from collections.abc import Mapping
from pathlib import Path

from tslab.lab.study import Settings
from tslab.ts.adapters.chronocratic.datasets import ucr_datasets

__all__ = ['TOY_ETT', 'TOY_UCR']

TOY_UCR: Mapping[str, Settings] = ucr_datasets(root=Path('data'), names=('ToySpectra',))
TOY_ETT: Mapping[str, Settings] = {
    'ETTh1': {
        'data': 'ett',
        'data.dataset_file_path': 'data/toy_hourly.csv.gz',
        'data.variant': 'ETTh1',
    }
}
