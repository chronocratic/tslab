"""Torch and Lightning on top of the lab: training stage, trainer interfaces, trackers.

Importing this package registers torch seeding and the torch metadata with the lab, but
does not import torch; the names below load their module on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from tslab.lab.core.meta.run import register_hardware, register_versions
from tslab.lab.core.seeding import register_seeder
from tslab.ml.torch_environment import seed_torch, torch_hardware, torch_versions

if TYPE_CHECKING:
    from tslab.ml.artifacts import Checkpoint, Tensors
    from tslab.ml.interfaces import Batch, DataSource, LabeledSplits, Metric, SPLITS, Trainer
    from tslab.ml.split import Split
    from tslab.ml.stages import train
    from tslab.ml.tracking.wandb_mode import WandbMode

register_seeder(seed_torch, name='torch')
register_versions(torch_versions)
register_hardware(torch_hardware)

# why: tslab wandb-sync imports a submodule of this package and must not load torch
_LAZY = {
    'Batch': 'tslab.ml.interfaces',
    'Checkpoint': 'tslab.ml.artifacts',
    'DataSource': 'tslab.ml.interfaces',
    'LabeledSplits': 'tslab.ml.interfaces',
    'Metric': 'tslab.ml.interfaces',
    'SPLITS': 'tslab.ml.interfaces',
    'Split': 'tslab.ml.split',
    'Tensors': 'tslab.ml.artifacts',
    'Trainer': 'tslab.ml.interfaces',
    'WandbMode': 'tslab.ml.tracking.wandb_mode',
    'train': 'tslab.ml.stages',
}

__all__ = [
    'SPLITS',
    'Batch',
    'Checkpoint',
    'DataSource',
    'LabeledSplits',
    'Metric',
    'Split',
    'Tensors',
    'Trainer',
    'WandbMode',
    'train',
]


def __getattr__(name: str) -> object:
    if name not in _LAZY:
        return _submodule(name=name)
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def _submodule(*, name: str) -> object:
    """A submodule, as after an eager import; a missing dependency of it is not hidden."""
    try:
        return import_module(f'{__name__}.{name}')
    except ModuleNotFoundError as error:
        if error.name != f'{__name__}.{name}':
            raise
        message = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(message) from None


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
