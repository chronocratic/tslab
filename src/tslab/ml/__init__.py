"""Torch and Lightning on top of the lab: training stage, trainer interfaces, trackers."""

from tslab.lab.core.meta.run import register_hardware, register_versions
from tslab.lab.core.seeding import register_seeder
from tslab.ml.artifacts import Checkpoint, Tensors
from tslab.ml.interfaces import Batch, DataSource, LabeledSplits, Metric, SPLITS, Trainer
from tslab.ml.split import Split
from tslab.ml.stages import train
from tslab.ml.torch_environment import seed_torch, torch_hardware, torch_versions
from tslab.ml.tracking.wandb_mode import WandbMode

register_seeder(seed_torch)
register_versions(torch_versions)
register_hardware(torch_hardware)

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
