"""Abstract interfaces for torch training: data, trainer and metric."""

from tslab.ml.interfaces.data import Batch, DataSource, LabeledSplits, SPLITS
from tslab.ml.interfaces.metric import Metric
from tslab.ml.interfaces.trainer import Trainer

__all__ = ['SPLITS', 'Batch', 'DataSource', 'LabeledSplits', 'Metric', 'Trainer']
