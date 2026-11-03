"""Abstract interfaces that TSLab core talks to.

ABCs are used where TSLab provides shared behaviour users inherit; Protocols
where existing objects should fit without inheriting.
"""

from tslab.interfaces.data import Batch, DataSource, LabeledSplits, SPLITS
from tslab.interfaces.downstream import DownstreamModel, Probe
from tslab.interfaces.encoder import Encoder, SequenceEncoder
from tslab.interfaces.metric import Metric
from tslab.interfaces.trainer import Trainer

__all__ = [
    'SPLITS',
    'Batch',
    'DataSource',
    'DownstreamModel',
    'Encoder',
    'LabeledSplits',
    'Metric',
    'Probe',
    'SequenceEncoder',
    'Trainer',
]
