"""Ready-made time series stages: ``encode`` and ``evaluate``."""

from tslab.ts.stages.encode import check_representations, encode
from tslab.ts.stages.encoders import ModelAsEncoder
from tslab.ts.stages.evaluate import evaluate

__all__ = ['ModelAsEncoder', 'check_representations', 'encode', 'evaluate']
