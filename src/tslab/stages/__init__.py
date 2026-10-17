"""Ready-made generic stages: ``train``, ``encode`` and ``evaluate``."""

from tslab.stages.encode import check_representations, encode
from tslab.stages.evaluate import evaluate
from tslab.stages.train import train

__all__ = ['check_representations', 'encode', 'evaluate', 'train']
