"""Experiment tracking: options, the ready trackers and ``build_trackers``.

This module stays importable without Lightning or any tracker SDK: the
concrete trackers are imported inside ``build_trackers``.
"""

from tslab.ml.tracking.factory import build_trackers
from tslab.ml.tracking.options import TrackingOptions

__all__ = ['TrackingOptions', 'build_trackers']
