"""Abstract interfaces for time series: encoders and downstream models."""

from tslab.ts.interfaces.downstream import DownstreamModel, Probe
from tslab.ts.interfaces.encoder import Encoder, SequenceEncoder

__all__ = ['DownstreamModel', 'Encoder', 'Probe', 'SequenceEncoder']
