"""Abstract interfaces that TSLab core talks to.

ABCs are used where TSLab provides shared behaviour users inherit; Protocols
where existing objects should fit without inheriting.
"""

from tslab.interfaces.data import Batch, DataSource
from tslab.interfaces.tracker import RunStatus, Tracker
from tslab.interfaces.trainer import Trainer

__all__ = ['Batch', 'DataSource', 'RunStatus', 'Tracker', 'Trainer']
