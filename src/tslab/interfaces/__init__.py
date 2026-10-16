"""Abstract interfaces that TSLab core talks to.

ABCs are used where TSLab provides shared behaviour users inherit; Protocols
where existing objects should fit without inheriting.
"""

from tslab.interfaces.tracker import RunStatus, Tracker

__all__ = ['RunStatus', 'Tracker']
