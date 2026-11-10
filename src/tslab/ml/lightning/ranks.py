"""Share stage cache decisions between tasks on one SLURM node."""

from collections.abc import Mapping
from datetime import timedelta

from torch._C._distributed_c10d import TCPStore

__all__ = ['agree']

_STORE: TCPStore | None = None


def agree(*, key: str, value: bool, environment: Mapping[str, str]) -> bool:
    """Share the rank-zero decision using a TCP store, blocking until it is available.

    Args:
        key: Unique stage decision key.
        value: The decision of this rank; rank zero decides for all ranks.
        environment: SLURM job and process variables.

    Returns:
        The value chosen by rank zero.
    """
    global _STORE  # noqa: PLW0603 # why: one persistent store per process for all stage decisions
    rank = int(environment['SLURM_PROCID'])
    if _STORE is None:
        # why: job ids separated by 10000 can collide on the same node; revisit the port if needed
        _STORE = TCPStore(
            host_name='127.0.0.1',
            port=25000 + int(environment['SLURM_JOB_ID']) % 10000,
            world_size=int(environment['SLURM_NTASKS']),
            is_master=rank == 0,
            timeout=timedelta(minutes=10),
        )
    if rank == 0:
        _STORE.set(key, '1' if value else '0')
        return value
    return _STORE.get(key) == b'1'
