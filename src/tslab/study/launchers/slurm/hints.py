"""Failure hints for the scheduler states of a SLURM job."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from tslab.jobs.slurm import format_slurm_memory
from tslab.study.launchers import JobUsage, Resources

__all__ = ['state_hint']

_TIMEOUT_HINT = (
    'raise time, or check that the timeout signal is on (signal_seconds), '
    'or lower job_time_share so the stages after training fit'
)
_NOT_THE_CODE_HINT = 'not caused by the code; resubmit'


def _out_of_memory(*, usage: JobUsage | None, resources: Resources) -> str:
    requested = resources.requested_memory
    if usage is None or usage.max_rss is None or requested is None:
        return 'raise mem_per_cpu'
    return (
        'raise mem_per_cpu; peak '
        f'{format_slurm_memory(size=usage.max_rss)} '
        f'of {format_slurm_memory(size=requested)} requested'
    )


def _timeout(*, usage: JobUsage | None, resources: Resources) -> str:  # noqa: ARG001 # why: the hint signature
    return _TIMEOUT_HINT


def _not_the_code(*, usage: JobUsage | None, resources: Resources) -> str:  # noqa: ARG001 # why: the hint signature
    return _NOT_THE_CODE_HINT


_HINTS: Mapping[str, Callable[..., str]] = MappingProxyType(
    {
        'OUT_OF_MEMORY': _out_of_memory,
        'TIMEOUT': _timeout,
        'PREEMPTED': _not_the_code,
        'NODE_FAIL': _not_the_code,
    }
)


def state_hint(
    *, job_state: str | None, usage: JobUsage | None, resources: Resources
) -> str | None:
    """The hint for the scheduler state of a job, or ``None`` when the state has none.

    Args:
        job_state: The final SLURM state of the job.
        usage: The accounting usage of the job, when queried.
        resources: The resources the job requested.

    Returns:
        The hint text, or ``None``.
    """
    hint = _HINTS.get(job_state or '')
    return None if hint is None else hint(usage=usage, resources=resources)
