"""Training deadline from the SLURM job end time.

SLURM exports the end of the job as a Unix timestamp in ``SLURM_JOB_END_TIME``;
training gets a fraction of the time left, so the rest of the job can still run the
stages after training and write the checkpoint.
"""

from collections.abc import Mapping

from tslab.jobs.slurm import remaining_job_seconds

__all__ = ['job_training_deadline']


def job_training_deadline(
    *, share: float, environment: Mapping[str, str], now: float
) -> float | None:
    """Return the deadline for training in the current SLURM job.

    Args:
        share: Fraction of the remaining SLURM job time used for training.
        environment: Environment variables, usually ``os.environ``.
        now: Current Unix time, usually ``time.time()``.

    Returns:
        ``now`` plus ``share`` of the remaining job time, or ``None`` outside SLURM.

    Raises:
        ValueError: If ``share`` is not in ``(0, 1]``.
    """
    if not 0 < share <= 1:
        message = f'job_time_share must be in (0, 1], got {share}'
        raise ValueError(message)
    remaining = remaining_job_seconds(environment=environment, now=now)
    if remaining is None:
        return None
    return now + share * remaining
