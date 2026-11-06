"""The helper jobs queued behind submitted runs: the automatic retry and the W&B sync."""

from collections.abc import Callable, Sequence
import dataclasses
import logging
from pathlib import Path
import shlex
import time

from tslab.jobs.slurm import SlurmError
from tslab.study.context import Context
from tslab.study.definition import followup_job_name, RETRY_JOB, WANDB_SYNC_JOB
from tslab.study.launchers.slurm.submit import submit_helper_job
from tslab.study.machine import SlurmSettings
from tslab.study.machine.profile import wants_retry_followup
from tslab.study.paths import study_folder
from tslab.study.records import JobRecord, submission_lock

__all__ = ['queue_followups']

_LOGGER = logging.getLogger(__name__)


def queue_followups(
    *,
    context: Context,
    job_ids: Sequence[str],
    now: Callable[[], str],
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, list[JobRecord]]:
    """The small helper jobs of the study, after the runs it was called for.

    With ``followup.retry`` or ``followup.chained_retry``: one job resubmits interrupted
    runs with ``--auto``, so the retry chain continues on the cluster
    without a person at the keyboard. With ``followup.wandb_sync``: one job
    that syncs the run folders of the study to W&B. Each waits for ``job_ids``
    with ``afterany``. The retry job needs the study reference; a context
    that has none logs a warning and skips it.

    Args:
        context: The context the runs were submitted under.
        job_ids: The job ids the helper jobs wait for; empty does nothing.
        now: Return the current UTC ISO timestamp for each record.
        sleep: What waits out the submission pause of the cluster.

    Returns:
        ``1`` when a helper job could not be submitted (the later ones are
        skipped), else ``0``; and the submitted helper jobs.
    """
    machine = context.machine
    followup = machine.followup
    if (
        machine.slurm is None
        or not job_ids
        or not (wants_retry_followup(followup=followup) or followup.wandb_sync)
    ):
        return 0, []
    dependency = 'afterany:' + ':'.join(job_ids)
    slurm = dataclasses.replace(
        machine.slurm,
        time=followup.time,
        cpus_per_task=1,
        gpus=0,
        mem_per_cpu=None,
        signal_seconds=0,
    )
    folder = study_folder(machine=machine, study=context.study)
    with submission_lock(study_folder=folder):
        records, failure = _submit_followups(
            context=context, folder=folder, slurm=slurm, dependency=dependency, now=now, sleep=sleep
        )
    # why: logged here, not from inside the except (TRY400)
    if failure is not None:
        _LOGGER.error('%s', failure)
    return (0 if failure is None else 1), records


def _submit_followups(  # noqa: PLR0913 # why: one keyword per submission input
    *,
    context: Context,
    folder: Path,
    slurm: SlurmSettings,
    dependency: str,
    now: Callable[[], str],
    sleep: Callable[[float], None],
) -> tuple[list[JobRecord], str | None]:
    """Submit each helper job in order; the records, and why it stopped early, if it did."""
    records: list[JobRecord] = []
    for job_name, command in _followup_commands(context=context):
        try:
            record = submit_helper_job(
                study_folder=folder,
                job_name=job_name,
                command=command,
                slurm=slurm,
                machine=context.machine,
                project_root=context.project_root,
                dependency=dependency,
                now=now,
                sleep=sleep,
            )
        except SlurmError as error:
            return records, f'follow-up {job_name} not submitted: {error}'
        records.append(record)
        _LOGGER.info(
            'submitted follow-up %s as %s (waits for %s)', job_name, record.job_id, dependency
        )
    return records, None


def _retry_command(*, context: Context) -> tuple[str, str] | None:
    """The automatic retry follow-up, or ``None`` (with a warning) without a study reference."""
    if not context.reference:
        _LOGGER.warning('no study reference: the automatic retry follow-up is skipped')
        return None
    machine = context.machine
    command = shlex.join(
        [
            *machine.tslab_command,
            'study',
            'resubmit',
            context.reference,
            '--machine',
            context.machine_path.as_posix(),
            '--auto',
        ]
    )
    return followup_job_name(study=context.study.name, kind=RETRY_JOB), command


def _followup_commands(*, context: Context) -> list[tuple[str, str]]:
    machine = context.machine
    followup = machine.followup
    commands: list[tuple[str, str]] = []
    retry = _retry_command(context=context) if wants_retry_followup(followup=followup) else None
    if retry is not None:
        commands.append(retry)
    if followup.wandb_sync:
        commands.append(
            (
                followup_job_name(study=context.study.name, kind=WANDB_SYNC_JOB),
                shlex.join(
                    [*machine.tslab_command, 'wandb-sync', context.study_directory.as_posix()]
                ),
            )
        )
    return commands
