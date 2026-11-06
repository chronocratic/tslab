"""The launch loop: submit each group and record its job before its tasks run."""

from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from uuid import uuid4

from tslab.study.definition import Study
from tslab.study.launchers.common.protocols import Launcher, WriteJobFiles
from tslab.study.launchers.common.types import LaunchError, LaunchReport, LaunchStopped
from tslab.study.machine import MachineProfile
from tslab.study.paths import study_folder
from tslab.study.plan import JobGroup
from tslab.study.records import append_job, JobRecord, take_progress, TaskRecord

__all__ = ['launch']


def launch(  # noqa: PLR0913 # why: keyword-only public API; each launch input is explicit
    *,
    launcher: Launcher,
    record_type: type[JobRecord],
    write_job_files: WriteJobFiles,
    machine: MachineProfile,
    study: Study,
    groups: Sequence[JobGroup],
    project_root: Path,
    dry_run: bool,
    now: Callable[[], str],
    start_delay_seconds: int | None = None,
) -> LaunchReport:
    """Write backend files, then submit groups one by one.

    A dry run writes files and stops. Otherwise each group is recorded as soon
    as its job exists, so a failed launch keeps records for submitted groups.

    Args:
        launcher: The backend's implementation that submits one group.
        record_type: The job record class of the backend.
        write_job_files: The backend's file writer.
        machine: The machine the jobs run on.
        study: The study the groups came from.
        groups: Planned groups in submission order.
        project_root: The directory the jobs run in.
        dry_run: Write files and stop without submitting.
        now: Return the current UTC ISO timestamp for each submitted group.
        start_delay_seconds: Seconds the jobs wait before they may start, when set.

    Returns:
        What was submitted, what was not, and any error or stopping signal.
    """
    folder = study_folder(machine=machine, study=study)
    write_job_files(
        study_folder=folder,
        groups=groups,
        machine=machine,
        project_root=project_root,
        start_delay_seconds=start_delay_seconds,
    )
    if dry_run:
        return LaunchReport(submitted=(), not_submitted=tuple(groups), error=None)
    submission_id = uuid4().hex
    study_directory = machine.output_root / study.name
    records: list[JobRecord] = []
    upstream: dict[PurePosixPath, str] = {}
    error: str | None = None
    stopped_signal: int | None = None
    for group in groups:
        tasks = tuple(
            TaskRecord(
                index=task.index,
                grid=task.resolved.run.grid,
                run_path=task.resolved.run.run_path,
                command=task.command,
                extra_args=task.extra_args,
                resources=task.resolved.resources.as_record(),
                attempt=task.attempt,
                progress=take_progress(run_directory=study_directory / task.resolved.run.run_path),
                uncounted_attempts=task.uncounted_attempts,
                uncounted_in_row=task.uncounted_in_row,
            )
            for task in group.tasks
        )

        def record_job(
            *, job_id: str, group: JobGroup = group, tasks: tuple[TaskRecord, ...] = tasks
        ) -> None:
            record = record_type.for_group(
                group=group,
                submission_id=submission_id,
                time=now(),
                launcher=machine.launcher,
                job_id=job_id,
                tasks=tasks,
            )
            records.append(record)
            append_job(path=folder / 'submissions.jsonl', record=record)
            upstream.update({task.run_path: f'{job_id}_{task.index}' for task in tasks})

        try:
            launcher.submit(group=group, study_folder=folder, record=record_job, upstream=upstream)
        except LaunchError as exc:
            error = str(exc)
            break
        except LaunchStopped as exc:
            # why: the record of this group is kept; resubmit continues the stopped runs
            stopped_signal = exc.signal_number
            break
    return LaunchReport(
        submitted=tuple(records),
        not_submitted=tuple(groups[len(records) :]),
        error=error,
        stopped_signal=stopped_signal,
    )
