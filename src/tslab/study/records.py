"""The ``submissions.jsonl`` record of a study: one line per submitted job."""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from tslab.core.paths import STATUS_FILE
from tslab.core.status import InstanceState, read_status

__all__ = [
    'SUBMISSIONS_FORMAT_VERSION',
    'JobRecord',
    'ProgressSnapshot',
    'TaskRecord',
    'append_job',
    'latest_tasks',
    'read_jobs',
    'submission_lock',
    'take_progress',
]

SUBMISSIONS_FORMAT_VERSION = 1

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ProgressSnapshot:
    """How far a run had gotten when its job was submitted.

    Args:
        done_instances: Instances in state ``done`` in ``status.json``; 0 when
            the file is missing or unreadable.
        newest_checkpoint_mtime: The newest ``last.ckpt`` below the run
            folder, or ``None``.
    """

    done_instances: int
    newest_checkpoint_mtime: float | None

    def to_json(self) -> dict[str, object]:
        """The JSON form of the snapshot."""
        return {
            'done_instances': self.done_instances,
            'newest_checkpoint_mtime': self.newest_checkpoint_mtime,
        }

    @classmethod
    def from_json(cls, *, document: Mapping[str, Any]) -> 'ProgressSnapshot':
        """A snapshot from its JSON form."""
        mtime = document.get('newest_checkpoint_mtime')
        return cls(
            done_instances=int(document['done_instances']),
            newest_checkpoint_mtime=None if mtime is None else float(mtime),
        )


def take_progress(*, run_directory: Path) -> ProgressSnapshot:
    """The progress of one run, for the record of its next submission.

    Args:
        run_directory: The run folder, e.g. ``<study dir>/<run path>``.

    Returns:
        The snapshot; an unreadable or missing folder counts as no progress.
    """
    try:
        snapshot = read_status(path=run_directory / STATUS_FILE)
    except (TypeError, ValueError, OSError):
        snapshot = None
    done = 0
    if snapshot is not None:
        done = sum(
            1 for record in snapshot.instances.values() if record.state is InstanceState.DONE
        )
    checkpoints = [
        path.stat().st_mtime for path in run_directory.rglob('last.ckpt') if path.is_file()
    ]
    return ProgressSnapshot(
        done_instances=done, newest_checkpoint_mtime=max(checkpoints) if checkpoints else None
    )


@dataclass(frozen=True, kw_only=True)
class TaskRecord:
    """One run in one submitted job.

    Args:
        index: The array index of the run, the same in every attempt.
        grid: The grid the run belongs to.
        run_path: The run path within the study.
        command: The full command of the run.
        extra_args: Pass-through arguments after the flags.
        resources: ``dataclasses.asdict`` of the SLURM settings, or ``{}``
            locally.
        attempt: The attempt number of this submission.
        progress: How far the run had gotten before this submission.
    """

    index: int
    grid: str
    run_path: str
    command: tuple[str, ...]
    extra_args: tuple[str, ...]
    resources: Mapping[str, object]
    attempt: int
    progress: ProgressSnapshot

    def to_json(self) -> dict[str, object]:
        """The JSON form of the record."""
        return {
            'index': self.index,
            'grid': self.grid,
            'run_path': self.run_path,
            'command': list(self.command),
            'extra_args': list(self.extra_args),
            'resources': dict(self.resources),
            'attempt': self.attempt,
            'progress': self.progress.to_json(),
        }

    @classmethod
    def from_json(cls, *, document: Mapping[str, Any]) -> 'TaskRecord':
        """A record from its JSON form."""
        return cls(
            index=int(document['index']),
            grid=str(document['grid']),
            run_path=str(document['run_path']),
            command=tuple(str(item) for item in document['command']),
            extra_args=tuple(str(item) for item in document['extra_args']),
            resources=dict(document['resources']),
            attempt=int(document['attempt']),
            progress=ProgressSnapshot.from_json(document=document['progress']),
        )


@dataclass(frozen=True, kw_only=True)
class JobRecord:
    """One submitted job: a group of run tasks or a follow-up job.

    Args:
        kind: ``'run'`` or ``'followup'``.
        submission_id: One id per ``launch`` call.
        time: The submission time, UTC ISO.
        launcher: The launcher that submitted the job.
        job_id: The scheduler job id, or the local id.
        job_file: The number of the ``jobs/<n>`` files of this job.
        tasks: The run tasks; empty for a follow-up.
    """

    kind: Literal['run', 'followup']
    submission_id: str
    time: str
    launcher: Literal['slurm', 'local']
    job_id: str
    job_file: int
    tasks: tuple[TaskRecord, ...]

    def to_json(self) -> dict[str, object]:
        """The JSON form of the record, with the format version."""
        return {
            'format_version': SUBMISSIONS_FORMAT_VERSION,
            'kind': self.kind,
            'submission_id': self.submission_id,
            'time': self.time,
            'launcher': self.launcher,
            'job_id': self.job_id,
            'job_file': self.job_file,
            'tasks': [task.to_json() for task in self.tasks],
        }

    @classmethod
    def from_json(cls, *, document: Mapping[str, Any]) -> 'JobRecord':
        """A record from its JSON form; the format version must be known."""
        version = document.get('format_version')
        if version != SUBMISSIONS_FORMAT_VERSION:
            message = f'format_version {version!r} is not {SUBMISSIONS_FORMAT_VERSION}'
            raise ValueError(message)
        return cls(
            kind=document['kind'],
            submission_id=str(document['submission_id']),
            time=str(document['time']),
            launcher=document['launcher'],
            job_id=str(document['job_id']),
            job_file=int(document['job_file']),
            tasks=tuple(TaskRecord.from_json(document=item) for item in document['tasks']),
        )


@contextmanager
def submission_lock(*, study_folder: Path) -> Generator[None]:
    """Hold the submission lock of a study, so two submissions never interleave.

    Planning a job reads the records and the job files and ends with a
    new record; the lock makes that sequence one step across processes
    (a person and the follow-up job of the cluster, for example).

    Args:
        study_folder: The ``.study`` folder of the study; created if missing.
    """
    study_folder.mkdir(parents=True, exist_ok=True)
    with (study_folder / 'submit.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_job(*, path: Path, record: JobRecord) -> None:
    """Append one record as one line, flushed to disk.

    The line goes out in a single write, so concurrent appenders do not
    interleave inside a line.

    Args:
        path: The ``submissions.jsonl`` file; parent folders are created.
        record: The job to record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_json(), sort_keys=True) + '\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, line.encode('utf-8'))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jobs(*, path: Path) -> list[JobRecord]:
    """The job records of a file; a torn or malformed line is skipped, not fatal.

    Args:
        path: The ``submissions.jsonl`` file; a missing file means no jobs.

    Returns:
        The records, in file order.

    Raises:
        ValueError: If a line holds an unknown ``format_version``.
    """
    if not path.is_file():
        return []
    records: list[JobRecord] = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as error:
            _LOGGER.warning('%s: skipping line %d: %s', path, line_number, error)
            continue
        version = document.get('format_version') if isinstance(document, dict) else None
        if version != SUBMISSIONS_FORMAT_VERSION:
            message = (
                f'{path}: line {line_number}: format_version {version!r} '
                f'is not {SUBMISSIONS_FORMAT_VERSION}'
            )
            raise ValueError(message)
        try:
            records.append(JobRecord.from_json(document=document))
        except (ValueError, KeyError, TypeError) as error:
            _LOGGER.warning('%s: skipping line %d: %s', path, line_number, error)
    return records


def latest_tasks(*, jobs: Sequence[JobRecord]) -> dict[str, tuple[JobRecord, TaskRecord]]:
    """The last submitted task of every run path.

    Follow-up jobs hold no run tasks and never win.

    Args:
        jobs: The records, in submission order.

    Returns:
        ``{run path: (job, task)}`` with the latest job per path.
    """
    latest: dict[str, tuple[JobRecord, TaskRecord]] = {}
    for job in jobs:
        if job.kind != 'run':
            continue
        for task in job.tasks:
            latest[task.run_path] = (job, task)
    return latest
