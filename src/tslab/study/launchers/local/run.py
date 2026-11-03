"""Running local job groups in parallel slots."""

from collections.abc import Callable, Mapping, Sequence
import logging
import os
from pathlib import Path
import shlex
import signal
import sys
import threading
from typing import TextIO

from tslab.core.files import atomic_write_text
from tslab.jobs.local import (
    busy_gpus,
    gpu_lock_folder,
    PoolCommand,
    run_commands,
    select_gpus,
    stop_on_signals,
    TaskInfo,
    TaskOutput,
    Watchdog,
    without_scheduler_variables,
)
from tslab.study.code_snapshot import SRC_FOLDER, without_code_snapshot
from tslab.study.launchers import LaunchError, LaunchStopped, OnTaskState, RecordJob
from tslab.study.launchers.local.pid_file import DETACHED_VARIABLE
from tslab.study.machine import LocalSettings, MachineProfile, timeout_seconds
from tslab.study.paths import usage_file
from tslab.study.plan import JobGroup, PlannedTask

__all__ = ['LocalLauncher', 'local_job_id', 'resolve_local_gpus', 'write_job_files']

_LOGGER = logging.getLogger(__name__)

#: The signals that stop a local launch; the records of the groups sent survive.
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def write_job_files(
    *,
    study_folder: Path,
    groups: Sequence[JobGroup],
    machine: MachineProfile,  # noqa: ARG001 # why: the shared writer signature
    project_root: Path,  # noqa: ARG001 # why: the shared writer signature
    start_delay_seconds: int | None = None,  # noqa: ARG001 # why: the shared writer signature
) -> None:
    """Write one task file per group before running local tasks."""
    (study_folder / 'jobs').mkdir(parents=True, exist_ok=True)
    (study_folder / 'logs').mkdir(parents=True, exist_ok=True)
    for group in groups:
        tasks_text = '\n'.join(
            f'{task.index} {task.attempt} {shlex.join(task.command)}' for task in group.tasks
        )
        atomic_write_text(
            path=study_folder / 'jobs' / f'{group.job_file}.tasks', text=tasks_text + '\n'
        )


def local_job_id(*, job_file: int, pid: int) -> str:
    """The job id of the local launch of one group file.

    Args:
        job_file: The group file number of the launch.
        pid: The process id of the launch.
    """
    return f'local-{pid}-{job_file}'


def resolve_local_gpus(*, local: LocalSettings, environment: Mapping[str, str]) -> tuple[str, ...]:
    """The GPU ids of a local launch, with ``"auto"`` resolved and the busy ones left out.

    Args:
        local: The ``[local]`` table: the ``gpus`` list, ``skip_busy_gpus`` and
            ``busy_gpu_memory_fraction``.
        environment: The environment to discover the GPUs from.

    Returns:
        The ids the launch may lease, in order; empty for CPU-only slots.

    Raises:
        ValueError: As ``select_gpus``.
    """
    gpus = 'auto' if local.gpus == 'auto' else tuple(str(gpu) for gpu in local.gpus)
    gpu_ids = select_gpus(
        gpus=gpus,
        skip_busy=local.skip_busy_gpus,
        environment=environment,
        memory_fraction=local.busy_gpu_memory_fraction,
    )
    if gpu_ids and local.max_parallel != 1:
        _LOGGER.warning('local.gpus is set, so local.max_parallel is ignored')
    if not gpu_ids and local.tasks_per_gpu > 1:
        _LOGGER.warning('no local GPUs, so local.tasks_per_gpu is ignored')
    return gpu_ids


class _LaunchStates:
    """The run pids of one launch, kept in step with the watchdog and the pid file.

    ``on_task_state`` is the callback the task output calls; ``on_spawn`` the
    callback the pool calls with the pid of a task that starts.
    """

    def __init__(self, *, on_task_state: OnTaskState | None, watchdog: Watchdog) -> None:
        self._on_task_state = on_task_state
        self._watchdog = watchdog
        self._pids: dict[str, int] = {}
        self._lock = threading.Lock()

    def _report(self, *, key: str, state: str, pid: int | None = None) -> None:
        if self._on_task_state is not None:
            self._on_task_state(task_id=key, state=state, pid=pid)

    def on_task_state(self, key: str, state: str) -> None:
        if state != 'ended':
            self._report(key=key, state=state)
            return
        with self._lock:
            pid = self._pids.pop(key, None)
        if pid is not None:
            self._watchdog.forget(pid=pid)
        self._report(key=key, state=state)

    def on_spawn(self, *, key: str, pid: int) -> None:
        with self._lock:
            self._pids[key] = pid
        self._watchdog.watch(pid=pid)
        self._report(key=key, state='running', pid=pid)


def _task_header(
    *, task: PlannedTask, job_id: str, host: str, now: Callable[[], str], gpu_ids: tuple[str, ...]
) -> str:
    """The header line of the log file of one task."""
    header = f'=== tslab attempt {task.attempt} | job {job_id} | {host} | {now()}'
    if gpu_ids:
        header += f' | gpus {",".join(gpu_ids)}'
    return header + ' ===\n'


class LocalLauncher:
    """The tasks of the group in parallel slots, under a per-task time limit.

    Without GPUs at most ``local.max_parallel`` tasks run at once; with GPUs
    one slot per (GPU, lane), the GPUs locked across launches in
    ``gpu_lock_folder``.
    """

    def __init__(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
        self,
        *,
        project_root: Path,
        env: Mapping[str, str],
        local: LocalSettings,
        pid: int,
        host: str,
        now: Callable[[], str],
        gpu_ids: tuple[str, ...] = (),
        stream: TextIO | None = None,
        on_task_state: OnTaskState | None = None,
        grace_seconds: float = 30.0,
    ) -> None:
        """Set up the launcher.

        Args:
            project_root: The directory the tasks run in.
            env: The ``[env]`` of the machine.
            local: The ``[local]`` table: time limit, signals, parallelism.
            pid: The process id of the launch.
            host: The host name of the launch.
            now: Return the current UTC ISO timestamp.
            gpu_ids: The resolved GPU ids the tasks lease; the caller
                resolves ``"auto"``. Empty for CPU-only slots.
            stream: Console output stream; standard output when omitted.
            on_task_state: Called with the task id and ``'running'`` or
                ``'ended'``; the task id is the job id and the task index.
            grace_seconds: Seconds to wait after a stop's ``SIGTERM`` before
                a ``SIGKILL``.
        """
        self._project_root = project_root
        self._env = env
        self._local = local
        self._pid = pid
        self._host = host
        self._now = now
        self._gpu_ids = gpu_ids
        self._stream = stream
        self._on_task_state = on_task_state
        self._grace_seconds = grace_seconds

    def submit(
        self,
        *,
        group: JobGroup,
        study_folder: Path,  # noqa: ARG002 # why: the Protocol signature
        record: RecordJob,
    ) -> str:
        """Run every task in slots, continuing after a failing one; the job id.

        The job is recorded before the first task starts. Each task appends
        to its log file (header, then the output line by line); with one
        slot the console gets the lines too, with more slots it gets one
        line per start and per end. After each task the launcher writes the
        usage file and a footer line to the log; a task that exits
        non-zero or cannot start does not stop the group.

        A SIGINT, SIGTERM or SIGHUP to this process stops the pool: running
        tasks get a SIGTERM, then a SIGKILL after ``grace_seconds``, and no
        task starts after the signal.

        Raises:
            LaunchError: If the code snapshot of the group has no ``src`` folder;
                the live code must not be run instead, and nothing is recorded.
            LaunchStopped: If a stop signal was caught, with its number.
        """
        snapshot = group.code_snapshot
        if snapshot is not None and not (snapshot.folder / SRC_FOLDER).is_dir():
            message = (
                f'code snapshot {snapshot.folder / SRC_FOLDER} is missing; '
                'not running the live code; rerun with --new-code-snapshot'
            )
            raise LaunchError(message)
        job_id = local_job_id(job_file=group.job_file, pid=self._pid)
        record(job_id=job_id)
        stream = sys.stdout if self._stream is None else self._stream
        if self._gpu_ids:
            slots = len(self._gpu_ids) * self._local.tasks_per_gpu
        else:
            slots = self._local.max_parallel
        by_key = {f'{job_id}_{task.index}': task for task in group.tasks}
        watchdog = Watchdog.start(grace_seconds=self._grace_seconds)
        states = _LaunchStates(on_task_state=self._on_task_state, watchdog=watchdog)

        output = TaskOutput(
            stream=stream,
            # why: a detached launch writes its console to a file; the task logs hold the lines
            single_slot=slots == 1 and DETACHED_VARIABLE not in os.environ,
            tasks={
                key: TaskInfo(
                    log_file=task.log_file,
                    index=task.index,
                    name=f'run {task.resolved.run.run_path}',
                    usage_file=usage_file(log_file=task.log_file),
                    usage_document={
                        'schema': 'tslab.local-usage/1',
                        'attempt': task.attempt,
                        'job_id': job_id,
                        'timeout': timeout_seconds(local=self._local),
                    },
                )
                for key, task in by_key.items()
            },
            header=lambda key: _task_header(
                task=by_key[key],
                job_id=job_id,
                host=self._host,
                now=self._now,
                gpu_ids=self._gpu_ids,
            ),
            on_task_state=states.on_task_state,
        )

        # why: the recheck after an expired answer window would log the same busy GPU again
        reported_busy: set[str] = set()
        reported_busy_lock = threading.Lock()

        def gpu_free(gpu_id: str) -> bool:
            """Whether the GPU is free of processes outside this launch."""
            busy = busy_gpus(
                gpu_ids=(gpu_id,), memory_fraction=self._local.busy_gpu_memory_fraction
            )
            if gpu_id not in busy:
                return True
            with reported_busy_lock:
                # why: the message is logged once per GPU, not on every recheck
                first = gpu_id not in reported_busy
                reported_busy.add(gpu_id)
            if first:
                _LOGGER.info('GPU %s is busy; waiting for it', gpu_id)
            return False

        def task_environment() -> dict[str, str]:
            # why: a nested launch must not run from the snapshot of the launch that started it
            env = without_code_snapshot(environ=without_scheduler_variables(environ=os.environ))
            env.update(self._env)
            env.setdefault('OMP_NUM_THREADS', str(group.resources.cpus_per_run))
            if snapshot is not None:
                env.update(snapshot.variables())
                env['PYTHONPATH'] = snapshot.pythonpath(current=env.get('PYTHONPATH'))
            return env

        try:
            with stop_on_signals(signals=_STOP_SIGNALS) as request:
                run_commands(
                    commands=[
                        PoolCommand(
                            key=key,
                            command=task.command,
                            cwd=self._project_root,
                            environment=task_environment(),
                        )
                        for key, task in by_key.items()
                    ],
                    slots=slots,
                    gpu_ids=self._gpu_ids,
                    tasks_per_gpu=self._local.tasks_per_gpu,
                    lock_folder=(
                        gpu_lock_folder(name='tslab-gpus', environment=os.environ)
                        if self._gpu_ids
                        else None
                    ),
                    stop=request.event,
                    timeout_seconds=timeout_seconds(local=self._local),
                    signal_seconds=self._local.signal_seconds,
                    on_start=output.on_start,
                    on_line=output.on_line,
                    on_event=output.on_event,
                    on_wait=output.on_wait,
                    on_end=output.on_end,
                    grace_seconds=self._grace_seconds,
                    on_spawn=states.on_spawn,
                    gpu_free=(gpu_free if self._local.skip_busy_gpus else None),
                )
            if request.signal_number is not None:
                raise LaunchStopped(signal_number=request.signal_number)
        finally:
            # why: a launch that dies must not leave the watchdog holding its runs
            watchdog.close()
        return job_id
