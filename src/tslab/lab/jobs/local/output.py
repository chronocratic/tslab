"""The log files and console lines of the tasks of one local launch."""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import TextIO
import uuid

from tslab.lab.jobs.local.process import ProcessResult

__all__ = ['TaskInfo', 'TaskOutput', 'tail_lines']


def tail_lines(*, path: Path, lines: int) -> list[str] | None:
    """Read the last lines of a file, or return ``None`` when it is missing.

    Args:
        path: The file to read.
        lines: Maximum number of trailing lines.

    Returns:
        The trailing lines without line endings, or ``None`` when missing.
    """
    try:
        with path.open(encoding='utf-8', errors='replace') as file:
            return [line.rstrip('\r\n') for line in deque(file, maxlen=lines)]
    except FileNotFoundError:
        return None


@dataclass(frozen=True, kw_only=True)
class TaskInfo:
    """One task of the output.

    Attributes:
        log_file: The file the lines of the task are appended to.
        index: The task number, shown on the console lines.
        name: The task name, shown between the start and end of the console
            lines.
        usage_file: The file the usage document is written to when the task
            exits.
        usage_document: The fixed part of the usage document; the metrics of
            the finished process are added by the writer.
    """

    log_file: Path
    index: int
    name: str
    usage_file: Path
    usage_document: Mapping[str, object]


class TaskOutput:
    """The log and console lines of the tasks of one launch, under one lock."""

    def __init__(
        self,
        *,
        stream: TextIO,
        single_slot: bool,
        tasks: Mapping[str, TaskInfo],
        header: Callable[[str], str],
        on_task_state: Callable[[str, str], None] | None,
    ) -> None:
        """Set up the output.

        Args:
            stream: The console stream.
            single_slot: Whether the launch runs one task at a time; then the
                console gets every line, not one line per start and end.
            tasks: The task info per key.
            header: The header line of the log of a task, given its key; called
                when the task starts.
            on_task_state: Called with the key and ``'running'`` or
                ``'ended'``, or ``None``.
        """
        self._stream = stream
        self._single_slot = single_slot
        self._tasks = tasks
        self._header = header
        self._on_task_state = on_task_state
        self._lock = threading.Lock()
        self._logs: dict[str, TextIO] = {}
        self._waits: dict[str, list[str]] = {}

    def on_start(self, key: str, gpu_id: str | None) -> None:
        """The log header of one task, and its start line on more than one slot."""
        task = self._tasks[key]
        task.log_file.parent.mkdir(parents=True, exist_ok=True)
        header = self._header(key)
        with self._lock:
            log = task.log_file.open('a', encoding='utf-8')
            log.write(header)
            # why: the wait lines come before the start, so they stay under the header
            for line in self._waits.pop(key, []):
                log.write(line)
            log.flush()
            self._logs[key] = log
            if not self._single_slot:
                started = (
                    f'=== task {task.index} started | {task.name} | gpu {gpu_id or "none"} ===\n'
                )
                self._console(line=started)
        if self._on_task_state is not None:
            self._on_task_state(key, 'running')

    def on_line(self, key: str, line: str) -> None:
        """One output line of a task: to the log, and to the console on one slot."""
        with self._lock:
            self._logs[key].write(line)
            self._logs[key].flush()
            if self._single_slot:
                self._console(line=line)

    def on_event(self, key: str, text: str) -> None:
        """One signal event of a task, as a line."""
        self.on_line(key=key, line=text + '\n')

    def on_wait(self, key: str) -> None:
        """One line when a task waits for a free GPU."""
        line = f'=== task {self._tasks[key].index} waiting for a free gpu ===\n'
        with self._lock:
            # why: the task may not have started, so its log is not open yet
            self._waits.setdefault(key, []).append(line)
            self._console(line=line)

    def on_end(self, key: str, gpu_id: str | None, outcome: ProcessResult | OSError) -> None:
        """The footer of a finished task, the usage file, and the closed log."""
        task = self._tasks[key]
        with self._lock:
            log = self._logs.pop(key)
            if isinstance(outcome, OSError):
                line = f'=== task {task.index} could not start: {outcome} ===\n'
                log.write(line)
                log.flush()
                self._console(line=line)
            else:
                log.write(self._footer(task=task, outcome=outcome))
                log.flush()
                self._console(line=self._end_line(task=task, gpu_id=gpu_id, outcome=outcome))
            log.close()
        if isinstance(outcome, ProcessResult):
            _write_json_atomic(
                path=task.usage_file,
                document={
                    **task.usage_document,
                    'exit_code': outcome.exit_code,
                    'elapsed_seconds': outcome.elapsed_seconds,
                    'cpu_seconds': outcome.cpu_seconds,
                    'max_rss_bytes': outcome.max_rss_bytes,
                    'gpu_id': gpu_id,
                    'timed_out': outcome.timed_out,
                },
            )
        if self._on_task_state is not None:
            self._on_task_state(key, 'ended')

    def _footer(self, *, task: TaskInfo, outcome: ProcessResult) -> str:
        """The line the log of the finished task ends with."""
        peak = _format_peak_memory(peak_bytes=outcome.max_rss_bytes)
        return (
            f'=== task {task.index}: exit {outcome.exit_code} '
            f'| elapsed {_format_elapsed(seconds=outcome.elapsed_seconds)} '
            f'| peak memory {peak} ===\n'
        )

    def _end_line(self, *, task: TaskInfo, gpu_id: str | None, outcome: ProcessResult) -> str:
        """The console line for a finished task; the footer itself on one slot."""
        if self._single_slot:
            return self._footer(task=task, outcome=outcome)
        return (
            f'=== task {task.index} ended | {task.name} '
            f'| gpu {gpu_id or "none"} | exit {outcome.exit_code} ===\n'
        )

    def _console(self, *, line: str) -> None:
        # why: the console output is the design
        print(line, end='', file=self._stream)


def _format_elapsed(*, seconds: float) -> str:
    """The elapsed time of the footer, as ``H:MM:SS``."""
    total = int(seconds)
    return f'{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}'


def _format_peak_memory(*, peak_bytes: int) -> str:
    """The peak memory of the footer, in the largest unit that keeps it at 1 or more."""
    for unit, divisor in (('GiB', 1024**3), ('MiB', 1024**2), ('KiB', 1024)):
        if peak_bytes >= divisor:
            return f'{peak_bytes / divisor:.1f} {unit}'
    return f'{peak_bytes / 1024:.1f} KiB'


def _write_json_atomic(*, path: Path, document: object) -> None:
    """Write ``document`` as indented JSON with sorted keys, atomically.

    A temporary sibling is flushed and renamed into place, so a crash cannot
    leave a half-written file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = f'{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}'
    temporary_path = path.with_name(f'{path.name}.tmp{unique}')
    try:
        with temporary_path.open(mode='w', encoding='utf-8') as file_handle:
            file_handle.write(json.dumps(document, indent=2, sort_keys=True) + '\n')
            file_handle.flush()
            os.fsync(file_handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    _sync_directory(directory=path.parent)


def _sync_directory(*, directory: Path) -> None:
    """Flush a directory entry to disk so a rename survives a crash (best effort)."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # not supported here (for example on Windows)
    try:
        os.fsync(descriptor)
    except OSError:
        pass  # some network filesystems refuse fsync on a directory
    finally:
        os.close(descriptor)
