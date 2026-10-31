"""Running many commands in CPU slots, or GPU slots with lock files."""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
from types import MappingProxyType
from typing import cast

from tslab.jobs.local.gpus import GpuLease, try_lease_gpu
from tslab.jobs.local.process import ProcessResult, run_process

__all__ = ['PoolCommand', 'run_commands']


@dataclass(frozen=True, kw_only=True)
class PoolCommand:
    """One command of a pool, with the context in which it runs.

    Attributes:
        key: The id of the command; the first argument of every callback.
        command: The command and its arguments.
        cwd: The working directory of the command.
        environment: The full environment of the command, kept read-only.
    """

    key: str
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        """Keep the environment read-only, so that frozen really means frozen."""
        object.__setattr__(self, 'environment', MappingProxyType(self.environment))


def _check_lock_folder(*, gpu_ids: Sequence[str], lock_folder: Path | None) -> None:
    """Require a lock folder when the pool leases GPUs."""
    if gpu_ids and lock_folder is None:
        message = 'lock_folder is required when gpu_ids is not empty'
        raise ValueError(message)


def _release_lease(*, lease: GpuLease | None) -> None:
    """Release the lease, when there is one."""
    if lease is not None:
        lease.release()


def _environment_of(*, command: PoolCommand, slot: str | None) -> dict[str, str]:
    """The environment of one command, with the visible GPU id of its slot."""
    environment = {**command.environment}
    if slot is not None and slot != 'mps':
        environment['CUDA_VISIBLE_DEVICES'] = slot
    return environment


def _slot_queue_of(
    *, slots: int, gpu_ids: Sequence[str], tasks_per_gpu: int
) -> queue.Queue[str | None]:
    """The queue of free slots: one token per (gpu, lane), or one per slot."""
    slots_queue: queue.Queue[str | None] = queue.Queue()
    for slot in [gpu_id for gpu_id in gpu_ids for _ in range(tasks_per_gpu)] or [None] * slots:
        slots_queue.put(slot)
    return slots_queue


def _never_started_of(
    *, commands: Sequence[PoolCommand], never_started: set[str]
) -> tuple[str, ...]:
    """The keys that never started, in input order."""
    return tuple(command.key for command in commands if command.key in never_started)


def run_commands(  # noqa: PLR0913 # why: one keyword per slot, callback and limit setting
    *,
    commands: Sequence[PoolCommand],
    slots: int,
    gpu_ids: Sequence[str],
    tasks_per_gpu: int,
    lock_folder: Path | None,
    stop: threading.Event,
    timeout_seconds: float | None,
    signal_seconds: float,
    on_start: Callable[[str, str | None], None],
    on_line: Callable[[str, str], None],
    on_event: Callable[[str, str], None],
    on_wait: Callable[[str], None],
    on_end: Callable[[str, str | None, ProcessResult | OSError], None],
    grace_seconds: float = 30.0,
    gpu_poll_seconds: float = 5.0,
) -> tuple[str, ...]:
    """Run the commands in order, at most ``slots`` of them at once.

    Without ``gpu_ids`` a command only needs a free slot. With ``gpu_ids`` it
    also needs a lease from ``try_lease_gpu`` in ``lock_folder`` on the id of
    its slot; while none is free, ``on_wait`` is called once for the command
    and the pool retries every ``gpu_poll_seconds``, waking on a stop. A
    leased command runs with ``CUDA_VISIBLE_DEVICES`` set to the id of the
    lease, unless it is ``'mps'``, and keeps the lock while it runs.

    ``on_line``, ``on_event`` and ``on_end`` are called from the worker
    threads, not from the calling thread; the caller guards its own state.
    An ``OSError`` from starting a command is passed to ``on_end`` and does
    not stop the pool.

    When ``stop`` is set, no command starts; the running ones are stopped by
    ``run_process``, and the pool returns after every thread has ended.

    Args:
        commands: The commands, in the order they start.
        slots: The number of commands that run at once.
        gpu_ids: The ids to lease, in order; empty for CPU-only slots.
        tasks_per_gpu: The number of tasks that may share one GPU.
        lock_folder: The lock folder, required when ``gpu_ids`` is not empty.
        stop: An event that requests a stop at any time.
        timeout_seconds: The time limit of one command, or ``None`` for none.
        signal_seconds: Seconds before the limit to send ``SIGUSR2``;
            zero disables the signal.
        on_start: Called with the key and the GPU id when a command starts;
            the id is ``None`` without GPUs.
        on_line: Called with the key and each line a command writes.
        on_event: Called with the key and each signal event of a command.
        on_wait: Called with the key when a command waits for a GPU lease.
        on_end: Called with the key, the GPU id, and the ``ProcessResult`` or
            the ``OSError`` of a finished command.
        grace_seconds: Seconds to wait after a ``SIGTERM`` before a ``SIGKILL``.
        gpu_poll_seconds: Seconds between retries for a GPU lease.

    Returns:
        The keys of the commands that never started because ``stop`` was set.

    Raises:
        ValueError: If ``gpu_ids`` is not empty and ``lock_folder`` is ``None``.
    """
    _check_lock_folder(gpu_ids=gpu_ids, lock_folder=lock_folder)
    slot_queue = _slot_queue_of(slots=slots, gpu_ids=gpu_ids, tasks_per_gpu=tasks_per_gpu)
    never_started: set[str] = set()
    never_started_lock = threading.Lock()

    def take_lease(*, command: PoolCommand) -> tuple[GpuLease | None, str | None]:
        """Take one free slot, and the GPU lock of its id, for the command."""
        slot = slot_queue.get()
        if slot is None:
            return None, slot
        if lock_folder is None:
            # why: the check at the top of run_commands already raised; this narrows the type here
            message = 'lock_folder is required when gpu_ids is not empty'
            raise ValueError(message)
        waited = False
        while (
            lease := try_lease_gpu(folder=lock_folder, gpu_ids=(slot,), tasks_per_gpu=tasks_per_gpu)
        ) is None and not stop.is_set():
            if not waited:
                on_wait(command.key)
                waited = True
            slot_queue.put(slot)
            stop.wait(gpu_poll_seconds)
            # why: a pool with GPUs only ever puts GPU tokens into the queue
            slot = cast('str', slot_queue.get())
        return lease, slot

    def run_one(*, command: PoolCommand) -> None:
        lease, slot = take_lease(command=command)
        try:
            if stop.is_set():
                with never_started_lock:
                    never_started.add(command.key)
                return
            on_start(command.key, slot)
            environment = _environment_of(command=command, slot=slot)
            outcome: ProcessResult | OSError
            try:
                outcome = run_process(
                    command=command.command,
                    cwd=command.cwd,
                    environment=environment,
                    on_line=lambda line: on_line(command.key, line),
                    on_event=lambda text: on_event(command.key, text),
                    stop=stop,
                    timeout_seconds=timeout_seconds,
                    signal_seconds=signal_seconds,
                    grace_seconds=grace_seconds,
                    pass_fds=lease.fds if lease is not None else (),
                )
            except OSError as error:
                outcome = error
            on_end(command.key, slot, outcome)
        finally:
            # why: the slot and the lock must come back even when the command errors
            _release_lease(lease=lease)
            slot_queue.put(slot)

    with ThreadPoolExecutor(max_workers=slots) as executor:
        futures = [executor.submit(run_one, command=command) for command in commands]
    for future in futures:
        future.result()
    return _never_started_of(commands=commands, never_started=never_started)
