"""Running many commands in CPU slots, or GPU slots with lock files."""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import queue
import threading
import time
from types import MappingProxyType
from typing import cast, Protocol

from tslab.lab.jobs.local.gpus import GpuLease, MPS_GPU, try_lease_gpu
from tslab.lab.jobs.local.process import ProcessResult, run_process

__all__ = [
    'OnEnd',
    'OnEvent',
    'OnLine',
    'OnSpawn',
    'OnStart',
    'OnWait',
    'PoolCommand',
    'run_commands',
]


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


class OnStart(Protocol):
    """A callback of the pool, called when a command starts."""

    def __call__(self, *, key: str, gpu_id: str | None) -> None:
        """Called with the key and the GPU id when a command starts."""
        ...


class OnSpawn(Protocol):
    """A callback of the pool, called when the process of a command starts."""

    def __call__(self, *, key: str, pid: int) -> None:
        """Called with the key and the pid of the process of a command."""
        ...


class OnLine(Protocol):
    """A callback of the pool, called for the lines of a command."""

    def __call__(self, *, key: str, line: str) -> None:
        """Called with the key and each line a command writes."""
        ...


class OnEvent(Protocol):
    """A callback of the pool, called for the signal events of a command."""

    def __call__(self, *, key: str, text: str) -> None:
        """Called with the key and each signal event of a command."""
        ...


class OnWait(Protocol):
    """A callback of the pool, called while a command waits for a GPU lease."""

    def __call__(self, *, key: str) -> None:
        """Called with the key when a command waits for a GPU lease."""
        ...


class OnEnd(Protocol):
    """A callback of the pool, called when a command ends."""

    def __call__(self, *, key: str, gpu_id: str | None, outcome: ProcessResult | OSError) -> None:
        """Called with the key, the GPU id, and the outcome when a command ends."""
        ...


def _check_lock_folder(*, gpu_ids: Sequence[str], lock_folder: Path | None) -> None:
    """Require a lock folder when the pool leases GPUs."""
    if gpu_ids and lock_folder is None:
        message = 'lock_folder is required when gpu_ids is not empty'
        raise ValueError(message)


def _add_locked(*, lock: threading.Lock, store: set[str], key: str) -> None:
    """Add a key to a set that several worker threads share."""
    with lock:
        store.add(key)


def _caught(run: Callable[[], ProcessResult]) -> ProcessResult | OSError:
    """The result of a process run, or the ``OSError`` that kept it from starting."""
    try:
        return run()
    except OSError as error:
        return error


def _release_lease(*, lease: GpuLease | None) -> None:
    """Release the lease, when there is one."""
    if lease is not None:
        lease.release()


def _environment_of(*, command: PoolCommand, slot: str | None) -> dict[str, str]:
    """The environment of one command, with the visible GPU id of its slot."""
    environment = {**command.environment}
    if slot is not None and slot != MPS_GPU:
        environment['CUDA_VISIBLE_DEVICES'] = slot
        # why: nvidia-smi ids follow the PCI bus; CUDA numbers the fastest GPU first
        # unless told otherwise
        environment.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
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


def _spawn_callback(*, on_spawn: OnSpawn | None, key: str) -> Callable[[int], None] | None:
    """The per-key ``on_spawn`` callback for one command, or ``None`` without one."""
    if on_spawn is None:
        return None
    return lambda pid: on_spawn(key=key, pid=pid)


class _GpuLeaseGate:
    """The lease of a GPU slot, gated by the own leases of the pool and a free recheck.

    A slot the pool holds a lease for is leased without a recheck. Otherwise
    ``gpu_free`` decides, and its answer is kept per GPU for the recheck
    window, so the poll does not call it again inside the window. The answer
    dies with the last lease of the pool on the GPU, so a free answer is
    never trusted for a new lease.
    """

    def __init__(
        self,
        *,
        lock_folder: Path | None,
        tasks_per_gpu: int,
        gpu_free: Callable[[str], bool] | None,
        busy_recheck_seconds: float,
    ) -> None:
        self._lock_folder = lock_folder
        self._tasks_per_gpu = tasks_per_gpu
        self._gpu_free = gpu_free
        self._busy_recheck_seconds = busy_recheck_seconds
        self._counts: dict[str, int] = {}
        self._answers: dict[str, tuple[float, bool]] = {}
        self._generations: dict[str, int] = {}
        self._gpu_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def _holds(self, *, gpu_id: str) -> bool:
        """Whether the pool holds at least one lease on the GPU."""
        with self._lock:
            return self._counts.get(gpu_id, 0) > 0

    def _gpu_lock(self, *, gpu_id: str) -> threading.Lock:
        """The lock that serializes the free checks of one GPU."""
        with self._lock:
            return self._gpu_locks.setdefault(gpu_id, threading.Lock())

    def _fresh_answer(self, *, slot: str) -> bool | None:
        """Whether the GPU was free at the last check, or ``None`` when the answer is stale."""
        with self._lock:
            answer = self._answers.get(slot)
        if answer is None or time.monotonic() - answer[0] >= self._busy_recheck_seconds:
            return None
        return answer[1]

    def _is_busy(self, *, slot: str) -> bool:
        """Whether the pool must wait for the GPU of the slot before leasing it."""
        if self._gpu_free is None:
            return False
        # why: the lock of this GPU stays held during the call, so a second thread on the
        # same GPU reads the answer instead of calling again, while the shared lock stays
        # free for the other GPUs and for the lease counts (the call can take 30 seconds)
        with self._gpu_lock(gpu_id=slot):
            fresh = self._fresh_answer(slot=slot)
            if fresh is not None:
                return not fresh
            return not self._ask(slot=slot)

    def _ask(self, *, slot: str) -> bool:
        """Call ``gpu_free`` for the slot; keep the answer unless the last lease ended meanwhile."""
        gpu_free = self._gpu_free
        if gpu_free is None:
            message = 'gpu_free is required to ask about a GPU'
            raise ValueError(message)
        with self._lock:
            generation = self._generations.get(slot, 0)
        free = gpu_free(slot)
        with self._lock:
            # why: the call can take 30 seconds; when the last lease ended meanwhile, the
            # answer is from before the release and must not be trusted for a new lease
            if self._generations.get(slot, 0) == generation:
                self._answers[slot] = (time.monotonic(), free)
        return free

    def try_lease(self, *, slot: str) -> GpuLease | None:
        """The lease of the slot, or ``None`` while the pool must wait for it."""
        if not self._holds(gpu_id=slot) and self._is_busy(slot=slot):
            return None
        if self._lock_folder is None:
            # why: the check at the top of run_commands already raised; this narrows the type here
            message = 'lock_folder is required when gpu_ids is not empty'
            raise ValueError(message)
        lease = try_lease_gpu(
            folder=self._lock_folder, gpu_ids=(slot,), tasks_per_gpu=self._tasks_per_gpu
        )
        if lease is not None:
            with self._lock:
                self._counts[slot] = self._counts.get(slot, 0) + 1
        return lease

    def drop(self, *, lease: GpuLease | None, gpu_id: str | None) -> None:
        """Count one less lease of the pool on the GPU; nothing without a lease."""
        if lease is None or gpu_id is None:
            return
        with self._lock:
            self._counts[gpu_id] = self._counts.get(gpu_id, 1) - 1
            if self._counts[gpu_id] == 0:
                # why: a free answer from before the release must not lease the GPU again
                self._answers.pop(gpu_id, None)
                self._generations[gpu_id] = self._generations.get(gpu_id, 0) + 1


_WAKE_SECONDS = 0.1


def _wait_for_either(*, first: threading.Event, second: threading.Event, seconds: float) -> None:
    """Wait up to ``seconds``, ending early when either event is set."""
    deadline = time.monotonic() + seconds
    while not (first.is_set() or second.is_set()) and (left := deadline - time.monotonic()) > 0:
        first.wait(min(left, _WAKE_SECONDS))


def _wait_for_next_slot(  # noqa: PLR0913 # why: one keyword per event and setting of the wait
    *,
    slot_queue: queue.Queue[str | None],
    slot: str,
    on_wait: Callable[[], None] | None,
    stop: threading.Event,
    halt: threading.Event,
    poll_seconds: float,
) -> str:
    """Give the slot back, wait one poll and take the next free slot.

    Args:
        slot_queue: The queue of free slots.
        slot: The slot the caller holds.
        on_wait: Called once before the slot goes back, or ``None`` when it was called.
        stop: An event that ends the wait early.
        halt: A second event that ends the wait early.
        poll_seconds: The seconds to wait.

    Returns:
        The next free slot.
    """
    # why: a failing callback must not keep the slot token, or the other commands
    # wait for it forever
    try:
        if on_wait is not None:
            on_wait()
    finally:
        slot_queue.put(slot)
    _wait_for_either(first=stop, second=halt, seconds=poll_seconds)
    # why: a pool with GPUs only ever puts GPU tokens into the queue
    return cast('str', slot_queue.get())


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
    on_start: OnStart,
    on_line: OnLine,
    on_event: OnEvent,
    on_wait: OnWait,
    on_end: OnEnd,
    grace_seconds: float = 30.0,
    gpu_poll_seconds: float = 5.0,
    on_spawn: OnSpawn | None = None,
    gpu_free: Callable[[str], bool] | None = None,
    busy_recheck_seconds: float = 60.0,
) -> tuple[str, ...]:
    """Run the commands in order, at most ``slots`` of them at once.

    Without ``gpu_ids`` a command only needs a free slot. With ``gpu_ids`` it
    also needs a lease from ``try_lease_gpu`` in ``lock_folder`` on the id of
    its slot; while none is free, ``on_wait`` is called once for the command
    and the pool retries every ``gpu_poll_seconds``, waking on a stop. With
    ``gpu_free``, a slot the pool holds no lease for is leased only while
    ``gpu_free`` reports its GPU free; the busy answer is kept per GPU for
    ``busy_recheck_seconds``, so the poll does not recheck inside the window,
    and a busy GPU waits like a held one. A
    leased command runs with ``CUDA_VISIBLE_DEVICES`` set to the id of the
    lease; a numeric slot also gets ``CUDA_DEVICE_ORDER`` set to
    ``PCI_BUS_ID`` unless the environment already sets it. The ``MPS_GPU``
    slot leaves the environment untouched. The command keeps the lock
    while it runs.

    The callbacks are called with keyword arguments only. ``on_line``,
    ``on_event`` and ``on_end`` are called from the worker threads, not from
    the calling thread; the caller guards its own state.
    An ``OSError`` from starting a command is passed to ``on_end`` and does
    not stop the pool.

    When ``stop`` is set, no command starts; the running ones are stopped by
    ``run_process``, and the pool returns after every thread has ended.
    A failing lease check starts no new command; the running ones finish and
    the error is raised after them.

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
        on_spawn: Called with the key and the pid of the process right after
            it starts, or ``None`` for no callback.
        gpu_free: Called with a GPU id to ask whether it is free, while the
            pool holds no lease on it; a busy GPU waits like a held one.
            ``None`` skips the check.
        busy_recheck_seconds: Seconds to keep the busy answer of ``gpu_free``
            per GPU, so the lease poll does not recheck inside the window.

    Returns:
        The keys of the commands that never started because ``stop`` was set.

    Raises:
        ValueError: If ``gpu_ids`` is not empty and ``lock_folder`` is ``None``.
    """
    _check_lock_folder(gpu_ids=gpu_ids, lock_folder=lock_folder)
    slot_queue = _slot_queue_of(slots=slots, gpu_ids=gpu_ids, tasks_per_gpu=tasks_per_gpu)
    never_started: set[str] = set()
    # why: set by a failing lease check; no new command starts, and the running ones finish
    halt = threading.Event()
    never_started_lock = threading.Lock()

    gate = _GpuLeaseGate(
        lock_folder=lock_folder,
        tasks_per_gpu=tasks_per_gpu,
        gpu_free=gpu_free,
        busy_recheck_seconds=busy_recheck_seconds,
    )

    def try_lease_or_return_slot(*, slot: str) -> GpuLease | None:
        """Try to lease the GPU of the slot; a failing check gives the slot back."""
        try:
            return gate.try_lease(slot=slot)
        except Exception:
            # why: a failing lease check must not keep the slot token, or the other commands
            # wait for it forever
            slot_queue.put(slot)
            raise

    def take_lease(*, command: PoolCommand) -> tuple[GpuLease | None, str | None]:
        """Take one free slot, and the GPU lock of its id, for the command."""
        slot = slot_queue.get()
        if slot is None:
            return None, slot
        waited = False
        while (lease := try_lease_or_return_slot(slot=slot)) is None and not (
            stop.is_set() or halt.is_set()
        ):
            slot = _wait_for_next_slot(
                slot_queue=slot_queue,
                slot=slot,
                on_wait=None if waited else partial(on_wait, key=command.key),
                stop=stop,
                halt=halt,
                poll_seconds=gpu_poll_seconds,
            )
            waited = True
        return lease, slot

    def run_one(*, command: PoolCommand) -> None:
        try:
            lease, slot = take_lease(command=command)
        except Exception:
            # why: no new command starts once a lease check fails; the running ones finish and
            # the error is raised after them
            halt.set()
            raise
        try:
            if stop.is_set() or halt.is_set():
                _add_locked(lock=never_started_lock, store=never_started, key=command.key)
                return
            on_start(key=command.key, gpu_id=slot)
            outcome = _caught(
                lambda: run_process(
                    command=command.command,
                    cwd=command.cwd,
                    environment=_environment_of(command=command, slot=slot),
                    on_line=lambda line: on_line(key=command.key, line=line),
                    on_event=lambda text: on_event(key=command.key, text=text),
                    stop=stop,
                    timeout_seconds=timeout_seconds,
                    signal_seconds=signal_seconds,
                    grace_seconds=grace_seconds,
                    pass_fds=lease.fds if lease is not None else (),
                    on_spawn=_spawn_callback(on_spawn=on_spawn, key=command.key),
                )
            )
            on_end(key=command.key, gpu_id=slot, outcome=outcome)
        finally:
            # why: the slot and the lock must come back even when the command errors
            _release_lease(lease=lease)
            gate.drop(lease=lease, gpu_id=slot)
            slot_queue.put(slot)

    with ThreadPoolExecutor(max_workers=slots) as executor:
        futures = [executor.submit(run_one, command=command) for command in commands]
    for future in futures:
        future.result()
    return _never_started_of(commands=commands, never_started=never_started)
