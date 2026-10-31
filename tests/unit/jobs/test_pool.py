from collections.abc import Sequence
import itertools
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tslab.jobs.local import PoolCommand, ProcessResult, run_commands

_LOCK_CHILD = (
    'import fcntl, os, sys, time\n'
    'fd = os.open(os.path.join(sys.argv[1], "gpu-0.lock"), os.O_RDWR | os.O_CREAT)\n'
    'fcntl.flock(fd, fcntl.LOCK_EX)\n'
    'open(sys.argv[2], "w").write("ready")\n'
    'time.sleep(120)\n'
)


def _hold_gpu_zero(*, folder: Path) -> subprocess.Popen[bytes]:
    ready = folder / 'ready'
    process = subprocess.Popen(  # noqa: S603 # why: the child only takes the lock file
        [sys.executable, '-c', _LOCK_CHILD, folder.as_posix(), ready.as_posix()], cwd=folder
    )
    deadline = time.monotonic() + 10
    while not ready.exists():
        if process.poll() is not None:
            message = 'the lock holder exited before taking the lock'
            raise AssertionError(message)
        if time.monotonic() > deadline:
            message = 'the lock holder did not take the lock in time'
            raise AssertionError(message)
        time.sleep(0.05)
    return process


def _fake_task(*, sleep_seconds: float) -> str:
    return (
        'import os, time\n'
        'with open(os.environ["MARK"], "w") as handle:\n'
        '    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")\n'
        '    handle.write(f"{time.time():.6f} start {gpu}\\n")\n'
        f'time.sleep({sleep_seconds})\n'
        'with open(os.environ["MARK"], "a") as handle:\n'
        '    handle.write(f"{time.time():.6f} end\\n")\n'
    )


def _command_of(*, tmp_path: Path, key: str, sleep_seconds: float) -> PoolCommand:
    return PoolCommand(
        key=key,
        command=(sys.executable, '-c', _fake_task(sleep_seconds=sleep_seconds)),
        cwd=tmp_path,
        environment={**os.environ, 'MARK': (tmp_path / f'{key}.times').as_posix()},
    )


def _run_pool(  # noqa: PLR0913 # why: one keyword per pool setting the test overrides
    *,
    tmp_path: Path,
    keys: Sequence[str],
    slots: int,
    gpu_ids: Sequence[str] = (),
    tasks_per_gpu: int = 1,
    lock_folder: Path | None = None,
    sleep_seconds: float = 1.0,
    gpu_poll_seconds: float = 5.0,
    stop: threading.Event | None = None,
) -> tuple[
    tuple[str, ...], list[tuple[str, str | None]], list[str], list[tuple[str, str | None, object]]
]:
    starts: list[tuple[str, str | None]] = []
    waits: list[str] = []
    ends: list[tuple[str, str | None, object]] = []
    commands = [
        _command_of(tmp_path=tmp_path, key=key, sleep_seconds=sleep_seconds) for key in keys
    ]
    never_started = run_commands(
        commands=commands,
        slots=slots,
        gpu_ids=gpu_ids,
        tasks_per_gpu=tasks_per_gpu,
        lock_folder=lock_folder,
        stop=stop or threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
        on_start=lambda key, gpu_id: starts.append((key, gpu_id)),
        on_line=lambda _key, _line: None,
        on_event=lambda _key, _text: None,
        on_wait=waits.append,
        on_end=lambda key, gpu_id, outcome: ends.append((key, gpu_id, outcome)),
        gpu_poll_seconds=gpu_poll_seconds,
    )
    return never_started, starts, waits, ends


def _max_overlap(*, tmp_path: Path, keys: Sequence[str]) -> int:
    events: list[tuple[float, int]] = []
    for key in keys:
        lines = (tmp_path / f'{key}.times').read_text().splitlines()
        events.append((float(lines[0].split()[0]), 1))
        events.append((float(lines[1].split()[0]), -1))
    peak = 0
    current = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def _seen_gpus(*, tmp_path: Path, keys: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    for key in keys:
        line = (tmp_path / f'{key}.times').read_text().splitlines()[0]
        parts = line.split()
        if len(parts) > 2:
            seen.add(parts[2])
    return seen


def test_pool_three_slots_overlap_at_most_three(*, tmp_path: Path) -> None:
    keys = tuple(f'k{i}' for i in range(5))
    never_started, starts, waits, ends = _run_pool(tmp_path=tmp_path, keys=keys, slots=3)
    assert never_started == ()
    assert _max_overlap(tmp_path=tmp_path, keys=keys) <= 3
    assert len(starts) == 5
    assert waits == []
    assert sorted(key for key, _gpu_id, _outcome in ends) == list(keys)
    for _key, _gpu_id, outcome in ends:
        assert isinstance(outcome, ProcessResult)
        assert outcome.exit_code == 0


def test_pool_one_slot_runs_in_order(*, tmp_path: Path) -> None:
    keys = tuple(f'k{i}' for i in range(3))
    never_started, _starts, _waits, ends = _run_pool(
        tmp_path=tmp_path, keys=keys, slots=1, sleep_seconds=0.3
    )
    assert never_started == ()
    assert len(ends) == 3
    for first, second in itertools.pairwise(keys):
        end_first = float((tmp_path / f'{first}.times').read_text().splitlines()[1].split()[0])
        start_second = float((tmp_path / f'{second}.times').read_text().splitlines()[0].split()[0])
        assert end_first < start_second


def test_pool_gpu_slots_set_cuda_visible_devices(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    keys = tuple(f'k{i}' for i in range(4))
    never_started, starts, _waits, _ends = _run_pool(
        tmp_path=tmp_path, keys=keys, slots=2, gpu_ids=('0', '1'), lock_folder=folder
    )
    assert never_started == ()
    assert _max_overlap(tmp_path=tmp_path, keys=keys) <= 2
    assert _seen_gpus(tmp_path=tmp_path, keys=keys) <= {'0', '1'}
    assert {gpu_id for _key, gpu_id in starts} == {'0', '1'}


def test_pool_tasks_per_gpu_two_shares_one_gpu(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    keys = tuple(f'k{i}' for i in range(3))
    never_started, _starts, _waits, _ends = _run_pool(
        tmp_path=tmp_path, keys=keys, slots=2, gpu_ids=('0',), tasks_per_gpu=2, lock_folder=folder
    )
    assert never_started == ()
    assert _max_overlap(tmp_path=tmp_path, keys=keys) == 2
    assert _seen_gpus(tmp_path=tmp_path, keys=keys) == {'0'}


def test_pool_mps_leaves_environment_untouched(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    never_started, starts, _waits, ends = _run_pool(
        tmp_path=tmp_path, keys=('k0',), slots=1, gpu_ids=('mps',), lock_folder=folder
    )
    assert never_started == ()
    assert starts == [('k0', 'mps')]
    assert 'CUDA_VISIBLE_DEVICES' not in (tmp_path / 'k0.times').read_text()
    assert len(ends) == 1


def test_pool_waits_for_gpu_held_elsewhere(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    holder = _hold_gpu_zero(folder=folder)
    timer = threading.Timer(0.5, holder.kill)
    timer.start()
    never_started, _starts, waits, ends = _run_pool(
        tmp_path=tmp_path,
        keys=('k0',),
        slots=1,
        gpu_ids=('0',),
        lock_folder=folder,
        sleep_seconds=0.5,
        gpu_poll_seconds=0.2,
    )
    holder.wait()
    assert never_started == ()
    assert waits == ['k0']
    assert (tmp_path / 'k0.times').exists()
    assert len(ends) == 1


def test_pool_start_error_reaches_on_end(*, tmp_path: Path) -> None:
    ends: list[tuple[str, str | None, object]] = []
    commands = [
        PoolCommand(
            key='k0',
            command=('/nonexistent/definitely-missing',),
            cwd=tmp_path,
            environment=os.environ,
        ),
        _command_of(tmp_path=tmp_path, key='k1', sleep_seconds=0.2),
    ]
    never_started = run_commands(
        commands=commands,
        slots=2,
        gpu_ids=(),
        tasks_per_gpu=1,
        lock_folder=None,
        stop=threading.Event(),
        timeout_seconds=None,
        signal_seconds=0,
        on_start=lambda _key, _gpu_id: None,
        on_line=lambda _key, _line: None,
        on_event=lambda _key, _text: None,
        on_wait=lambda _key: None,
        on_end=lambda key, gpu_id, outcome: ends.append((key, gpu_id, outcome)),
    )
    assert never_started == ()
    by_key = {key: outcome for key, _gpu_id, outcome in ends}
    first = by_key['k0']
    second = by_key['k1']
    assert isinstance(first, OSError)
    assert isinstance(second, ProcessResult)
    assert second.exit_code == 0


def test_pool_stop_starts_nothing_new(*, tmp_path: Path) -> None:
    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set)
    timer.start()
    start = time.monotonic()
    never_started, starts, _waits, ends = _run_pool(
        tmp_path=tmp_path, keys=('k0', 'k1', 'k2'), slots=1, sleep_seconds=60, stop=stop
    )
    assert time.monotonic() - start < 3
    assert tuple(sorted(never_started)) == ('k1', 'k2')
    assert [key for key, _gpu_id in starts] == ['k0']
    assert [key for key, _gpu_id, _outcome in ends] == ['k0']
