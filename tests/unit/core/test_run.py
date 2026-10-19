from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import override

import pytest

from tslab.core.meta import hostname
from tslab.core.run import (
    _acquire_lock,
    _release_lock,
    is_run_active,
    Run,
    RunLockedError,
    RunOptions,
)
from tslab.core.status import InstanceRecord, InstanceState
from tslab.core.tracker import LocalTracker
from tslab.interfaces.tracker import RunStatus, Tracker


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 7
    run: RunOptions = field(default_factory=RunOptions)


class RecordingTracker(Tracker):
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    @override
    def start(self, *, run_directory: Path, run_name: str) -> None:
        self.events.append(('start', run_name))

    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        self.events.append(('params', dict(params)))

    @override
    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.events.append(('metrics', dict(metrics)))

    @override
    def log_file(self, *, path: Path) -> None:
        self.events.append(('file', path))

    @override
    def finish(self, *, status: RunStatus) -> None:
        self.events.append(('finish', status))


class BrokenTracker(RecordingTracker):
    @override
    def log_params(self, *, params: Mapping[str, object]) -> None:
        message = 'tracker down'
        raise ConnectionError(message)


def test_run_rejects_str_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r'root must be pathlib\.Path'):
        Run(
            config=Config(),
            root=str(tmp_path),  # ty: ignore[invalid-argument-type]
            relative_path=Path('a'),
        )
    with pytest.raises(TypeError, match=r'relative_path must be pathlib\.Path'):
        Run(config=Config(), root=tmp_path, relative_path='a')  # ty: ignore[invalid-argument-type]


def test_run_rejects_escaping_relative_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='invalid name part'):
        Run(config=Config(), root=tmp_path, relative_path=Path('../outside'))


def test_run_requires_dataclass_config(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match='dataclass instance'):
        Run(config={'seed': 1}, root=tmp_path, relative_path=Path('a'))


def test_meta_and_hashes(tmp_path: Path) -> None:
    with Run(config=Config(seed=1), root=tmp_path, relative_path=Path('a') / 'seed_1'):
        pass
    with Run(config=Config(seed=2), root=tmp_path, relative_path=Path('a') / 'seed_2'):
        pass
    first = json.loads((tmp_path / 'a' / 'seed_1' / 'meta.json').read_text(encoding='utf-8'))
    second = json.loads((tmp_path / 'a' / 'seed_2' / 'meta.json').read_text(encoding='utf-8'))
    assert first['format_version'] == 1
    assert first['relative_path'] == 'a/seed_1'
    assert first['config_hash'] != second['config_hash']
    assert first['config_hash_without_seed'] == second['config_hash_without_seed']
    assert first['finished_at'] is not None


def test_run_options_do_not_change_config_hash(tmp_path: Path) -> None:
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        pass
    with Run(config=Config(run=RunOptions(force_all=True)), root=tmp_path, relative_path=Path('b')):
        pass
    first = json.loads((tmp_path / 'a' / 'meta.json').read_text(encoding='utf-8'))
    second = json.loads((tmp_path / 'b' / 'meta.json').read_text(encoding='utf-8'))
    assert first['config_hash'] == second['config_hash']


def test_tracker_lifecycle_and_failure_status(tmp_path: Path) -> None:
    tracker = RecordingTracker()
    with (
        pytest.raises(ZeroDivisionError),
        Run(config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[tracker]),
    ):
        raise ZeroDivisionError
    assert tracker.events[0] == ('start', 'a')
    # Parameters are the config.yaml view: numbers stay numbers, not fingerprint strings.
    assert tracker.events[1] == (
        'params',
        {
            'seed': 7,
            'run/force_all': False,
            'run/rerun_from': None,
            'run/rerun': [],
            'run/fail_fast': False,
            'run/dry_run': False,
            'run/log_level': 'INFO',
        },
    )
    assert tracker.events[-1] == ('finish', RunStatus.FAILED)
    status = json.loads((tmp_path / 'a' / 'status.json').read_text(encoding='utf-8'))
    assert status['run_state'] == 'failed'


def test_keyboard_interrupt_marks_crashed(tmp_path: Path) -> None:
    tracker = RecordingTracker()
    with (
        pytest.raises(KeyboardInterrupt),
        Run(config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[tracker]),
    ):
        raise KeyboardInterrupt
    assert tracker.events[-1] == ('finish', RunStatus.CRASHED)


def test_broken_tracker_never_stops_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tracker = BrokenTracker()
    with Run(config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[tracker]):
        pass
    assert 'tracker BrokenTracker failed in log_params' in caplog.text
    assert tracker.events[-1] == ('finish', RunStatus.FINISHED)


def test_second_live_process_is_locked_out(tmp_path: Path) -> None:
    second = Run(config=Config(), root=tmp_path, relative_path=Path('a'))
    with (
        Run(config=Config(), root=tmp_path, relative_path=Path('a')),
        pytest.raises(RunLockedError, match='is in use'),
    ):
        second.__enter__()
    assert not (tmp_path / 'a' / '.lock').exists()


def test_stale_lock_of_dead_process_is_replaced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dead = subprocess.run(
        [sys.executable, '-c', 'import os; print(os.getpid())'],
        capture_output=True,
        text=True,
        check=True,
    )
    directory = tmp_path / 'a'
    directory.mkdir()
    (directory / '.lock').write_text(
        json.dumps({'pid': int(dead.stdout), 'host': hostname()}), encoding='utf-8'
    )
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        pass
    assert 'removing stale lock' in caplog.text


def test_lock_from_other_host_is_respected(tmp_path: Path) -> None:
    (tmp_path / '.lock').write_text(json.dumps({'pid': 1, 'host': 'elsewhere'}), encoding='utf-8')
    with pytest.raises(RunLockedError):
        _acquire_lock(path=tmp_path / '.lock')


def _age_file(*, path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, times=(then, then))


def test_lock_without_heartbeat_from_other_host_is_replaced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / 'a'
    directory.mkdir()
    lock = directory / '.lock'
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123'}), encoding='utf-8')
    _age_file(path=lock, seconds=601)
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        pass
    assert 'removing stale lock' in caplog.text


def test_heartbeat_refreshes_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('tslab.core.run._HEARTBEAT_SECONDS', 0.01)
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        lock = tmp_path / 'a' / '.lock'
        _age_file(path=lock, seconds=500)
        deadline = time.time() + 5
        while time.time() - lock.stat().st_mtime > 60 and time.time() < deadline:
            time.sleep(0.01)
        assert time.time() - lock.stat().st_mtime < 60
    assert 'tslab-heartbeat' not in {thread.name for thread in threading.enumerate()}


def test_is_run_active(tmp_path: Path) -> None:
    assert not is_run_active(directory=tmp_path)
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        assert is_run_active(directory=tmp_path / 'a')
    lock = tmp_path / '.lock'
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123'}), encoding='utf-8')
    assert is_run_active(directory=tmp_path)
    _age_file(path=lock, seconds=601)
    assert not is_run_active(directory=tmp_path)


def test_non_zero_rank_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOCAL_RANK', '1')
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
        assert not run.writes_files
    assert not (tmp_path / 'a').exists()


def test_seed_defaults(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    @dataclass(frozen=True, kw_only=True)
    class NoSeed:
        value: int = 1

    run = Run(config=NoSeed(), root=tmp_path, relative_path=Path('a'))
    assert run.seed == 0
    assert 'using seed 0' in caplog.text
    assert Run(config=Config(seed=5), root=tmp_path, relative_path=Path('a')).seed == 5


def test_run_cannot_be_entered_twice(tmp_path: Path) -> None:
    run = Run(config=Config(), root=tmp_path, relative_path=Path('a'))
    with run:
        pass
    with pytest.raises(RuntimeError, match='only once'):
        run.__enter__()


def test_concurrent_takeover_of_a_stale_lock_has_one_winner(tmp_path: Path) -> None:
    lock = tmp_path / '.lock'
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123'}), encoding='utf-8')
    _age_file(path=lock, seconds=601)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def contender() -> None:
        barrier.wait()
        try:
            _acquire_lock(path=lock)
        except RunLockedError:
            outcomes.append('locked out')
        else:
            outcomes.append('owner')

    threads = [threading.Thread(target=contender) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count('owner') == 1
    assert outcomes.count('locked out') == 7


def test_release_only_removes_the_lock_of_the_owner(tmp_path: Path) -> None:
    lock = tmp_path / '.lock'
    mine = _acquire_lock(path=lock)
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123', 'nonce': 'theirs'}), encoding='utf-8')
    _release_lock(path=lock, nonce=mine)
    assert lock.exists()
    _release_lock(path=lock, nonce='theirs')
    assert not lock.exists()


def test_run_leaves_a_lock_taken_over_by_another_process(tmp_path: Path) -> None:
    lock = tmp_path / 'a' / '.lock'
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        lock.write_text(json.dumps({'pid': 1, 'host': 'node123', 'nonce': 'x'}), encoding='utf-8')
    assert json.loads(lock.read_text(encoding='utf-8'))['nonce'] == 'x'


def test_heartbeat_does_not_touch_a_foreign_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('tslab.core.run._HEARTBEAT_SECONDS', 0.01)
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        lock = tmp_path / 'a' / '.lock'
        lock.write_text(json.dumps({'pid': 1, 'host': 'node123', 'nonce': 'x'}), encoding='utf-8')
        _age_file(path=lock, seconds=500)
        before = lock.stat().st_mtime
        time.sleep(0.2)
        assert lock.stat().st_mtime == before


def test_default_tracker_is_local_and_core_stays_free_of_lightning(tmp_path: Path) -> None:
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
        assert isinstance(run.trackers[0], LocalTracker)
        run.log_metrics(metrics={'loss': 0.5}, step=3)
    assert (tmp_path / 'a' / 'trackers' / 'csv' / 'version_0' / 'metrics.csv').is_file()


def test_log_file_reaches_every_tracker(tmp_path: Path) -> None:
    recorder = RecordingTracker()
    note = tmp_path / 'note.txt'
    note.write_text('hi', encoding='utf-8')
    with Run(config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[recorder]) as run:
        run.log_file(path=note)
    assert ('file', note) in recorder.events


def test_a_given_local_tracker_replaces_the_default(tmp_path: Path) -> None:
    mine = LocalTracker()
    run = Run(config=Config(), root=tmp_path, relative_path=Path('a'), trackers=[mine])
    assert run.trackers == (mine,)


def test_sigterm_ends_the_run_as_crashed(tmp_path: Path) -> None:
    before = signal.getsignal(signal.SIGTERM)
    with (
        pytest.raises(SystemExit) as raised,
        Run(config=Config(), root=tmp_path, relative_path=Path('a')),
    ):
        signal.raise_signal(signal.SIGTERM)
    assert raised.value.code == 128 + signal.SIGTERM
    status = json.loads((tmp_path / 'a' / 'status.json').read_text(encoding='utf-8'))
    assert status['run_state'] == 'crashed'
    assert not (tmp_path / 'a' / '.lock').exists()
    assert signal.getsignal(signal.SIGTERM) == before


def test_logger_levels_are_restored(tmp_path: Path) -> None:
    tslab_logger = logging.getLogger('tslab')
    before = tslab_logger.level
    with Run(
        config=Config(run=RunOptions(log_level='DEBUG')), root=tmp_path, relative_path=Path('a')
    ):
        assert tslab_logger.level == logging.DEBUG
    assert tslab_logger.level == before


def test_summary_counts_only_this_execution(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
        run.claim_instance(stage_name='old', name_parts=('default',))
        run.record_instance(
            instance_id='old[default]', record=InstanceRecord(state=InstanceState.DONE)
        )
    caplog.clear()
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
        run.claim_instance(stage_name='new', name_parts=('default',))
        run.record_instance(
            instance_id='new[default]', record=InstanceRecord(state=InstanceState.DONE)
        )
    summary = next(line for line in caplog.messages if line.startswith('run finished'))
    assert '1 done' in summary


def test_non_zero_rank_logs_no_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv('LOCAL_RANK', '1')
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        pass
    assert not [line for line in caplog.messages if line.startswith('run finished')]


def test_name_conflict_with_an_earlier_execution_names_the_remedy(tmp_path: Path) -> None:
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run:
        run.claim_instance(stage_name='transform', name_parts=('a',))
        run.record_instance(
            instance_id='transform[a]', record=InstanceRecord(state=InstanceState.DONE)
        )
    with (
        Run(config=Config(), root=tmp_path, relative_path=Path('a')) as run,
        pytest.raises(ValueError, match='delete its folder and its entry in'),
    ):
        run.claim_instance(stage_name='transform', name_parts=('a', 'b'))


def test_environment_is_computed_once_and_has_git_and_versions(tmp_path: Path) -> None:
    run = Run(config=Config(), root=tmp_path, relative_path=Path('a'))
    environment = run.environment()
    assert set(environment) == {'git', 'versions'}
    assert run.environment() is environment
