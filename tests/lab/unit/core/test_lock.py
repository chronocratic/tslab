from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import time

import pytest

from tslab.lab.core import lock as lock_module
from tslab.lab.core.lock import acquire_lock, Heartbeat, is_run_active, release_lock, RunLockedError
from tslab.lab.core.run import Run, RunOptions


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 7
    run: RunOptions = field(default_factory=RunOptions)


def _age_file(*, path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, times=(then, then))


def test_lock_from_other_host_is_respected(tmp_path: Path) -> None:
    (tmp_path / '.lock').write_text(json.dumps({'pid': 1, 'host': 'elsewhere'}), encoding='utf-8')
    with pytest.raises(RunLockedError):
        acquire_lock(path=tmp_path / '.lock')


def test_is_run_active(tmp_path: Path) -> None:
    assert not is_run_active(directory=tmp_path)
    with Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        assert is_run_active(directory=tmp_path / 'a')
    lock = tmp_path / '.lock'
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123'}), encoding='utf-8')
    assert is_run_active(directory=tmp_path)
    _age_file(path=lock, seconds=601)
    assert not is_run_active(directory=tmp_path)


def test_concurrent_takeover_of_a_stale_lock_has_one_winner(tmp_path: Path) -> None:
    lock = tmp_path / '.lock'
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123'}), encoding='utf-8')
    _age_file(path=lock, seconds=601)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def contender() -> None:
        barrier.wait()
        try:
            acquire_lock(path=lock)
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
    mine = acquire_lock(path=lock)
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123', 'nonce': 'theirs'}), encoding='utf-8')
    release_lock(path=lock, nonce=mine)
    assert lock.exists()
    release_lock(path=lock, nonce='theirs')
    assert not lock.exists()


def test_heartbeat_survives_one_unreadable_lock(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lock_module, '_HEARTBEAT_SECONDS', 0.02)
    path = tmp_path / 'lock'
    nonce = acquire_lock(path=path)
    good = path.read_text(encoding='utf-8')
    path.write_text('{broken', encoding='utf-8')
    beat = Heartbeat(path=path, nonce=nonce)
    try:
        time.sleep(0.2)
        path.write_text(good, encoding='utf-8')
        _age_file(path=path, seconds=1000)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and time.time() - path.stat().st_mtime > 100:
            time.sleep(0.02)
        assert time.time() - path.stat().st_mtime < 100
    finally:
        beat.stop()
