import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import types

import pytest

from tslab.jobs.local import busy_gpus, gpu_lock_folder, select_gpus, try_lease_gpu, visible_gpus

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


def _free_gpu_zero_holder(process: subprocess.Popen[bytes]) -> None:
    process.kill()
    process.wait()


def _fake_binary(*, tmp_path: Path, name: str, body: str) -> Path:
    binary = tmp_path / 'bin' / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(body)
    binary.chmod(0o755)
    return binary


def test_lock_folder_uses_xdg_runtime_dir(*, tmp_path: Path) -> None:
    environment = {**os.environ, 'XDG_RUNTIME_DIR': tmp_path.as_posix()}
    folder = gpu_lock_folder(name='gpus', environment=environment)
    assert folder == tmp_path / 'gpus'
    assert folder.stat().st_mode & 0o777 == 0o700


def test_lock_folder_fallback_holds_uid(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {key: value for key, value in os.environ.items() if key != 'XDG_RUNTIME_DIR'}
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    folder = gpu_lock_folder(name='gpus', environment=environment)
    assert folder == tmp_path / f'gpus-{os.getuid()}'


def test_lock_folder_owned_by_other_user_fails(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        os, 'stat', lambda *_args, **_kwargs: types.SimpleNamespace(st_uid=os.getuid() + 1)
    )
    with pytest.raises(PermissionError) as excinfo:
        gpu_lock_folder(name='gpus', environment={'XDG_RUNTIME_DIR': tmp_path.as_posix()})
    assert (tmp_path / 'gpus').as_posix() in str(excinfo.value)


def test_lease_is_exclusive_across_processes(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    holder = _hold_gpu_zero(folder=folder)
    try:
        assert try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=1) is None
        lease = try_lease_gpu(folder=folder, gpu_ids=['0', '1'], tasks_per_gpu=1)
        assert lease is not None
        assert lease.gpu_id == '1'
        lease.release()
    finally:
        _free_gpu_zero_holder(holder)


def test_lease_freed_when_holder_killed(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    holder = _hold_gpu_zero(folder=folder)
    _free_gpu_zero_holder(holder)
    lease = try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=1)
    assert lease is not None
    assert lease.gpu_id == '0'
    lease.release()


def test_shared_leases_capped_at_tasks_per_gpu(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    first = try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=2)
    second = try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=2)
    assert first is not None
    assert second is not None
    assert try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=2) is None
    first.release()
    second.release()


def test_exclusive_and_shared_exclude_each_other(*, tmp_path: Path) -> None:
    folder = tmp_path / 'locks'
    folder.mkdir()
    exclusive = try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=1)
    assert exclusive is not None
    assert try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=2) is None
    exclusive.release()
    shared = try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=2)
    assert shared is not None
    assert try_lease_gpu(folder=folder, gpu_ids=['0'], tasks_per_gpu=1) is None
    shared.release()


def test_visible_gpus_from_cuda_visible_devices() -> None:
    assert visible_gpus(environment={'CUDA_VISIBLE_DEVICES': '1,3'}) == ('1', '3')
    assert visible_gpus(environment={'CUDA_VISIBLE_DEVICES': ''}) == ()
    with pytest.raises(ValueError, match='gpus'):
        visible_gpus(environment={'CUDA_VISIBLE_DEVICES': 'GPU-abc'})


def test_visible_gpus_from_nvidia_smi(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body='#!/bin/sh\necho 0\necho 1\n')
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert visible_gpus(environment={}) == ('0', '1')


def test_visible_gpus_without_nvidia_smi(*, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', '/nonexistent-bin-dir')
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    assert visible_gpus(environment={}) == ()
    monkeypatch.setattr(platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    assert visible_gpus(environment={}) == ('mps',)


def test_visible_gpus_failing_nvidia_smi_warns(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body='#!/bin/sh\nexit 1\n')
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert visible_gpus(environment={}) == ()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_busy_gpus_maps_uuids(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        '#!/bin/sh\n'
        'if [ "$1" = "--query-compute-apps=gpu_uuid" ]; then\n'
        "  echo 'GPU-BUSY-1'\n"
        'else\n'
        "  echo '0,GPU-BUSY-0'\n"
        "  echo '1,GPU-BUSY-1'\n"
        'fi\n'
    )
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body=body)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert busy_gpus(gpu_ids=('0', '1')) == frozenset({'1'})


def _memory_fake(*, tmp_path: Path) -> None:
    body = (
        '#!/bin/sh\n'
        'if [ "$1" = "--query-compute-apps=gpu_uuid" ]; then\n'
        '  : # no compute apps\n'
        'else\n'
        "  echo '0,GPU-0,30000,40000'\n"
        "  echo '1,GPU-1,10,40000'\n"
        'fi\n'
    )
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body=body)


def test_busy_by_memory(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _memory_fake(tmp_path=tmp_path)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert busy_gpus(gpu_ids=('0', '1'), memory_fraction=0.1) == frozenset({'0'})


def test_memory_rule_off_at_one(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _memory_fake(tmp_path=tmp_path)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert busy_gpus(gpu_ids=('0', '1'), memory_fraction=1.0) == frozenset()


def test_rows_without_memory_use_compute_only(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '#!/bin/sh\n'
        'if [ "$1" = "--query-compute-apps=gpu_uuid" ]; then\n'
        "  echo 'GPU-BUSY-1'\n"
        'else\n'
        "  echo '0,GPU-BUSY-0'\n"
        "  echo '1,GPU-BUSY-1'\n"
        'fi\n'
    )
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body=body)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert busy_gpus(gpu_ids=('0', '1'), memory_fraction=0.1) == frozenset({'1'})


def test_memory_query_failure_keeps_the_compute_rule(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '#!/bin/sh\n'
        'if [ "$1" = "--query-compute-apps=gpu_uuid" ]; then\n'
        "  echo 'GPU-BUSY-1'\n"
        'elif [ "$1" = "--query-gpu=index,uuid,memory.used,memory.total" ]; then\n'
        '  exit 1\n'
        'else\n'
        "  echo '0,GPU-BUSY-0'\n"
        "  echo '1,GPU-BUSY-1'\n"
        'fi\n'
    )
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body=body)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert busy_gpus(gpu_ids=('0', '1'), memory_fraction=0.1) == frozenset({'1'})


def test_select_gpus_auto_uses_visible_gpus() -> None:
    environment = {'CUDA_VISIBLE_DEVICES': '1,3'}
    assert select_gpus(gpus='auto', skip_busy=False, environment=environment) == ('1', '3')


def test_select_gpus_keeps_the_given_ids_in_order() -> None:
    assert select_gpus(gpus=('1', '0'), skip_busy=False, environment={}) == ('1', '0')


def test_select_gpus_skip_busy_leaves_busy_out(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '#!/bin/sh\n'
        'if [ "$1" = "--query-compute-apps=gpu_uuid" ]; then\n'
        "  echo 'GPU-BUSY-1'\n"
        'else\n'
        "  echo '0,GPU-BUSY-0'\n"
        "  echo '1,GPU-BUSY-1'\n"
        'fi\n'
    )
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body=body)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    assert select_gpus(gpus=('0', '1'), skip_busy=True, environment={}) == ('0',)


def test_select_gpus_all_busy_fails(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        '#!/bin/sh\n'
        'if [ "$1" = "--query-compute-apps=gpu_uuid" ]; then\n'
        "  echo 'GPU-BUSY-1'\n"
        'else\n'
        "  echo '1,GPU-BUSY-1'\n"
        'fi\n'
    )
    _fake_binary(tmp_path=tmp_path, name='nvidia-smi', body=body)
    monkeypatch.setenv('PATH', f'{tmp_path / "bin"}:{os.environ["PATH"]}')
    with pytest.raises(ValueError, match='all local GPUs are busy: 1'):
        select_gpus(gpus=('1',), skip_busy=True, environment={})


def test_select_gpus_skip_busy_without_nvidia_gpus_fails(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('PATH', '/nonexistent-bin-dir')
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    with pytest.raises(ValueError, match='needs NVIDIA GPUs'):
        select_gpus(gpus='auto', skip_busy=True, environment={'CUDA_VISIBLE_DEVICES': ''})
    with pytest.raises(ValueError, match='needs NVIDIA GPUs'):
        select_gpus(gpus=('mps',), skip_busy=True, environment={})
