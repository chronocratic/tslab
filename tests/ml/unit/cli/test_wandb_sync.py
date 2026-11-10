import logging
import os
from pathlib import Path
import struct
import subprocess
import time
from types import SimpleNamespace
import zlib

import pytest
from wandb.proto import wandb_internal_pb2

from tslab.lab.cli import main
from tslab.ml.cli.wandb_sync import WandbSync
from tslab.ml.tracking.wandb import sync
from tslab.ml.tracking.wandb.sync import LocalRun

FILESTREAM_LOG = 'POST https://api.wandb.ai/files/team/proj/abc123/file_stream 200\n'


def make_run(*, root: Path, name: str = 'run-20261015_120000-abc123', log: str = '') -> Path:
    run_directory = root / 'trackers' / 'wandb' / name
    (run_directory / 'logs').mkdir(parents=True)
    (run_directory / f'{name.rsplit("-", maxsplit=1)[-1]}.wandb').write_bytes(b'')
    (run_directory / 'logs' / 'debug-internal.log').write_text(log, encoding='utf-8')
    return run_directory


def write_transaction_log(*, path: Path, entity: str, project: str, run_id: str) -> None:
    """Write a minimal ``.wandb`` log with one run record (format: tslab.ml.tracking.wandb.log)."""
    # why: protobuf class is invisible to type checkers
    record = getattr(wandb_internal_pb2, 'Record')()  # noqa: B009
    record.run.entity = entity
    record.run.project = project
    record.run.run_id = run_id
    payload = record.SerializeToString()
    full = 1
    checksum = zlib.crc32(bytes([full]) + payload)
    header = struct.pack('<4sHB', b':W&B', 0xBEE1, 0)
    path.write_bytes(header + struct.pack('<IHB', checksum, len(payload), full) + payload)


def make_stale(*, run_directory: Path) -> None:
    old = time.time() - 3600
    for path in run_directory.glob('*.wandb'):
        os.utime(path, (old, old))


def test_environment_fills_gaps_but_never_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_directory = make_run(root=tmp_path)
    write_transaction_log(
        path=run_directory / 'abc123.wandb', entity='', project='proj', run_id='abc123'
    )
    monkeypatch.setenv('WANDB_ENTITY', 'envteam')
    monkeypatch.setenv('WANDB_PROJECT', 'another-project')
    queried: list[str] = []

    def fake_run(_pool: object, *, path: str, fresh: bool) -> SimpleNamespace:
        queried.append(path)
        return SimpleNamespace(state='finished')

    monkeypatch.setattr(
        sync._ApiPool,  # noqa: SLF001 # why: replaces the network boundary
        'run',
        fake_run,
    )
    assert WandbSync(root=tmp_path, dry_run=True).run(sleep=no_sleep) == 0
    assert queried == ['envteam/proj/abc123']


class FakeServer:
    """Server states per call; records the ``wandb sync`` commands."""

    def __init__(self, *, states: list[str], sync_output: str = 'Syncing... done') -> None:
        self.states = states
        self.sync_output = sync_output
        self.commands: list[list[str]] = []

    def state(self, *, run: LocalRun, fresh: bool = False) -> str:
        return self.states.pop(0)

    def run(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=self.sync_output, stderr=''
        )


def install(*, monkeypatch: pytest.MonkeyPatch, server: FakeServer, exit_codes: list[int]) -> None:
    def fake_state(_query: object, *, run: LocalRun, fresh: bool = False) -> str:
        return server.state(run=run, fresh=fresh)

    def fake_exit_codes(*, path: Path) -> list[int]:
        return exit_codes

    # Replace the network and subprocess boundaries; everything else runs for real.
    monkeypatch.setattr(
        sync._ServerQuery,  # noqa: SLF001 # why: the server query is private
        'state',
        fake_state,
    )
    monkeypatch.setattr(sync.subprocess, 'run', server.run)
    monkeypatch.setattr(sync, 'read_exit_codes', fake_exit_codes)


def finished_crashed_run(*, root: Path) -> Path:
    run_directory = make_run(root=root, log=FILESTREAM_LOG + 'finishing run\ngot exitcode: 0\n')
    make_stale(run_directory=run_directory)
    return run_directory


def no_sleep(_seconds: float) -> None:
    return None


def test_resync_is_verified_on_the_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = finished_crashed_run(root=tmp_path)
    server = FakeServer(states=['crashed', 'crashed', 'finished'])
    install(monkeypatch=monkeypatch, server=server, exit_codes=[0])
    assert WandbSync(root=tmp_path).run(sleep=no_sleep) == 0
    (command,) = server.commands
    assert command[-3:] == ['--no-skip-online', '--no-skip-synced', str(run_directory)]


def test_resync_fails_when_the_server_state_does_not_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished_crashed_run(root=tmp_path)
    install(monkeypatch=monkeypatch, server=FakeServer(states=['crashed'] * 4), exit_codes=[0])
    assert WandbSync(root=tmp_path).run(sleep=no_sleep) == 1


def test_resync_fails_when_wandb_sync_uploads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished_crashed_run(root=tmp_path)
    server = FakeServer(states=['crashed'], sync_output='No runs to sync')
    install(monkeypatch=monkeypatch, server=server, exit_codes=[0])
    assert WandbSync(root=tmp_path).run(sleep=no_sleep) == 1


def test_non_zero_exit_record_is_a_data_only_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    finished_crashed_run(root=tmp_path)
    # A clean exit followed by 130: SLURM interrupted teardown, so the state cannot flip.
    install(monkeypatch=monkeypatch, server=FakeServer(states=['crashed']), exit_codes=[0, 130])
    with caplog.at_level(logging.INFO):
        assert WandbSync(root=tmp_path).run(sleep=no_sleep) == 0
    assert 'data-only exit=130' in caplog.text


def test_dry_run_never_syncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    finished_crashed_run(root=tmp_path)
    server = FakeServer(states=['crashed'])
    install(monkeypatch=monkeypatch, server=server, exit_codes=[0])
    assert WandbSync(root=tmp_path, dry_run=True).run(sleep=no_sleep) == 0
    assert server.commands == []


def test_offline_finished_run_syncs_without_a_server_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An offline run has no server state to flip; a clean resync is the whole outcome.
    run_directory = make_run(
        root=tmp_path,
        name='offline-run-20261015_120000-off1',
        log='finishing run\ngot exitcode: 0\n',
    )
    server = FakeServer(states=['offline'] * 4)
    install(monkeypatch=monkeypatch, server=server, exit_codes=[0])
    with caplog.at_level(logging.INFO):
        assert WandbSync(root=tmp_path, include_offline=True).run(sleep=no_sleep) == 0
    (command,) = server.commands
    assert command[-1] == str(run_directory)
    assert 'synced' in caplog.text


def test_command_line_entry_point(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['wandb-sync', str(tmp_path), '--dry-run'])
    assert raised.value.code == 0
