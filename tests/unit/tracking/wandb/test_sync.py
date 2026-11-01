import os
from pathlib import Path
import struct
import time
import zlib

import pytest

from tslab.tracking.wandb.sync import (
    Action,
    discover_runs,
    LocalRun,
    LocalState,
    parse_run_url,
    plan_action,
    resolve_run_path,
)
from wandb.proto import wandb_internal_pb2

TARGETS = frozenset({'crashed', 'failed', 'killed'})
FILESTREAM_LOG = 'POST https://api.wandb.ai/files/team/proj/abc123/file_stream 200\n'


def make_run(*, root: Path, name: str = 'run-20261015_120000-abc123', log: str = '') -> Path:
    run_directory = root / 'trackers' / 'wandb' / name
    (run_directory / 'logs').mkdir(parents=True)
    (run_directory / f'{name.rsplit("-", maxsplit=1)[-1]}.wandb').write_bytes(b'')
    (run_directory / 'logs' / 'debug-internal.log').write_text(log, encoding='utf-8')
    return run_directory


def write_transaction_log(*, path: Path, entity: str, project: str, run_id: str) -> None:
    """Write a minimal ``.wandb`` log with one run record (format: tslab.tracking.wandb.log)."""
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


def test_parse_run_url_reads_both_url_forms() -> None:
    assert parse_run_url(text=FILESTREAM_LOG) == ('team', 'proj', 'abc123')
    assert parse_run_url(text='see https://wandb.ai/team/proj/runs/xyz?x=1') == (
        'team',
        'proj',
        'xyz',
    )
    assert parse_run_url(text='nothing here') is None


def test_discover_runs_falls_back_to_the_transaction_log(tmp_path: Path) -> None:
    # The run crashed before its debug logs showed a URL; the run record still names the project.
    run_directory = make_run(root=tmp_path)
    write_transaction_log(
        path=run_directory / 'abc123.wandb', entity='', project='proj', run_id='abc123'
    )
    (run,) = discover_runs(root=tmp_path)
    assert (run.entity, run.project, run.run_id) == (None, 'proj', 'abc123')


def test_resolve_run_path_prefers_the_run_and_fills_gaps(tmp_path: Path) -> None:
    online = tmp_path / 'run-20261015_120000-abc123'
    complete = LocalRun(run_directory=online, entity='team', project='proj', run_id='abc123')
    assert resolve_run_path(run=complete, entity='x', project='y') == 'team/proj/abc123'
    partial = LocalRun(run_directory=online, entity=None, project='proj', run_id='abc123')
    assert resolve_run_path(run=partial, entity='x', project='y') == 'x/proj/abc123'
    assert resolve_run_path(run=partial, entity=None, project='y') is None
    offline = LocalRun(
        run_directory=tmp_path / 'offline-run-20261015_120000-abc123',
        entity='team',
        project='proj',
        run_id='abc123',
    )
    assert resolve_run_path(run=offline, entity='x', project='y') is None


def test_discover_runs_reads_coordinates_and_skips_latest_run(tmp_path: Path) -> None:
    make_run(root=tmp_path / 'a', log=FILESTREAM_LOG)
    make_run(root=tmp_path / 'b', name='offline-run-20261015_120000-off1')
    make_run(root=tmp_path / 'c', name='latest-run')
    runs = discover_runs(root=tmp_path)
    assert [(run.entity, run.project, run.run_id) for run in runs] == [
        ('team', 'proj', 'abc123'),
        (None, None, 'off1'),
    ]


@pytest.mark.parametrize(
    ('log', 'stale', 'expected'),
    [
        ('finishing run\ngot exitcode: 0\n', True, LocalState.FINISHED),
        ('finishing run\ngot exitcode: 1\n', True, LocalState.EXIT_ERROR),
        ('', False, LocalState.RUNNING),
        ('', True, LocalState.NO_EXIT),
    ],
)
def test_local_state(tmp_path: Path, *, log: str, stale: bool, expected: LocalState) -> None:
    run_directory = make_run(root=tmp_path, log=log)
    if stale:
        make_stale(run_directory=run_directory)
    run = LocalRun(run_directory=run_directory, entity=None, project=None, run_id='abc123')
    assert run.local_state() is expected


@pytest.mark.parametrize(
    ('local', 'state', 'dry_run', 'expected'),
    [
        (LocalState.FINISHED, 'finished', False, (Action.SKIP, '')),
        (LocalState.RUNNING, 'crashed', False, (Action.SKIP, ' live')),
        (LocalState.FINISHED, 'crashed', False, (Action.SYNC, '')),
        (LocalState.NO_EXIT, 'crashed', False, (Action.SYNC, ' data-only')),
        (LocalState.FINISHED, 'crashed', True, (Action.WOULD, '')),
    ],
)
def test_plan_action(
    *, local: LocalState, state: str, dry_run: bool, expected: tuple[Action, str]
) -> None:
    assert plan_action(local=local, state=state, target_states=TARGETS, dry_run=dry_run) == (
        expected
    )
