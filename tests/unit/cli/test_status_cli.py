from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time

import pytest

import tslab
from tslab.cli import main
from tslab.cli.status import Status


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0
    run: tslab.RunOptions = field(default_factory=tslab.RunOptions)


@tslab.stage(name='scale', output=tslab.Json)
def scale(*, run: tslab.Run, name: tslab.InstanceName | None = None, factor: int) -> tslab.Json:
    if factor < 0:
        message = 'negative factor\nsecond line'
        raise ValueError(message)
    return tslab.Json(data={'factor': factor})


@tslab.stage(name='report', output=tslab.Json)
def report(
    *, run: tslab.Run, name: tslab.InstanceName | None = None, scaled: tslab.Json
) -> tslab.Json:
    return scaled


def make_run(*, root: Path, relative_path: Path, factors: tuple[int, ...]) -> None:
    with tslab.Run(config=Config(), root=root, relative_path=relative_path) as run:
        for factor in factors:
            scaled = scale(run=run, name=f'f{factor}', factor=factor)
            report(run=run, name=f'f{factor}', scaled=scaled)


def kill(*, directory: Path) -> None:
    """Make a finished run look like a job killed while running."""
    status_file = directory / 'status.json'
    document = json.loads(status_file.read_text(encoding='utf-8'))
    document['run_state'] = 'running'
    status_file.write_text(json.dumps(document), encoding='utf-8')
    lock = directory / '.lock'
    lock.write_text(json.dumps({'pid': 1, 'host': 'node123'}), encoding='utf-8')
    then = time.time() - 601
    os.utime(lock, times=(then, then))


def shown(
    *, caplog: pytest.LogCaptureFixture, path: Path, stage: str | None = None
) -> tuple[int, str]:
    caplog.clear()
    with caplog.at_level(logging.INFO):
        code = Status(path=path, stage=stage).run()
    return code, caplog.text


def test_one_run_lists_instances_and_stage_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1, -1))
    code, text = shown(caplog=caplog, path=tmp_path / 'a')
    assert code == 0
    assert 'scale[f-1]' in text
    assert 'ValueError: negative factor' in text
    assert 'second line' not in text
    assert 'scale: 1 done, 1 failed' in text
    assert 'report: 1 done, 1 skipped' in text


def test_stage_filter(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    _, text = shown(caplog=caplog, path=tmp_path / 'a', stage='report')
    assert 'report[f1]' in text
    assert 'scale[f1]' not in text


def test_folder_of_runs_gives_one_line_per_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_run(root=tmp_path, relative_path=Path('ts2vec') / 'seed_1', factors=(1,))
    make_run(root=tmp_path, relative_path=Path('ts2vec') / 'seed_2', factors=(1, -1))
    make_run(root=tmp_path, relative_path=Path('cost') / 'seed_1', factors=(2,))
    kill(directory=tmp_path / 'cost' / 'seed_1')
    code, text = shown(caplog=caplog, path=tmp_path)
    assert code == 0
    lines = text.splitlines()
    assert any('cost/seed_1' in line and 'interrupted' in line for line in lines)
    assert any('ts2vec/seed_2' in line and 'finished' in line for line in lines)
    assert '3 runs: 2 finished, 1 interrupted' in text


def test_one_malformed_status_file_does_not_abort_the_scan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_run(root=tmp_path, relative_path=Path('good'), factors=(1,))
    make_run(root=tmp_path, relative_path=Path('bad'), factors=(1,))
    (tmp_path / 'bad' / 'status.json').write_text('{"run_state": ', encoding='utf-8')
    code, text = shown(caplog=caplog, path=tmp_path)
    assert code == 0
    assert 'good' in text
    assert 'bad/status.json' in text
    assert '1 runs: 1 finished' in text


def test_a_malformed_run_folder_exits_1(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    (tmp_path / 'a' / 'status.json').write_text('[]', encoding='utf-8')
    code, text = shown(caplog=caplog, path=tmp_path / 'a')
    assert code == 1
    assert 'status.json' in text


def test_live_run_is_running(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with tslab.Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        _, text = shown(caplog=caplog, path=tmp_path)
    assert '1 runs: 1 running' in text


def test_nothing_to_show(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert shown(caplog=caplog, path=tmp_path)[0] == 1
    assert shown(caplog=caplog, path=tmp_path / 'missing')[0] == 2


def test_command_line_entry_point(tmp_path: Path) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    with pytest.raises(SystemExit) as raised:
        main(args=['status', str(tmp_path), '--stage', 'scale'])
    assert raised.value.code == 0
