from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import cast

import pytest

import tslab.lab
from tslab.lab.cli import main
import tslab.lab.cli.status as status_module
from tslab.lab.cli.status import Status


@dataclass(frozen=True, kw_only=True)
class Config:
    seed: int = 0
    run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)


@tslab.lab.stage(name='scale', output=tslab.lab.Json)
def scale(
    *, run: tslab.lab.Run, name: tslab.lab.InstanceName | None = None, factor: int
) -> tslab.lab.Json:
    if factor < 0:
        message = 'negative factor\nsecond line'
        raise ValueError(message)
    return tslab.lab.Json(data={'factor': factor})


@tslab.lab.stage(name='report', output=tslab.lab.Json)
def report(
    *, run: tslab.lab.Run, name: tslab.lab.InstanceName | None = None, scaled: tslab.lab.Json
) -> tslab.lab.Json:
    return scaled


def make_run(*, root: Path, relative_path: Path, factors: tuple[int, ...]) -> None:
    with tslab.lab.Run(config=Config(), root=root, relative_path=relative_path) as run:
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
    assert '1 run: 1 finished' in text


def test_a_malformed_run_folder_exits_1(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    (tmp_path / 'a' / 'status.json').write_text('[]', encoding='utf-8')
    code, text = shown(caplog=caplog, path=tmp_path / 'a')
    assert code == 1
    assert 'status.json' in text


def test_live_run_is_running(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with tslab.lab.Run(config=Config(), root=tmp_path, relative_path=Path('a')):
        _, text = shown(caplog=caplog, path=tmp_path)
    assert '1 run: 1 running' in text
    assert 'follow active runs: tslab status' in text
    assert '--watch 60' in text


def test_nothing_to_show(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert shown(caplog=caplog, path=tmp_path)[0] == 1
    assert shown(caplog=caplog, path=tmp_path / 'missing')[0] == 2


def test_folder_status_exit_code_maps_failed_to_4(tmp_path: Path) -> None:
    make_run(root=tmp_path, relative_path=Path('failed'), factors=(-1,))
    assert Status(path=tmp_path, exit_code=True).run() == 4


def test_folder_status_defaults_to_today_codes(tmp_path: Path) -> None:
    make_run(root=tmp_path, relative_path=Path('failed'), factors=(-1,))
    assert Status(path=tmp_path).run() == 0


def test_folder_status_json_lists_instances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(root=tmp_path, relative_path=Path('failed'), factors=(1, -1))
    assert Status(path=tmp_path / 'failed', json=True).run() == 0
    document = json.loads(capsys.readouterr().out)
    runs = cast('list[dict[str, object]]', document['runs'])
    assert document['schema'] == 'tslab.run-status/1'
    assert document['exit_code'] == 4
    assert runs[0]['directory'] == (tmp_path / 'failed').resolve().as_posix()
    assert runs[0]['state'] == 'failed'
    instances = cast('list[dict[str, object]]', runs[0]['instances'])
    by_name = {str(instance['instance']): instance for instance in instances}
    assert by_name['scale[f1]']['stage'] == 'scale'
    assert by_name['scale[f1]']['state'] == 'done'
    assert by_name['scale[f-1]']['error'] == 'ValueError: negative factor\nsecond line'


def test_json_over_a_folder_reads_each_status_once_and_each_lock_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    make_run(root=tmp_path, relative_path=Path('b'), factors=(1,))
    kill(directory=tmp_path / 'b')
    reads: list[Path] = []
    locks: list[Path] = []
    real_read = status_module.read_status
    real_active = status_module.is_run_active

    def counting_read(*, path: Path) -> object:
        reads.append(path)
        return real_read(path=path)

    def counting_active(*, directory: Path) -> bool:
        locks.append(directory)
        return real_active(directory=directory)

    monkeypatch.setattr(status_module, 'read_status', counting_read)
    monkeypatch.setattr(status_module, 'is_run_active', counting_active)
    Status(path=tmp_path, json=True).run()
    capsys.readouterr()
    assert [reads.count(tmp_path / name / 'status.json') for name in 'ab'] == [1, 1]
    assert locks == [tmp_path / 'a', tmp_path / 'b']


def test_the_lock_is_read_for_a_failed_run_that_is_being_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    make_run(root=tmp_path, relative_path=Path('b'), factors=(1,))
    status_file = tmp_path / 'b' / 'status.json'
    document = json.loads(status_file.read_text(encoding='utf-8'))
    document['run_state'] = 'failed'
    status_file.write_text(json.dumps(document), encoding='utf-8')
    locks: list[Path] = []

    def live(*, directory: Path) -> bool:
        locks.append(directory)
        return True

    monkeypatch.setattr(status_module, 'is_run_active', live)
    Status(path=tmp_path, json=True).run()
    document = json.loads(capsys.readouterr().out)
    states = {Path(run['directory']).name: run['state'] for run in document['runs']}
    assert locks == [tmp_path / 'a', tmp_path / 'b']
    assert states == {'a': 'running', 'b': 'running'}


def test_a_finished_run_that_runs_again_is_shown_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))

    def live(*, directory: Path) -> bool:
        del directory
        return True

    monkeypatch.setattr(status_module, 'is_run_active', live)
    _, text = shown(caplog=caplog, path=tmp_path)
    assert '1 run: 1 running' in text
    _, text = shown(caplog=caplog, path=tmp_path / 'a')
    assert f'{(tmp_path / "a").as_posix()}: running' in text


@pytest.mark.parametrize('saved_state', ['failed', 'crashed'])
def test_a_failed_or_crashed_run_with_a_live_lock_is_shown_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    saved_state: str,
) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    status_file = tmp_path / 'a' / 'status.json'
    document = json.loads(status_file.read_text(encoding='utf-8'))
    document['run_state'] = saved_state
    status_file.write_text(json.dumps(document), encoding='utf-8')

    def live(*, directory: Path) -> bool:
        del directory
        return True

    monkeypatch.setattr(status_module, 'is_run_active', live)
    _, text = shown(caplog=caplog, path=tmp_path)
    assert '1 run: 1 running' in text
    _, text = shown(caplog=caplog, path=tmp_path / 'a')
    assert f'{(tmp_path / "a").as_posix()}: running' in text


def test_the_scan_does_not_descend_below_a_run(tmp_path: Path) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    nested = tmp_path / 'a' / 'scale' / 'inner'
    nested.mkdir(parents=True, exist_ok=True)
    (nested / 'status.json').write_text('not json', encoding='utf-8')
    assert status_module._run_folders(root=tmp_path) == [tmp_path / 'a']  # noqa: SLF001 # why: the scan is the unit


def test_folder_status_rejects_watch_with_json(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['status', str(tmp_path), '--watch', '10', '--json'])
    assert raised.value.code == 2


def test_failed_instance_points_at_its_error_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = tmp_path / 'a'
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1, -1))

    code, text = shown(caplog=caplog, path=run_dir)

    assert code == 0
    assert 'failed; full error:' in text
    assert 'error.txt' in text


def test_interrupted_run_says_how_to_resume(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir = tmp_path / 'a'
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    kill(directory=run_dir)

    _, text = shown(caplog=caplog, path=run_dir)

    assert 'run the same script again to resume it' in text


def test_finished_run_says_all_done(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    run_dir = tmp_path / 'a'
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1, 2))

    _, text = shown(caplog=caplog, path=run_dir)

    assert 'all 4 instances done' in text


def test_folder_of_runs_with_failures_suggests_one_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_run(root=tmp_path, relative_path=Path('good'), factors=(1,))
    make_run(root=tmp_path, relative_path=Path('bad'), factors=(1, -1))

    _, text = shown(caplog=caplog, path=tmp_path)

    assert 'details of one run: tslab status' in text


def test_empty_folder_warns_and_explains(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    code, text = shown(caplog=caplog, path=tmp_path)

    assert code == 1
    assert any(
        record.levelno == logging.WARNING and 'no runs below' in record.message
        for record in caplog.records
    )
    assert 'holds status.json' in text


def test_watch_refreshes_until_interrupted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / 'a'
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))

    def interrupt(seconds: float) -> None:
        assert seconds == 5
        raise KeyboardInterrupt

    monkeypatch.setattr('time.sleep', interrupt)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        code = Status(path=run_dir, watch=5).run()

    assert code == 0
    assert caplog.text.count('scale[f1]') == 1


def test_command_line_entry_point(tmp_path: Path) -> None:
    make_run(root=tmp_path, relative_path=Path('a'), factors=(1,))
    with pytest.raises(SystemExit) as raised:
        main(args=['status', str(tmp_path), '--stage', 'scale'])
    assert raised.value.code == 0


def test_status_reads_runs_from_tslab_toml_in_a_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'tslab.toml').write_text('runs = "runs"\n', encoding='utf-8')
    runs = project / 'runs'
    runs.mkdir()
    make_run(root=runs, relative_path=Path('ts2vec') / 'seed_1', factors=(1,))
    source = project / 'src'
    source.mkdir()
    monkeypatch.chdir(source)
    caplog.clear()
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as raised:
        main(args=['status'])
    assert raised.value.code == 0
    assert 'ts2vec/seed_1' in caplog.text
    line = f'runs folder {runs.as_posix()} from {(project / "tslab.toml").as_posix()}'
    assert line in caplog.text


def test_status_without_folder_or_setting_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    caplog.clear()
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as raised:
        main(args=['status'])
    assert raised.value.code == 2
    assert 'give a FOLDER, or set runs in tslab.toml' in caplog.text


def test_folder_argument_wins_over_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    project = tmp_path / 'project'
    runs = project / 'runs'
    runs.mkdir(parents=True)
    (project / 'tslab.toml').write_text('runs = "runs"\n', encoding='utf-8')
    make_run(root=runs, relative_path=Path('from_runs'), factors=(1,))
    other = tmp_path / 'other'
    make_run(root=other, relative_path=Path('from_arg'), factors=(1,))
    monkeypatch.chdir(project)
    caplog.clear()
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as raised:
        main(args=['status', str(other)])
    assert raised.value.code == 0
    assert 'from_arg' in caplog.text
    assert 'from_runs' not in caplog.text


def test_status_broken_tslab_toml_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / 'tslab.toml').write_text('runs = 3\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    caplog.clear()
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as raised:
        main(args=['status'])
    assert raised.value.code == 2
    assert 'tslab.toml' in caplog.text
