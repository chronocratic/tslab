"""The ``tslab study logs`` command: one run, its log, and the follow loop."""

import logging
import os
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.cli.study.helpers import toy_project
from tests.unit.study.helpers import context, study_directory, submit_study
from tslab.cli.study import Logs, logs as logs_module

_ONE = ('model=small', 'seed=1')


def _run_log(project: Path, text: str) -> Path:
    path = study_directory(project=project) / 'g/small/1/logs/run.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def test_logs_prints_the_last_lines_of_the_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _run_log(project, ''.join(f'line {index}\n' for index in range(10)))

    code = Logs(study='toy_studies:study', machine='local.toml', only=_ONE, lines=3).run()

    assert code == 0
    assert capsys.readouterr().out == 'line 7\nline 8\nline 9\n'


def test_logs_lists_the_matches_of_a_wide_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.ERROR):
        code = Logs(study='toy_studies:study', machine='local.toml', only=('seed=1',)).run()

    assert code == 2
    assert 'the selection matches 2 runs' in caplog.text
    assert 'g/small/1' in caplog.text
    assert 'g/large/1' in caplog.text


def test_logs_names_a_missing_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        code = Logs(study='toy_studies:study', machine='local.toml', only=_ONE).run()

    assert code == 0
    assert 'no log yet at' in caplog.text


def test_logs_follow_prints_new_data_and_restarts_on_a_shorter_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    path = _run_log(project, 'old\n')
    writes = iter(('old\nmore\n', 'new\n'))

    def fake_sleep(_seconds: float) -> None:
        try:
            path.write_text(next(writes), encoding='utf-8')
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(logs_module, '_sleep', fake_sleep)

    code = Logs(study='toy_studies:study', machine='local.toml', only=_ONE, follow=True).run()

    assert code == 0
    assert capsys.readouterr().out == 'old\nmore\nnew\n'


def test_logs_job_reads_the_scheduler_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend: FakeBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    task = next(task for job in ctx.jobs for task in job.tasks if task.run_path == 'g/small/1')
    logs = study_directory(project=project) / 'logs'
    logs.mkdir(exist_ok=True)
    (logs / f'g1_{task.index}.out').write_text('the job of small/1\n', encoding='utf-8')

    code = Logs(study='toy_studies:study', machine='cluster.toml', only=_ONE, job=True).run()

    assert code == 0
    assert capsys.readouterr().out == 'the job of small/1\n'


def test_logs_launch_without_a_launch_log_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.ERROR):
        code = Logs(study='toy_studies:study', machine='local.toml', launch=True).run()

    assert code == 2
    assert 'no launch log in' in caplog.text


def test_logs_launch_reads_the_newest_launch_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    logs = study_directory(project=project) / '.study' / 'local_logs'
    logs.mkdir(parents=True)
    (logs / '200.log').write_text('old launch\n', encoding='utf-8')
    (logs / '100.log').write_text('new launch\n', encoding='utf-8')
    os.utime(logs / '200.log', (1, 1))

    code = Logs(study='toy_studies:study', machine='local.toml', launch=True).run()

    assert code == 0
    assert capsys.readouterr().out == 'new launch\n'
