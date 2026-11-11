"""The ``tslab study slurm`` command and the ``slurm_output`` notebook function."""

import logging
from pathlib import Path

import pytest

from tests.lab.unit.cli.study.helpers import toy_project
from tests.lab.unit.study.helpers import submit_study
from tests.support.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.lab.cli.study.slurm import Slurm
from tslab.lab.jobs.slurm import SLURM_COMMANDS
from tslab.lab.study import slurm_output
from tslab.lab.study.definition.loading import load_study


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


def test_lists_commands_without_a_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert Slurm(study='toy_studies:study', machine='cluster.toml').run() == 0
    out = capsys.readouterr().out
    for command in SLURM_COMMANDS:
        assert command.name in out
        assert command.description in out


def test_runs_one_command_and_passes_output_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeSlurm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    submit_study(project=project)
    fake.set_answers('sacct', [{'stdout': '1000_0|RUNNING\n'}])
    fake.set_answers('squeue', [{'stdout': 'the queue\n'}])
    code = Slurm(study='toy_studies:study', name='squeue-full-name', machine='cluster.toml').run()
    assert code == 0
    assert capsys.readouterr().out == 'the queue\n'


def test_unknown_name_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    with caplog.at_level(logging.ERROR):
        code = Slurm(study='toy_studies:study', name='squeue', machine='cluster.toml').run()
    assert code == 2
    assert 'squeue-full-name' in caplog.text


def test_no_recorded_jobs_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeSlurm,
    caplog: pytest.LogCaptureFixture,
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    with caplog.at_level(logging.ERROR):
        code = Slurm(study='toy_studies:study', name='sacct-usage', machine='cluster.toml').run()
    assert code == 1
    assert 'no SLURM jobs recorded' in caplog.text


def test_missing_tool_names_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeSlurm,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    submit_study(project=project)
    (tmp_path / 'slurm' / 'seff').unlink()
    code = Slurm(study='toy_studies:study', name='seff-all', machine='cluster.toml').run()
    assert code == 127
    assert 'seff is not installed' in capsys.readouterr().err


def test_slurm_output_returns_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeSlurm
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    submit_study(project=project)
    fake.set_answers('sacct', [{'stdout': 'the failed jobs\n'}])
    text = slurm_output(
        study=load_study(reference='toy_studies:study'),
        machine=project / 'cluster.toml',
        name='sacct-failed',
        environment={},
    )
    assert text == 'the failed jobs\n'


def test_slurm_output_rejects_unknown_name_and_unsubmitted_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeSlurm
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    study = load_study(reference='toy_studies:study')
    with pytest.raises(ValueError, match='squeue-full-name'):
        slurm_output(study=study, machine=project / 'cluster.toml', name='x', environment={})
    with pytest.raises(ValueError, match='no SLURM jobs recorded'):
        slurm_output(
            study=study, machine=project / 'cluster.toml', name='sacct-usage', environment={}
        )


def test_failing_command_reaches_the_exit_code_and_the_notebook_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeSlurm
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    submit_study(project=project)
    fake.set_answers('sacct', [{'exit': 1, 'stderr': 'boom'}])
    code = Slurm(study='toy_studies:study', name='sacct-failed', machine='cluster.toml').run()
    assert code == 1
    with pytest.raises(RuntimeError, match='boom'):
        slurm_output(
            study=load_study(reference='toy_studies:study'),
            machine=project / 'cluster.toml',
            name='sacct-failed',
            environment={},
        )
