"""The ``tslab study status`` command: the hints of its run."""

import logging
from pathlib import Path

import pytest

from tests.support.fake_slurm import install_fake_slurm
from tests.unit.cli.study.helpers import toy_project
from tests.unit.study.helpers import finish_run, submit_study
from tslab.cli.study import Status


def test_status_suggests_a_dry_run_when_nothing_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        code = Status(study='toy_studies:study', machine='local.toml').run()

    assert code == 0
    assert 'nothing ran yet; check the plan: tslab study submit' in caplog.text
    assert 'tslab study resubmit' not in caplog.text


def test_status_suggests_following_queued_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    fake = install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])

    with caplog.at_level(logging.INFO):
        Status(study='toy_studies:study', machine='cluster.toml').run()

    assert (
        'follow progress: tslab study status toy_studies:study --machine cluster.toml --watch 60'
        in caplog.text
    )


def test_status_says_all_done_and_how_to_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})
    submit_study(project=project)
    for model in ('small', 'large'):
        for seed in (1, 2):
            finish_run(project=project, model=model, seed=seed)

    with caplog.at_level(logging.INFO):
        code = Status(study='toy_studies:study', machine='local.toml').run()

    assert code == 0
    assert 'all 4 runs done; collect metrics: tslab study results' in caplog.text


def test_status_warns_when_no_run_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        Status(
            study='toy_studies:study',
            machine='local.toml',
            grid=('g',),
            only=('model=small', 'model=large'),
        ).run()

    assert any(
        record.levelno == logging.WARNING and 'no runs match the selection' in record.message
        for record in caplog.records
    )


def test_hint_repeats_selection_flags_and_quotes_machine_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch, name='project with spaces')
    machine = (project / 'local.toml').as_posix()

    with caplog.at_level(logging.INFO):
        Status(
            study='toy_studies:study', machine=machine, grid=('g',), only=('model=small,large',)
        ).run()

    assert '--grid g --only model=small,large' in caplog.text
    assert f"--machine '{machine}'" in caplog.text
