"""The ``tslab study`` command group: parsing, filters and pass-through."""

from collections.abc import Iterator
import logging
from pathlib import Path
import sys

import pytest

from tests.unit.study.fake_slurm import install_fake_slurm
from tests.unit.study.helpers import finish_run, submit_study
from tests.unit.study.project import write_toy_project
from tslab.cli import main
from tslab.cli.study import Resubmit, Results, Status, Submit


@pytest.fixture(autouse=True)
def _clean_toy_imports() -> Iterator[None]:
    path = list(sys.path)
    yield
    for name in ('toy_studies', 'toy_script'):
        sys.modules.pop(name, None)
    sys.path[:] = path


def test_cli_parses_filters_and_pass_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Submit] = []

    def run(self: Submit) -> int:
        seen.append(self)
        return 0

    monkeypatch.setattr(Submit, 'run', run)
    with pytest.raises(SystemExit) as exit_info:
        main(
            args=[
                'study',
                'submit',
                'toy_studies:study',
                '--machine',
                str(tmp_path / 'm.toml'),
                '--only',
                'recipe=a,b',
                '--only',
                'seed=42',
                '--grid',
                'g',
                '--dry-run',
                '--',
                '--run.rerun-from',
                'evaluate',
            ]
        )
    assert exit_info.value.code == 0
    assert len(seen) == 1
    assert seen[0].study == 'toy_studies:study'
    assert seen[0].machine == str(tmp_path / 'm.toml')
    assert seen[0].only == ('recipe=a,b', 'seed=42')
    assert seen[0].grid == ('g',)
    assert seen[0].include_done is False
    assert seen[0].dry_run is True
    assert seen[0].extra_args == ('--run.rerun-from', 'evaluate')


def test_pass_through_needs_a_command_that_takes_it(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    # why: the arguments before `--` are valid, so only the guard can reject the call
    with pytest.raises(SystemExit) as exit_info:
        main(args=['status', str(tmp_path), '--', '--x'])
    assert exit_info.value.code == 2
    assert 'does not take arguments after --' in caplog.text


def _toy_project(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = 'project') -> Path:
    project = write_toy_project(project=tmp_path / name)
    monkeypatch.chdir(project)
    return project


def test_status_suggests_resubmit_for_missing_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        code = Status(study='toy_studies:study', machine='local.toml').run()

    assert code == 0
    assert 'retry interrupted and missing runs: tslab study resubmit' in caplog.text


def test_status_suggests_following_queued_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    project = _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
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
    project = _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
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
    _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

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
    project = _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch, name='project with spaces')
    machine = (project / 'local.toml').as_posix()

    with caplog.at_level(logging.INFO):
        Status(
            study='toy_studies:study', machine=machine, grid=('g',), only=('model=small,large',)
        ).run()

    assert '--grid g --only model=small,large' in caplog.text
    assert f"--machine '{machine}'" in caplog.text


def test_dry_run_says_nothing_was_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with caplog.at_level(logging.INFO):
        code = Submit(study='toy_studies:study', machine='local.toml', dry_run=True).run()

    assert code == 0
    assert 'dry run: nothing submitted' in caplog.text


def test_submit_resubmit_and_results_warn_for_empty_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    commands = (
        Submit(
            study='toy_studies:study',
            machine='local.toml',
            grid=('g',),
            only=('model=small', 'model=large'),
            dry_run=True,
        ),
        Resubmit(
            study='toy_studies:study',
            machine='local.toml',
            grid=('g',),
            only=('model=small', 'model=large'),
            dry_run=True,
        ),
        Results(
            study='toy_studies:study',
            machine='local.toml',
            grid=('g',),
            only=('model=small', 'model=large'),
            out=tmp_path / 'results.csv',
        ),
    )

    for command in commands:
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert command.run() == 0
        assert any(
            record.levelno == logging.WARNING and 'no runs match the selection' in record.message
            for record in caplog.records
        )
