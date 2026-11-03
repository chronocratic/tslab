"""The ``tslab study`` command group: parsing, filters and pass-through."""

import logging
from pathlib import Path
from typing import cast, TYPE_CHECKING

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.cli.study.helpers import toy_project
from tests.unit.study.helpers import submit_study
from tslab.cli import main
from tslab.cli.study import Resubmit, Results, Submit
from tslab.enums import JobPhase

if TYPE_CHECKING:
    from tslab.study.context import Context


def run_main(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', *args])
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def _tslab_toml_with_cluster(*, project: Path) -> None:
    """The project file: the toy module, and every host on the cluster."""
    (project / 'tslab.toml').write_text(
        'studies = "toy_studies"\n\n[machine_by_host]\n"*" = "cluster"\n', encoding='utf-8'
    )
    machines = project / 'machines'
    machines.mkdir()
    (machines / 'cluster.toml').write_text((project / 'cluster.toml').read_text(encoding='utf-8'))


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


def test_submit_resubmit_and_results_warn_for_empty_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
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


def test_submit_with_short_name_and_no_machine_records_the_full_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _tslab_toml_with_cluster(project=project)
    assert run_main('submit', 'study') == 0
    context = cast('Context', fake_backend.calls('launch')[0]['context'])
    assert context.reference == 'toy_studies:study'


def test_status_hint_repeats_the_full_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_backend: FakeBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _tslab_toml_with_cluster(project=project)
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.RUNNING})
    with caplog.at_level(logging.INFO):
        assert run_main('status', 'study') == 0
    (hint,) = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith('follow progress')
    ]
    assert 'toy_studies:study --machine cluster' in hint


def test_broken_tslab_toml_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = toy_project(tmp_path=tmp_path, monkeypatch=monkeypatch)
    (project / 'tslab.toml').write_text('studies = 3\n', encoding='utf-8')
    assert run_main('status', 'x', '--machine', 'cluster.toml') == 2
