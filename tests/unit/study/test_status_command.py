"""``tslab study status``: the state table and the lines of the runs not done."""

from pathlib import Path

import pytest

from study.fake_slurm import FakeSlurm, install_fake_slurm
from study.project import write_toy_project
from tslab.cli import main
from tslab.cli.study import Status
from tslab.core.status import (
    InstanceRecord,
    InstanceState,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.study import Context, load_context, parse_states, Selection, status, StudyState, submit


def no_sleep(_seconds: float) -> None:
    """Skip the submission pause."""


def normalized(text: str) -> str:
    """Collapse runs of spaces, so column widths are free."""
    return ' '.join(text.split())


def context(*, project: Path, machine_file: str) -> Context:
    return load_context(
        reference='toy_studies:study',
        machine_path=project / machine_file,
        selection=Selection(),
        project_root=Path.cwd(),
    )


def submit_study(*, project: Path) -> None:
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0


def finish_run(*, project: Path, model: str, seed: int) -> None:
    """Write a done status.json, so the run reads as done from its folder."""
    folder = project / 'runs' / 's' / f'g/{model}/{seed}'
    folder.mkdir(parents=True)
    write_status(
        path=folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.DONE)},
        ),
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = write_toy_project(project=tmp_path / 'proj')
    # why: load_study imports from the current directory, like a user in a project
    monkeypatch.chdir(folder)
    return folder


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


def test_status_counts_and_lines(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])
    finish_run(project=project, model='large', seed=1)
    with caplog.at_level('INFO'):
        assert status(context=context(project=project, machine_file='cluster.toml')) == 0
    text = normalized(caplog.text)
    assert 'grid queued running done interrupted failed missing total' in text
    assert 'g 1 0 1 0 0 2 4' in text
    assert 'all 1 0 1 0 0 2 4' in text
    assert 'g/small/1 queued 1000_0' in text
    assert 'g/small/2 missing 1000_1' in text
    assert 'g/large/2 missing 1000_3' in text
    assert 'g/large/1' not in text  # done runs get no line


def test_state_filter_hides_other_lines(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])
    finish_run(project=project, model='large', seed=1)
    with caplog.at_level('INFO'):
        status(
            context=context(project=project, machine_file='cluster.toml'),
            states=(StudyState.MISSING,),
        )
    text = normalized(caplog.text)
    assert 'g/small/1' not in text
    assert 'g/small/2 missing 1000_1' in text
    assert 'g/large/2 missing 1000_3' in text
    # why: the filter hides lines, not counts; the table stays complete
    assert 'g 1 0 1 0 0 2 4' in text


def test_time_limit_hint(project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture) -> None:
    submit_study(project=project)
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_3|TIMEOUT\n'}])
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g/large/2 interrupted 1000_3 SLURM TIMEOUT' in text
    assert (
        '1 runs stopped at a time limit: raise time in a machine override, '
        'or cap training with --trainer.max-training-time' in text
    )


def test_unknown_job_states_are_said_once(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (project / 'empty-bin').mkdir()
    # why: without squeue and sacct on PATH, the SLURM states are unknown
    monkeypatch.setenv('PATH', str(project / 'empty-bin'))
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    assert caplog.text.count('job states unknown') == 1


def test_parse_states_splits_commas() -> None:
    assert parse_states(values=('failed,missing', 'done')) == (
        StudyState.FAILED,
        StudyState.MISSING,
        StudyState.DONE,
    )


def test_parse_states_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match='nope'):
        parse_states(values=('nope',))


def test_cli_parses_state_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[Status] = []

    def run(self: Status) -> int:
        seen.append(self)
        return 0

    monkeypatch.setattr(Status, 'run', run)
    with pytest.raises(SystemExit) as exit_info:
        main(
            args=[
                'study',
                'status',
                'toy_studies:study',
                '--machine',
                str(tmp_path / 'm.toml'),
                '--state',
                'failed,missing',
                '--state',
                'done',
            ]
        )
    assert exit_info.value.code == 0
    assert seen[0].state == ('failed,missing', 'done')
