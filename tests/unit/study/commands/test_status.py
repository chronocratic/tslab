"""``tslab study status``: the state table and the lines of the runs not done."""

import dataclasses
from pathlib import Path
from typing import cast

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, finish_run, submit_study
from tslab.cli import main
from tslab.cli.study import Status
from tslab.study.commands import parse_states, status
from tslab.study.context import Context, expected_starts
from tslab.study.documents import status_document
from tslab.study.report import log_status
from tslab.study.state import StudyState


def normalized(text: str) -> str:
    """Collapse runs of spaces, so column widths are free."""
    return ' '.join(text.split())


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
    assert 'g/small/1 queued job 1000_0' in text
    assert 'g/small/2 missing job 1000_1' in text
    assert 'g/large/2 missing job 1000_3' in text
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
    assert 'g/small/2 missing job 1000_1' in text
    assert 'g/large/2 missing job 1000_3' in text
    # why: the filter hides lines, not counts; the table stays complete
    assert 'g 1 0 1 0 0 2 4' in text


def test_a_study_that_never_ran_gets_no_run_lines(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g 0 0 0 0 0 4 4' in text
    assert 'g/small/1' not in text


def test_run_lines_stop_after_a_limit(
    project: Path,
    fake: FakeSlurm,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr('tslab.study.report._PROBLEMS_SHOWN', 1)
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n'}])
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g/small/1 queued job 1000_0' in text
    assert 'g/small/2' not in text
    assert '… and 3 more; narrow with --states, --grid or --only' in text


def _queued_context(project: Path, fake: FakeSlurm) -> Context:
    """A submitted study where 1000_0 is queued and the rest completed."""
    submit_study(project=project)
    fake.set_answers(
        'squeue',
        [
            {'exit': 0, 'stdout': '1000_0|PENDING\n'},
            {'exit': 0, 'stdout': '1000_0|2026-11-01T10:00:00\n'},
        ],
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    return context(project=project, machine_file='cluster.toml')


def test_status_shows_expected_start_of_queued_runs(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _queued_context(project=project, fake=fake)
    starts = expected_starts(context=ctx)
    with caplog.at_level('INFO'):
        status(context=ctx, starts=starts)
    text = normalized(caplog.text)
    assert 'g/small/1 queued job 1000_0 starts 2026-11-01T10:00:00' in text


def test_status_json_has_expected_start(project: Path, fake: FakeSlurm) -> None:
    ctx = _queued_context(project=project, fake=fake)
    starts = expected_starts(context=ctx)
    document = status_document(context=ctx, exit_code=0, starts=starts)
    # why: status_document types the value as object; the runs list is a list of dicts
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert document['schema'] == 'tslab.study-status/1'
    assert runs['g/small/1']['expected_start'] == '2026-11-01T10:00:00'
    assert runs['g/small/2']['expected_start'] is None
    assert runs['g/large/1']['expected_start'] is None


def test_status_without_queued_runs_skips_squeue_start(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n'}])
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    starts = expected_starts(context=ctx)
    assert starts is None
    assert len(fake.calls('squeue')) == 1
    document = status_document(context=ctx, exit_code=0, starts=starts)
    # why: status_document types the value as object; the runs list is a list of dicts
    for run in cast('list[dict[str, object]]', document['runs']):
        assert 'expected_start' not in run


def test_status_keeps_working_when_squeue_start_fails(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake.set_answers(
        'squeue',
        [
            {'exit': 0, 'stdout': '1000_0|PENDING\n'},
            {'exit': 1, 'stderr': 'slurm_load_jobs error: Socket timed out'},
        ],
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': '1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    with caplog.at_level('WARNING', logger='tslab.study.launchers.slurm.states'):
        starts = expected_starts(context=ctx)
    assert starts is None
    assert 'squeue' in caplog.text
    with caplog.at_level('INFO'):
        status(context=ctx, starts=starts)
    text = normalized(caplog.text)
    assert 'g/small/1 queued job 1000_0' in text
    assert ' starts ' not in text


def test_time_limit_hint(project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture) -> None:
    submit_study(project=project)
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_3|TIMEOUT\n'}])
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g/large/2 interrupted job 1000_3 SLURM TIMEOUT' in text
    assert (
        '1 runs stopped at a time limit: raise time in a machine override, '
        'or cap training with --trainer.max-training-time' in text
    )


def test_training_paused_displays_time_limit_hint(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    loaded = context(project=project, machine_file='cluster.toml')
    report = dataclasses.replace(
        loaded.reports[0],
        state=StudyState.INTERRUPTED,
        reason='TrainingPaused: job time share reached',
        job_state=None,
    )
    loaded = dataclasses.replace(loaded, reports=[report])

    with caplog.at_level('INFO'):
        log_status(context=loaded)

    assert '1 runs stopped at a time limit:' in normalized(caplog.text)


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
                '--states',
                'failed,missing',
                '--states',
                'done',
            ]
        )
    assert exit_info.value.code == 0
    assert seen[0].states == ('failed,missing', 'done')
