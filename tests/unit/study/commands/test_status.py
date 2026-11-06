"""``tslab study status``: the state table and the lines of the runs not done."""

import dataclasses
import os
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from tests.support.fake_backend import FakeBackend
from tests.unit.study.helpers import context, finish_run, submit_study
from tslab.cli import main
from tslab.cli.study import Status
from tslab.core.status import InstanceRecord, RunStatusSnapshot, utc_now_iso, write_status
from tslab.enums import InstanceState, JobFlag, JobPhase, RunState, StudyState
from tslab.study.commands import parse_states, status
from tslab.study.context import Context, context_for, expected_starts, Selection
from tslab.study.definition import After, Grid, Study
from tslab.study.launchers.common.types import QueuedJob
from tslab.study.reporting.documents import status_document
from tslab.study.reporting.report import log_status


def normalized(text: str) -> str:
    """Collapse runs of spaces, so column widths are free."""
    return ' '.join(text.split())


def test_status_counts_and_lines(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.QUEUED})
    finish_run(project=project, model='large', seed=1)
    with caplog.at_level('INFO'):
        assert status(context=context(project=project, machine_file='cluster.toml')) == 0
    text = normalized(caplog.text)
    assert (
        'grid queued waiting running done interrupted failed dependency_failed missing total'
        in text
    )
    assert 'g 1 0 0 1 0 0 0 2 4' in text
    assert 'all 1 0 0 1 0 0 0 2 4' in text
    assert 'g/small/1 queued job job-0_0' in text
    assert 'g/small/2 missing job job-0_1' in text
    assert 'g/large/2 missing job job-0_3' in text
    assert 'g/large/1' not in text  # done runs get no line


def test_state_filter_hides_other_lines(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.QUEUED})
    finish_run(project=project, model='large', seed=1)
    with caplog.at_level('INFO'):
        status(
            context=context(project=project, machine_file='cluster.toml'),
            states=(StudyState.MISSING,),
        )
    text = normalized(caplog.text)
    assert 'g/small/1' not in text
    assert 'g/small/2 missing job job-0_1' in text
    assert 'g/large/2 missing job job-0_3' in text
    # why: the filter hides lines, not counts; the table stays complete
    assert 'g 1 0 0 1 0 0 0 2 4' in text


def test_a_study_that_never_ran_gets_no_run_lines(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g 0 0 0 0 0 0 0 4 4' in text
    assert 'g/small/1' not in text


def test_run_lines_stop_after_a_limit(
    project: Path,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr('tslab.study.reporting.report._PROBLEMS_SHOWN', 1)
    submit_study(project=project)
    fake_backend.set_states(phases={'job-0_0': JobPhase.QUEUED})
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g/small/1 queued job job-0_0' in text
    assert 'g/small/2' not in text
    assert '… and 3 more; narrow with --states, --grid or --only' in text


def _queued_context(project: Path, fake_backend: FakeBackend) -> Context:
    """A submitted study where job-0_0 is queued and the rest completed."""
    submit_study(project=project)
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.QUEUED,
            'job-0_1': JobPhase.COMPLETED,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.COMPLETED,
        }
    )
    fake_backend.expected_starts_answer = {
        'job-0_0': QueuedJob(start='2026-11-01T10:00:00', reason=None, stuck=False)
    }
    return context(project=project, machine_file='cluster.toml')


def test_status_shows_expected_start_of_queued_runs(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _queued_context(project=project, fake_backend=fake_backend)
    starts = expected_starts(context=ctx)
    with caplog.at_level('INFO'):
        status(context=ctx, starts=starts)
    text = normalized(caplog.text)
    assert 'g/small/1 queued job job-0_0 starts 2026-11-01T10:00:00' in text


def test_status_json_has_expected_start(project: Path, fake_backend: FakeBackend) -> None:
    ctx = _queued_context(project=project, fake_backend=fake_backend)
    starts = expected_starts(context=ctx)
    document = status_document(context=ctx, exit_code=0, starts=starts)
    # why: status_document types the value as object; the runs list is a list of dicts
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert document['schema'] == 'tslab.study-status/1'
    assert runs['g/small/1']['expected_start'] == '2026-11-01T10:00:00'
    assert runs['g/small/1']['pending_reason'] is None
    assert runs['g/small/1']['pending_stuck'] is False
    assert runs['g/small/2']['expected_start'] is None
    assert runs['g/large/1']['expected_start'] is None


def test_status_without_queued_runs_skips_the_start_query(
    project: Path, fake_backend: FakeBackend
) -> None:
    submit_study(project=project)
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.RUNNING,
            'job-0_1': JobPhase.COMPLETED,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.COMPLETED,
        }
    )
    fake_backend.expected_starts_answer = None
    ctx = context(project=project, machine_file='cluster.toml')
    starts = expected_starts(context=ctx)
    assert starts is None
    document = status_document(context=ctx, exit_code=0, starts=starts)
    # why: status_document types the value as object; the runs list is a list of dicts
    for run in cast('list[dict[str, object]]', document['runs']):
        assert 'expected_start' not in run


def test_status_keeps_working_without_expected_starts(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake_backend.set_states(
        phases={
            'job-0_0': JobPhase.QUEUED,
            'job-0_1': JobPhase.COMPLETED,
            'job-0_2': JobPhase.COMPLETED,
            'job-0_3': JobPhase.COMPLETED,
        }
    )
    fake_backend.expected_starts_answer = None
    ctx = context(project=project, machine_file='cluster.toml')
    starts = expected_starts(context=ctx)
    assert starts is None
    with caplog.at_level('INFO'):
        status(context=ctx, starts=starts)
    text = normalized(caplog.text)
    assert 'g/small/1 queued job job-0_0' in text
    assert ' starts ' not in text


def test_time_limit_hint(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    fake_backend.set_states(
        phases={'job-0_3': JobPhase.INTERRUPTED},
        reasons={'job-0_3': 'stopped at the limit'},
        flags={'job-0_3': frozenset({JobFlag.TIME_LIMIT})},
    )
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    text = normalized(caplog.text)
    assert 'g/large/2 interrupted job job-0_3 stopped at the limit' in text
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
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    fake_backend.set_states(phases={}, known=False)
    with caplog.at_level('INFO'):
        status(context=context(project=project, machine_file='cluster.toml'))
    assert caplog.text.count('job states unknown') == 1


def test_parse_states_splits_commas() -> None:
    assert parse_states(values=('failed,missing', 'done')) == (
        StudyState.FAILED,
        StudyState.MISSING,
        StudyState.DONE,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ChainStatusConfig:
    output_root: Path = Path('runs')
    run_path: Path = Path('run')


def _chained_study() -> Study:
    pretrain = Grid(
        short_name='pre',
        command=('python', '-m', 'pretrain'),
        config=_ChainStatusConfig,
        axes={'dataset': {'a': {}}},
        run_path='{grid}/{dataset}',
    )
    probe = Grid(
        short_name='probe',
        command=('python', '-m', 'probe'),
        config=_ChainStatusConfig,
        axes={'dataset': {'a': {}}},
        after=(After(grid='pretrain'),),
        run_path='{grid}/{dataset}',
    )
    return Study(name='chain', grids={'pretrain': pretrain, 'probe': probe})


def test_status_selecting_downstream_uses_upstream_report(
    project: Path, fake_backend: FakeBackend, caplog: pytest.LogCaptureFixture
) -> None:
    upstream_path = PurePosixPath('pretrain/a')
    upstream_folder = project / 'runs' / 'chain' / upstream_path
    upstream_folder.mkdir(parents=True)
    write_status(
        path=upstream_folder / 'status.json',
        snapshot=RunStatusSnapshot(
            run_state=RunState.FINISHED,
            updated_at=utc_now_iso(),
            instances={'train': InstanceRecord(state=InstanceState.FAILED)},
        ),
    )
    ctx = context_for(
        study=_chained_study(),
        machine_path=project / 'cluster.toml',
        selection=Selection(grids=('probe',)),
        project_root=project,
        local_cpus=1,
        environment=os.environ,
    )

    assert len(ctx.reports) == 1
    assert ctx.reports[0].state is StudyState.DEPENDENCY_FAILED
    assert ctx.upstream_reports[upstream_path].state is StudyState.FAILED

    waiting = dataclasses.replace(ctx.reports[0], state=StudyState.WAITING)
    upstream = dataclasses.replace(
        ctx.upstream_reports[upstream_path], state=StudyState.QUEUED, job_id='job-9_0'
    )
    ctx = dataclasses.replace(ctx, reports=[waiting], upstream_reports={upstream_path: upstream})
    entry = status_document(context=ctx, exit_code=0)['runs']
    assert cast('list[dict[str, object]]', entry)[0]['waits_for'] == ['pretrain/a']
    with caplog.at_level('INFO'):
        status(context=ctx)
    assert 'waits for pretrain/a' in normalized(caplog.text)


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
