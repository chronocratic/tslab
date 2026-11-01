"""Versioned JSON documents for study status and the ``study show`` output."""

import dataclasses
import json
from pathlib import Path
from typing import cast

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    finish_run,
    jobs_of,
    study_directory,
    submit_study,
    write_run_status,
)
from tslab.cli import main
from tslab.core.status import InstanceState, RunState, utc_now_iso
from tslab.core.text import TextTable
from tslab.study.definition import expand_runs, load_study
from tslab.study.records import append_job, JobRecord
from tslab.study.reporting.documents import run_entry, show_document, status_document
from tslab.study.reporting.state import StudyState


def test_status_document_counts_every_state(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    finish_run(project=project, model='small', seed=1)
    ctx = context(project=project, machine_file='cluster.toml')
    document = status_document(context=ctx, exit_code=5)
    # why: status_document guarantees per-grid count rows with integer values
    counts = cast('dict[str, dict[str, int]]', document['counts'])
    assert next(iter(document)) == 'schema'
    assert counts['all']['done'] == 1
    assert counts['all']['total'] == len(ctx.reports)
    assert set(counts['all']) == {state.value for state in StudyState} | {'total'}


def test_run_entry_contains_absolute_directory_and_run_values(
    project: Path, fake: FakeSlurm
) -> None:
    submit_study(project=project)
    ctx = context(project=project, machine_file='cluster.toml')
    entry = run_entry(report=ctx.reports[0], context=ctx)
    assert set(entry) == {
        'run_path',
        'grid',
        'values',
        'state',
        'reason',
        'job_id',
        'job_state',
        'attempt',
        'directory',
    }
    assert (
        entry['directory']
        == (ctx.study_directory / ctx.reports[0].run.run_path).resolve().as_posix()
    )
    assert entry['values'] == dict(ctx.reports[0].run.values)


def test_run_entry_names_submission_in_progress(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    ctx = dataclasses.replace(
        context(project=project, machine_file='cluster.toml'), submission_in_progress=True
    )
    assert run_entry(report=ctx.reports[0], context=ctx)['reason'] == 'submission in progress'


def test_run_entry_names_live_auto_retry_job(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    job = jobs_of(project=project)[0]
    task = job.tasks[0]
    _, model, seed = task.run_path.split('/')
    write_run_status(
        project=project,
        model=model,
        seed=int(seed),
        run_state=RunState.RUNNING,
        instance=InstanceState.RUNNING,
    )
    followup = JobRecord(
        kind='followup',
        submission_id='followup',
        time=utc_now_iso(),
        launcher='slurm',
        job_id='2000',
        job_file=1,
        tasks=(),
    )
    append_job(
        path=study_directory(project=project) / '.study' / 'submissions.jsonl', record=followup
    )
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': f'{job.job_id}_{task.index}|TIMEOUT\n2000|PENDING\n'}]
    )
    ctx = context(project=project, machine_file='cluster.toml')
    assert run_entry(report=ctx.reports[0], context=ctx)['reason'] == 'auto retry queued (job 2000)'


def test_show_document_lists_axes_and_counts(project: Path) -> None:
    study = load_study(reference='toy_studies:study')
    document = show_document(study=study)
    grids = cast('list[dict[str, object]]', document['grids'])
    assert document['schema'] == 'tslab.study-show/1'
    assert grids[0]['axes'] == {'model': ['small', 'large'], 'seed': ['1', '2']}
    assert grids[0]['runs'] == len(expand_runs(study=study))
    assert document['total_runs'] == len(expand_runs(study=study))


def test_show_cli_prints_table_without_machine(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'), pytest.raises(SystemExit) as raised:
        main(args=['study', 'show', 'toy_studies:study'])
    assert raised.value.code == 0
    tables = [getattr(record, 'tslab_table', None) for record in caplog.records]
    assert any(
        isinstance(table, TextTable) and table.header == ('grid', 'short', 'runs', 'axes')
        for table in tables
    )
    assert 'command of g:' in caplog.text
    assert 'select with --grid <grid> or --only <axis>=<value>' in caplog.text


def test_show_cli_prints_json(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'show', 'toy_studies:study', '--json'])
    assert raised.value.code == 0
    assert json.loads(capsys.readouterr().out)['schema'] == 'tslab.study-show/1'
