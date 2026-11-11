"""``tslab study usage``: what each run used, and what to size the next jobs to."""

from datetime import timedelta
import json
import math
from pathlib import Path
from typing import cast

import pytest

from tests.lab.unit.study.helpers import (
    context,
    record_local_job,
    study_directory,
    submit_study,
    write_run_status,
)
from tests.support.fake_backend import FakeBackend, FakeResources
from tslab.lab.cli.entrypoint import main
from tslab.lab.enums import InstanceState, RunState
from tslab.lab.study.context import Context
from tslab.lab.study.launchers.common.types import JobUsage, LaunchError
from tslab.lab.study.reporting.usage import log_usage, usage_document


def _submitted_context(*, project: Path, fake_backend: FakeBackend) -> Context:
    """Submit the toy study and read its context on the cluster profile."""
    fake_backend.resources_answer = FakeResources(cpus=2, requested_memory=2 * 1024**3)
    submit_study(project=project)
    # two runs finished with data, two still queued without any
    fake_backend.usage_answer = {
        'job-0_0': JobUsage(
            state='completed',
            elapsed=timedelta(hours=1),
            time_limit=None,
            total_cpu=None,
            max_rss=1 * 1024**3,
            alloc_cpus=2,
            gpus=0,
        ),
        'job-0_1': JobUsage(
            state='completed',
            elapsed=timedelta(hours=2),
            time_limit=None,
            total_cpu=None,
            max_rss=2 * 1024**3,
            alloc_cpus=2,
            gpus=0,
        ),
    }
    return context(project=project, machine_file='cluster.toml')


def test_usage_rows_compare_peak_with_request(project: Path, fake_backend: FakeBackend) -> None:
    ctx = _submitted_context(project=project, fake_backend=fake_backend)
    document = usage_document(context=ctx)
    # why: usage_document types the values as object; runs is a list of dicts
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert runs['g/small/1']['max_rss'] == 1 * 1024**3
    assert runs['g/small/1']['requested_memory'] == 2 * 1024**3
    assert runs['g/small/2']['max_rss'] == 2 * 1024**3
    assert runs['g/small/2']['requested_memory'] == 2 * 1024**3
    assert runs['g/large/1']['max_rss'] is None
    assert runs['g/large/1']['requested_memory'] == 2 * 1024**3


def test_usage_group_suggests_time_and_memory_with_margin(
    project: Path, fake_backend: FakeBackend
) -> None:
    fake_backend.sizing_answer = ('T', 'M')
    ctx = _submitted_context(project=project, fake_backend=fake_backend)
    document = usage_document(context=ctx)
    # why: usage_document types the values as object; groups is a list of dicts
    (group,) = cast('list[dict[str, object]]', document['groups'])
    assert group['runs'] == 4
    assert group['suggested_time'] == 'T'
    assert group['suggested_mem_per_cpu'] == 'M'
    assert fake_backend.calls('suggest_sizing') == [
        {
            'elapsed': timedelta(hours=2, minutes=24),
            'peak_per_cpu': math.ceil(2 * 1024**3 * 1.2 / 2),
        }
    ]


def test_usage_leaves_paused_runs_out_of_time_suggestion(
    project: Path, fake_backend: FakeBackend
) -> None:
    fake_backend.sizing_answer = ('T', 'M')
    _submitted_context(project=project, fake_backend=fake_backend)
    # pause the 2:00:00 run (g/small/2)
    write_run_status(
        project=project,
        model='small',
        seed=2,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    ctx = context(project=project, machine_file='cluster.toml')
    document = usage_document(context=ctx)
    # why: usage_document types the values as object; runs is a list of dicts
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert runs['g/small/2']['paused'] is True
    # why: usage_document types the values as object; groups is a list of dicts
    (group,) = cast('list[dict[str, object]]', document['groups'])
    assert group['suggested_time'] == 'T'
    assert group['suggested_mem_per_cpu'] == 'M'
    assert fake_backend.calls('suggest_sizing') == [
        {
            'elapsed': timedelta(hours=1, minutes=12),
            'peak_per_cpu': math.ceil(2 * 1024**3 * 1.2 / 2),
        }
    ]


def test_usage_cli_json_schema(
    project: Path, fake_backend: FakeBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    _submitted_context(project=project, fake_backend=fake_backend)
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'cluster.toml', '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['schema'] == 'tslab.study-usage'


def test_usage_backend_failure_exits_1(project: Path, fake_backend: FakeBackend) -> None:
    _submitted_context(project=project, fake_backend=fake_backend)
    fake_backend.errors['usage'] = LaunchError('down')
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'cluster.toml'])
    assert raised.value.code == 1


def _write_local_usage(
    *, project: Path, index: int, elapsed: float, rss: int, gpu_id: str | None
) -> None:
    """Write the usage file of local task ``index`` of the recorded job, as the launcher does."""
    logs = study_directory(project=project) / '.study' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f'{logs.parent.parent.name}_g1_{index}.out.usage.json').write_text(
        json.dumps(
            {
                'schema': 'tslab.local-usage',
                'attempt': 1,
                'job_id': 'local-1-1',
                'exit_code': 0,
                'elapsed_seconds': elapsed,
                'cpu_seconds': 90.5,
                'max_rss_bytes': rss,
                'gpu_id': gpu_id,
                'timeout': None,
                'timed_out': False,
            }
        )
    )


def _local_usage_context(*, project: Path) -> Context:
    """Two recorded local runs with usage files, and two that left none."""
    record_local_job(project=project)
    _write_local_usage(project=project, index=0, elapsed=3600, rss=2 * 1024**3, gpu_id='0')
    _write_local_usage(project=project, index=1, elapsed=1800, rss=1024**3, gpu_id=None)
    return context(project=project, machine_file='local.toml')


def test_local_usage_document_reads_usage_files(project: Path) -> None:
    document = usage_document(context=_local_usage_context(project=project))
    assert document['launcher'] == 'local'
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    first = runs['g/small/1']
    assert first['elapsed'] == '1:00:00'
    assert first['max_rss'] == 2 * 1024**3
    assert first['total_cpu_seconds'] == 90
    assert (first['gpus'], first['gpu_id']) == (1, '0')
    assert first['time_limit'] is None
    assert (runs['g/small/2']['gpus'], runs['g/small/2']['gpu_id']) == (0, None)
    never = runs['g/large/1']
    assert never['elapsed'] is None
    assert never['max_rss'] is None
    assert never['total_cpu_seconds'] is None
    (group,) = cast('list[dict[str, object]]', document['groups'])
    assert group['suggested_time'] is None
    assert group['suggested_mem_per_cpu'] is None


def test_usage_document_uses_selected_backend(project: Path, fake_backend: FakeBackend) -> None:
    ctx = context(project=project, machine_file='local.toml')
    fake_backend.gpu_ids_answer = {'job': '3'}
    document = usage_document(context=ctx)
    assert fake_backend.calls('usage') == [{'context': ctx}]
    assert document['runs'] == []


def test_local_usage_log_shows_no_limit_and_dash(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    document = usage_document(context=_local_usage_context(project=project))
    with caplog.at_level('INFO', logger='tslab.lab.study.reporting.usage'):
        log_usage(document=document, has_time_limit=False)
    assert '1:00:00/no limit' in caplog.text
    assert 'suggest' not in caplog.text
    assert '-/no limit' in caplog.text


def test_usage_cli_local_json(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _local_usage_context(project=project)
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'local.toml', '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['launcher'] == 'local'
