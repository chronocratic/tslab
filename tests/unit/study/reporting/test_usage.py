"""``tslab study usage``: what each run used, and what to size the next jobs to."""

import json
from pathlib import Path
from typing import cast

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import (
    context,
    record_local_job,
    study_directory,
    submit_study,
    write_run_status,
)
from tslab.cli import main
from tslab.core.status import InstanceState, RunState
from tslab.study.context import Context
from tslab.study.launchers import JobUsage
import tslab.study.reporting.usage as usage_module
from tslab.study.reporting.usage import log_usage, usage_document

USAGE_MACHINE = """\
launcher = "slurm"
output_root = "{output_root}"

[slurm]
partition = "CHANGE_ME"
time = "00:10:00"
cpus_per_task = 2
mem_per_cpu = "1G"
"""

# two runs finished with data, two still queued without any
_USAGE_ANSWER = (
    '1000_0|COMPLETED|01:00:00|02:00:00|00:30:00||2|\n'
    '1000_0.batch|COMPLETED|01:00:00||00:30:00|1G|2|\n'
    '1000_1|COMPLETED|02:00:00|02:00:00|01:00:00||2|\n'
    '1000_1.batch|COMPLETED|02:00:00||01:00:00|2G|2|\n'
    '1000_2|PENDING||||||\n'
    '1000_3|PENDING||||||\n'
)

_STATE_ANSWER = '1000_0|COMPLETED\n1000_1|COMPLETED\n1000_2|PENDING\n1000_3|PENDING\n'


def _write_usage_machine(*, project: Path) -> None:
    text = USAGE_MACHINE.format(output_root=(project / 'runs').as_posix())
    (project / 'usage.toml').write_text(text)


def _submitted_context(*, project: Path, fake: FakeSlurm) -> Context:
    """Submit the toy study and read its context with the usage machine profile."""
    _write_usage_machine(project=project)
    submit_study(project=project, machine_file='usage.toml')
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': _STATE_ANSWER}, {'exit': 0, 'stdout': _USAGE_ANSWER}]
    )
    return context(project=project, machine_file='usage.toml')


def test_usage_rows_compare_peak_with_request(project: Path, fake: FakeSlurm) -> None:
    ctx = _submitted_context(project=project, fake=fake)
    document = usage_document(context=ctx)
    # why: usage_document types the values as object; runs is a list of dicts
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert runs['g/small/1']['max_rss'] == 1 * 1024**3
    assert runs['g/small/1']['requested_memory'] == 2 * 1024**3
    assert runs['g/small/2']['max_rss'] == 2 * 1024**3
    assert runs['g/small/2']['requested_memory'] == 2 * 1024**3
    assert runs['g/large/1']['max_rss'] is None
    assert runs['g/large/1']['requested_memory'] == 2 * 1024**3


def test_usage_group_suggests_time_and_memory_with_margin(project: Path, fake: FakeSlurm) -> None:
    ctx = _submitted_context(project=project, fake=fake)
    document = usage_document(context=ctx)
    # why: usage_document types the values as object; groups is a list of dicts
    (group,) = cast('list[dict[str, object]]', document['groups'])
    assert group['runs'] == 4
    assert group['suggested_time'] == '2:24:00'
    assert group['suggested_mem_per_cpu'] == '1229M'


def test_usage_leaves_paused_runs_out_of_time_suggestion(project: Path, fake: FakeSlurm) -> None:
    ctx = _submitted_context(project=project, fake=fake)
    # pause the 2:00:00 run (g/small/2 / 1000_1)
    write_run_status(
        project=project,
        model='small',
        seed=2,
        run_state=RunState.FAILED,
        instance=InstanceState.FAILED,
        error='TrainingPaused: job time share reached',
    )
    ctx = context(project=project, machine_file='usage.toml')
    document = usage_document(context=ctx)
    # why: usage_document types the values as object; runs is a list of dicts
    runs = {run['run_path']: run for run in cast('list[dict[str, object]]', document['runs'])}
    assert runs['g/small/2']['paused'] is True
    # why: usage_document types the values as object; groups is a list of dicts
    (group,) = cast('list[dict[str, object]]', document['groups'])
    assert group['suggested_time'] == '1:12:00'
    assert group['suggested_mem_per_cpu'] == '1229M'


def test_usage_cli_json_schema(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    _submitted_context(project=project, fake=fake)
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'usage.toml', '--json'])
    document = json.loads(capsys.readouterr().out)
    assert raised.value.code == 0
    assert document['schema'] == 'tslab.study-usage/1'


def test_usage_sacct_failure_exits_1(project: Path, fake: FakeSlurm) -> None:
    _write_usage_machine(project=project)
    submit_study(project=project, machine_file='usage.toml')
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': _STATE_ANSWER}, {'exit': 1, 'stderr': 'sacct: error: down'}]
    )
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'usage.toml'])
    assert raised.value.code == 1


def _write_local_usage(
    *, project: Path, index: int, elapsed: float, rss: int, gpu_id: str | None
) -> None:
    """Write the usage file of local task ``index`` of the recorded job, as the launcher does."""
    logs = study_directory(project=project) / '.study' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f'g1_{index}.out.usage.json').write_text(
        json.dumps(
            {
                'schema': 'tslab.local-usage/1',
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


def test_usage_document_uses_selected_backend(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(project=project, machine_file='local.toml')
    usage: dict[str, JobUsage] = {}
    gpu_ids: dict[str, str | None] = {'job': '3'}
    calls: list[Context] = []

    class Backend:
        suggests_memory = False

        def suggest_sizing(self, *, elapsed: object, peak_per_cpu: object) -> tuple[None, None]:
            return None, None

        def usage(self, *, context: Context) -> tuple[dict[str, JobUsage], dict[str, str | None]]:
            calls.append(context)
            return usage, gpu_ids

    backend = Backend()
    monkeypatch.setattr(usage_module, 'backend_for', lambda **_kwargs: backend, raising=False)

    document = usage_document(context=ctx)

    assert calls == [ctx]
    assert document['runs'] == []


def test_local_usage_log_shows_no_limit_and_dash(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    document = usage_document(context=_local_usage_context(project=project))
    with caplog.at_level('INFO', logger='tslab.study.reporting.usage'):
        log_usage(document=document, suggests_memory=False)
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
