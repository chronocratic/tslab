"""``tslab study usage``: what each run used, and what to size the next jobs to."""

import json
from pathlib import Path
from typing import cast

import pytest

from tests.unit.jobs.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, submit_study, write_run_status
from tslab.cli import main
from tslab.core.status import InstanceState, RunState
from tslab.study.context import Context
from tslab.study.usage import usage_document

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


def test_usage_on_local_machine_exits_1(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('ERROR', logger='tslab.cli.study'), pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'local.toml', '--json'])
    assert raised.value.code == 1
    assert 'study usage reads SLURM accounting; this machine is local' in caplog.text


def test_usage_sacct_failure_exits_1(project: Path, fake: FakeSlurm) -> None:
    _write_usage_machine(project=project)
    submit_study(project=project, machine_file='usage.toml')
    fake.set_answers(
        'sacct', [{'exit': 0, 'stdout': _STATE_ANSWER}, {'exit': 1, 'stderr': 'sacct: error: down'}]
    )
    with pytest.raises(SystemExit) as raised:
        main(args=['study', 'usage', 'toy_studies:study', '--machine', 'usage.toml'])
    assert raised.value.code == 1
