"""``tslab study <command>`` end to end through ``main``: the run bodies and usage errors."""

import json
import logging
from pathlib import Path
from typing import cast

import pytest

from tests.support.fake_slurm import FakeSlurm
from tests.unit.study.helpers import finish_run, submit_study, write_run_status
from tslab.cli import main
from tslab.core.status import InstanceState, RunState
from tslab.core.text import TextTable


def run_main(*args: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args=['study', *args])
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def test_status_prints_an_aligned_table(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    submit_study(project=project)
    finish_run(project=project, model='small', seed=1)
    with caplog.at_level('INFO'):
        code = run_main('status', 'toy_studies:study', '--machine', 'cluster.toml')
    assert code == 0
    (table,) = _tables(caplog=caplog)
    assert table.total
    assert table.rows[-1][0] == 'all'
    (message,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith('grid')]
    assert len({len(line) for line in message.splitlines()}) == 1


def test_status_json_prints_only_json(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    root.handlers.clear()
    try:
        code = run_main('status', 'toy_studies:study', '--machine', 'cluster.toml', '--json')
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)['schema'] == 'tslab.study-status/1'
    assert 'nothing ran yet' in captured.err


def test_status_exits_0_without_exit_code_flag_even_with_failures(
    project: Path, fake: FakeSlurm
) -> None:
    submit_study(project=project)
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FINISHED,
        instance=InstanceState.FAILED,
    )
    assert run_main('status', 'toy_studies:study', '--machine', 'cluster.toml') == 0


def test_status_exit_code_flag_returns_4_on_failure(project: Path, fake: FakeSlurm) -> None:
    submit_study(project=project)
    write_run_status(
        project=project,
        model='small',
        seed=1,
        run_state=RunState.FINISHED,
        instance=InstanceState.FAILED,
    )
    assert run_main('status', 'toy_studies:study', '--machine', 'cluster.toml', '--exit-code') == 4


def test_status_json_carries_exit_code_without_flag(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    submit_study(project=project)
    code = run_main('status', 'toy_studies:study', '--machine', 'cluster.toml', '--json')
    document = json.loads(capsys.readouterr().out)
    assert document['exit_code'] == 5
    assert code == 0


def test_status_exit_code_rejects_empty_selection(project: Path) -> None:
    assert (
        run_main(
            'status',
            'toy_studies:study',
            '--machine',
            'cluster.toml',
            '--grid',
            'missing',
            '--exit-code',
        )
        == 2
    )


def test_status_rejects_watch_with_json(project: Path) -> None:
    assert (
        run_main(
            'status', 'toy_studies:study', '--machine', 'cluster.toml', '--watch', '10', '--json'
        )
        == 2
    )


def test_submit_dry_run_and_resubmit_dry_run(project: Path, fake: FakeSlurm) -> None:
    assert run_main('submit', 'toy_studies:study', '--machine', 'cluster.toml', '--dry-run') == 0
    assert fake.calls('sbatch') == []
    submit_study(project=project)
    code = run_main(
        'resubmit', 'toy_studies:study', '--machine', 'cluster.toml', '--states', 'missing,failed'
    )
    assert code == 0


def test_submit_dry_run_json_lists_jobs(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_main(
        'submit', 'toy_studies:study', '--machine', 'cluster.toml', '--dry-run', '--json'
    )
    document = json.loads(capsys.readouterr().out)
    jobs = cast('list[dict[str, object]]', document['jobs'])
    assert code == 0
    assert document['schema'] == 'tslab.study-launch/1'
    assert jobs[0]['job_id'] is None
    assert sum(cast('int', job['runs']) for job in jobs) == 4


def test_resubmit_json_lists_cancelled_ids(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    submit_study(project=project)
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|RUNNING\n1000_1|PENDING\n'}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|COMPLETED\n1000_3|COMPLETED\n'}])
    code = run_main(
        'resubmit',
        'toy_studies:study',
        '--machine',
        'cluster.toml',
        '--states',
        'missing',
        '--cancel-live',
        '--json',
    )
    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document['cancelled'] == ['1000_0', '1000_1']
    assert fake.calls('scancel') == [['1000_0', '1000_1']]


def test_local_submit_json_keeps_task_output_off_stdout(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    study_file = project / 'toy_studies.py'
    source = study_file.read_text(encoding='utf-8')
    source = source.replace(
        "command=(sys.executable, 'toy_script.py'),",
        "command=(sys.executable, '-c', 'print(\"TASK-OUTPUT\")'),",
    )
    study_file.write_text(source, encoding='utf-8')
    code = run_main(
        'submit',
        'toy_studies:study',
        '--machine',
        'local.toml',
        '--only',
        'model=small',
        '--only',
        'seed=1',
        '--json',
    )
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)['schema'] == 'tslab.study-launch/1'
    assert 'TASK-OUTPUT' in captured.err


def test_results_writes_the_csv(project: Path, fake: FakeSlurm, tmp_path: Path) -> None:
    out = tmp_path / 'out.csv'
    code = run_main('results', 'toy_studies:study', '--machine', 'local.toml', '--out', str(out))
    assert code == 0
    assert out.read_text().splitlines()[0].startswith('grid,')


def test_results_out_dash_writes_csv_to_stdout(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    root.handlers.clear()
    try:
        code = run_main('results', 'toy_studies:study', '--machine', 'local.toml', '--out', '-')
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines()[0].startswith('grid,')
    assert 'wrote 4 rows' in captured.err


def test_results_json_to_stdout(
    project: Path, fake: FakeSlurm, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_main('results', 'toy_studies:study', '--machine', 'local.toml', '--out', '-', '--json')
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document['schema'] == 'tslab.study-results/1'
    assert len(document['rows']) == 4


def test_results_json_to_file(
    project: Path, fake: FakeSlurm, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / 'rows.json'
    assert (
        run_main(
            'results', 'toy_studies:study', '--machine', 'local.toml', '--out', str(out), '--json'
        )
        == 0
    )
    assert capsys.readouterr().out == ''
    assert json.loads(out.read_text(encoding='utf-8'))['schema'] == 'tslab.study-results/1'


@pytest.mark.parametrize(
    'args',
    [
        ['status', 'toy_studies:study', '--machine', 'cluster.toml', '--states', 'bogus'],
        ['resubmit', 'toy_studies:study', '--machine', 'cluster.toml', '--states', 'bogus'],
        ['status', 'no_such_module:study', '--machine', 'cluster.toml'],
        ['status', 'toy_studies', '--machine', 'cluster.toml'],
        ['status', 'toy_studies:study', '--machine', 'missing.toml'],
        ['status', 'toy_studies:nothing', '--machine', 'cluster.toml'],
    ],
)
def test_usage_errors_exit_two_without_a_traceback(
    project: Path, args: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    assert run_main(*args) == 2
    assert caplog.records
    assert all(record.exc_info is None for record in caplog.records)


def test_results_says_how_many_rows_it_wrote(
    project: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / 'out.csv'
    with caplog.at_level('INFO'):
        code = run_main(
            'results', 'toy_studies:study', '--machine', 'local.toml', '--out', str(out)
        )
    assert code == 0
    assert f'wrote 4 rows to {out.as_posix()}' in caplog.text


def test_a_missing_study_module_says_where_to_run(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert run_main('status', 'no_such_module:study', '--machine', 'cluster.toml') == 2
    assert 'cannot import the study module' in caplog.text


def _tables(*, caplog: pytest.LogCaptureFixture) -> list[TextTable]:
    tables = [getattr(record, 'tslab_table', None) for record in caplog.records]
    return [table for table in tables if isinstance(table, TextTable)]


def test_dry_run_and_submit_log_tables(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        run_main('submit', 'toy_studies:study', '--machine', 'cluster.toml', '--dry-run')
    assert [table.header for table in _tables(caplog=caplog)] == [('job', 'runs', 'resources')]
    caplog.clear()
    with caplog.at_level('INFO'):
        submit_study(project=project)
    assert [table.header for table in _tables(caplog=caplog)] == [
        ('job', 'job id', 'runs', 'indices')
    ]
