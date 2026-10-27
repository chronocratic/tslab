"""``tslab study <command>`` end to end through ``main``: the run bodies and usage errors."""

from pathlib import Path

import pytest

from tests.unit.study.fake_slurm import FakeSlurm
from tests.unit.study.helpers import finish_run, submit_study
from tslab.cli import main


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
    table = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith(('grid', 'g ', 'all'))
    ]
    assert len({len(line) for line in table}) == 1


def test_submit_dry_run_and_resubmit_dry_run(project: Path, fake: FakeSlurm) -> None:
    assert run_main('submit', 'toy_studies:study', '--machine', 'cluster.toml', '--dry-run') == 0
    assert fake.calls('sbatch') == []
    submit_study(project=project)
    code = run_main(
        'resubmit', 'toy_studies:study', '--machine', 'cluster.toml', '--states', 'missing,failed'
    )
    assert code == 0


def test_results_writes_the_csv(project: Path, fake: FakeSlurm, tmp_path: Path) -> None:
    out = tmp_path / 'out.csv'
    code = run_main('results', 'toy_studies:study', '--machine', 'local.toml', '--out', str(out))
    assert code == 0
    assert out.read_text().splitlines()[0].startswith('grid,')


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
