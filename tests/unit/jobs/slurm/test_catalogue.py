"""The catalogue of read-only SLURM commands and its live id selection."""

from pathlib import Path
import re

import pytest

from tests.unit.jobs.slurm.fake_slurm import FakeSlurm, install_fake_slurm
from tslab.jobs.slurm.catalogue import (
    command_lines,
    CommandOutput,
    CommandValues,
    IdScope,
    run_slurm_command,
    select_ids,
    slurm_command,
    SLURM_COMMANDS,
)

VALUES = CommandValues(job_ids=('1', '2'), width=14, start='2026-11-02')
_SACCT_MIXED = '1|COMPLETED\n2_[3-5]|PENDING\n2_1|RUNNING\n'
_SACCT_DONE = '1|COMPLETED\n2|COMPLETED\n'


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    return install_fake_slurm(tmp_path=tmp_path, monkeypatch=monkeypatch, answers={})


def test_names_are_tool_dash_variant_and_never_a_tool() -> None:
    tools = {'squeue', 'sacct', 'seff', 'sstat', 'sprio', 'scontrol', 'sbatch', 'scancel', 'sinfo'}
    assert len(SLURM_COMMANDS) == 12
    for command in SLURM_COMMANDS:
        assert command.name.startswith(f'{command.tool}-')
        assert command.name not in tools


def test_command_lines_fill_values() -> None:
    [line] = command_lines(
        command=slurm_command(name='squeue-full-name'), ids=['1', '2'], values=VALUES
    )
    assert line[:4] == ['squeue', '--array', '--jobs', '1,2']
    assert '%.14j' in line[-1]


def test_per_job_command_runs_once_per_id() -> None:
    lines = command_lines(command=slurm_command(name='seff-all'), ids=['1', '2'], values=VALUES)
    assert lines == [['seff', '1'], ['seff', '2']]


def test_no_ids_gives_no_lines() -> None:
    assert command_lines(command=slurm_command(name='seff-all'), ids=[], values=VALUES) == []


def test_live_ids_come_from_sacct(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'stdout': _SACCT_MIXED}])
    assert select_ids(scope=IdScope.LIVE, job_ids=['1', '2']) == (['2'], False)
    assert select_ids(scope=IdScope.RUNNING, job_ids=['1', '2']) == (['2_1'], False)


def test_live_ids_fall_back_when_sacct_fails(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'exit': 1, 'stderr': 'accounting off'}])
    assert select_ids(scope=IdScope.LIVE, job_ids=['1', '2']) == (['1', '2'], True)


def test_live_command_with_no_live_jobs_runs_nothing(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'stdout': _SACCT_DONE}])
    output = run_slurm_command(command=slurm_command(name='squeue-full-name'), values=VALUES)
    assert output == CommandOutput(returncode=0, stdout='', stderr='')
    assert fake.calls('squeue') == []


def test_fallback_squeue_tolerates_purged_ids(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'exit': 1}])
    fake.set_answers(
        'squeue', [{'exit': 1, 'stderr': 'slurm_load_jobs error: Invalid job id specified'}]
    )
    output = run_slurm_command(command=slurm_command(name='squeue-full-name'), values=VALUES)
    assert output.returncode == 0


def test_live_command_runs_with_live_ids(fake: FakeSlurm) -> None:
    fake.set_answers('sacct', [{'stdout': _SACCT_MIXED}])
    fake.set_answers('squeue', [{'stdout': 'row\n'}])
    output = run_slurm_command(command=slurm_command(name='squeue-stock'), values=VALUES)
    assert output == CommandOutput(returncode=0, stdout='row\n', stderr='')
    assert fake.calls('squeue') == [['--jobs', '2']]


def test_missing_tool_gives_127(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PATH', str(tmp_path))
    output = run_slurm_command(command=slurm_command(name='seff-all'), values=VALUES)
    assert output.returncode == 127
    assert output.stderr == 'seff is not installed on this machine\n'


def test_unknown_name_lists_valid_names() -> None:
    with pytest.raises(KeyError, match='squeue-full-name'):
        slurm_command(name='squeue')


def test_docs_table_matches_catalogue() -> None:
    repo = Path(__file__).resolve().parents[4]
    text = (repo / 'docs' / 'hpc.md').read_text(encoding='utf-8')
    names = re.findall(r'^\| `([a-z]+-[a-z-]+)` \|', text, re.MULTILINE)
    assert names == [command.name for command in SLURM_COMMANDS]
