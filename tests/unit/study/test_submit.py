"""``tslab study submit``: plans, writes, launches and records study runs."""

from pathlib import Path
import subprocess
import sys

import pytest

from tests.unit.jobs.fake_slurm import FakeSlurm
from tests.unit.study.helpers import context, jobs_of, no_sleep
from tslab.study.commands import submit
from tslab.study.context import load_context, Selection
from tslab.study.state import StudyState

# why: the small model has no ``depth`` field, so a study that sets
# ``model.depth`` for it does not load; the parse-bad study leaves the
# required ``data`` field unset, so its runs fail validation instead.
LOAD_BAD_STUDIES = """
import sys

import toy_script
from tslab.study import Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {
                    'small': {'model': 'small', 'model.depth': 7},
                    'large': {'model': 'large'},
                },
                'seed': {'1': {'seed': 1}},
            },
        )
    },
)
"""

PARSE_BAD_STUDIES = """
import sys

import toy_script
from tslab.study import Grid, Study

study = Study(
    name='s',
    grids={
        'g': Grid(
            short_name='g1',
            command=(sys.executable, 'toy_script.py'),
            config=toy_script.ToyConfig,
            axes={
                'model': {'small': {'model': 'small'}, 'large': {'model': 'large'}},
                'seed': {'1': {'seed': 1}, '2': {'seed': 2}},
            },
            # why: clears the data setting, so every run omits a required field
            run_settings={},
        )
    },
)
"""


def run_toy(*, project: Path, model: str, seed: int) -> None:
    """Run one toy run for real, so its status.json is a real one."""
    subprocess.run(  # noqa: S603 # why: the toy script of the project, not external input
        [
            sys.executable,
            'toy_script.py',
            f'model:{model}',
            '--seed',
            str(seed),
            # why: the runs of the study live under <output_root>/<study>
            '--output-root',
            str(project / 'runs' / 's'),
            '--run-path',
            f'g/{model}/{seed}',
            '--data',
            str(project / 'data'),
        ],
        cwd=project,
        check=True,
        capture_output=True,
    )


def test_submit_on_slurm_writes_arrays_and_records(project: Path, fake: FakeSlurm) -> None:
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    assert len(fake.calls('sbatch')) == 1
    jobs = jobs_of(project=project)
    assert len(jobs) == 1
    assert [(t.run_path, t.index) for t in jobs[0].tasks] == [
        ('g/small/1', 0),
        ('g/small/2', 1),
        ('g/large/1', 2),
        ('g/large/2', 3),
    ]
    assert all(t.attempt == 1 for t in jobs[0].tasks)


def test_submit_skips_done_and_live_runs(project: Path, fake: FakeSlurm) -> None:
    first = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert first == 0
    fake.set_answers('squeue', [{'exit': 0, 'stdout': '1000_0|PENDING\n1000_1|RUNNING\n'}])
    fake.set_answers('sacct', [{'exit': 0, 'stdout': '1000_2|COMPLETED\n1000_3|TIMEOUT\n'}])
    run_toy(project=project, model='large', seed=1)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path for t in jobs[1].tasks] == ['g/large/2']
    task = jobs[1].tasks[0]
    assert task.attempt == 2
    # why: the run keeps the array index of its first submission
    assert task.index == 3
    script = (project / 'runs' / 's' / '.study' / 'jobs' / f'{jobs[1].job_file}.sbatch').read_text()
    assert '#SBATCH --array=3%100' in script
    assert '#SBATCH --job-name=g1' in script
    assert 'logs/g1_%a.out' in script


def test_include_done_resubmits_done_runs(project: Path, fake: FakeSlurm) -> None:
    assert (
        submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
        == 0
    )
    fake.set_answers('squeue', [{'exit': 0, 'stdout': ''}])
    fake.set_answers(
        'sacct',
        [
            {
                'exit': 0,
                'stdout': (
                    '1000_0|COMPLETED\n1000_1|COMPLETED\n1000_2|COMPLETED\n1000_3|COMPLETED\n'
                ),
            }
        ],
    )
    run_toy(project=project, model='large', seed=1)
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    # why: the done run is skipped, the runs without a folder are failed, not done
    assert [t.run_path for t in jobs[1].tasks] == ['g/small/1', 'g/small/2', 'g/large/2']
    code = submit(
        context=context(project=project, machine_file='cluster.toml'),
        include_done=True,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert code == 0
    jobs = jobs_of(project=project)
    assert [t.run_path for t in jobs[2].tasks] == [
        'g/small/1',
        'g/small/2',
        'g/large/1',
        'g/large/2',
    ]
    assert len(fake.calls('sbatch')) == 3


def test_submit_stops_before_writing_when_a_run_does_not_parse(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    (project / 'bad_load.py').write_text(LOAD_BAD_STUDIES.lstrip('\n'))
    with pytest.raises(ValueError, match='not fields'):
        load_context(
            reference='bad_load:study',
            machine_path=project / 'cluster.toml',
            selection=Selection(),
            project_root=Path.cwd(),
        )
    (project / 'bad_parse.py').write_text(PARSE_BAD_STUDIES.lstrip('\n'))
    with caplog.at_level('INFO'):
        code = submit(
            context=load_context(
                reference='bad_parse:study',
                machine_path=project / 'cluster.toml',
                selection=Selection(),
                project_root=Path.cwd(),
            ),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 1
    assert fake.calls('sbatch') == []
    assert not (project / 'runs' / 's' / '.study' / 'jobs').exists()
    assert 'g/small/1' in caplog.text


def test_dry_run_prints_plan_and_records_nothing(
    project: Path, fake: FakeSlurm, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='cluster.toml'),
            include_done=False,
            dry_run=True,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert fake.calls('sbatch') == []
    assert not (project / 'runs' / 's' / '.study' / 'submissions.jsonl').exists()
    assert (project / 'runs' / 's' / '.study' / 'jobs' / '1.sbatch').is_file()
    text = ' '.join(caplog.text.split())
    assert 'g1 4 partition=' in text
    assert 'toy_script.py model:small --seed 1' in text


@pytest.mark.slow
def test_local_submit_runs_everything(project: Path, caplog: pytest.LogCaptureFixture) -> None:
    first = submit(
        context=context(project=project, machine_file='local.toml'),
        include_done=False,
        dry_run=False,
        extra_args=(),
        sleep=no_sleep,
    )
    assert first == 0
    reports = load_context(
        reference='toy_studies:study',
        machine_path=project / 'local.toml',
        selection=Selection(),
        project_root=Path.cwd(),
    ).reports
    assert {report.state for report in reports} == {StudyState.DONE}
    with caplog.at_level('INFO'):
        code = submit(
            context=context(project=project, machine_file='local.toml'),
            include_done=False,
            dry_run=False,
            extra_args=(),
            sleep=no_sleep,
        )
    assert code == 0
    assert 'nothing to submit' in caplog.text
    assert len(jobs_of(project=project)) == 1
