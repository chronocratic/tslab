"""Grouping runs into numbered job arrays: indices, groups, job files."""

import dataclasses
from pathlib import Path

import pytest

from study.configs import grid, PRESETS
from tslab.core.status import utc_now_iso
from tslab.study import (
    assign_indices,
    expand_runs,
    from_presets,
    JobRecord,
    MachineOverride,
    MachineProfile,
    next_job_file,
    plan_jobs,
    ProgressSnapshot,
    resolve_runs,
    SlurmSettings,
    Study,
    StudyRun,
    TaskRecord,
    validate_jobs,
)


@pytest.fixture
def machine(tmp_path: Path) -> MachineProfile:
    return MachineProfile(
        path=tmp_path / 'machine.toml',
        launcher='slurm',
        output_root=tmp_path / 'out',
        slurm=SlurmSettings(partition='gpu', time='01:00:00'),
        overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'gpus': 1}),),
    )


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid(short_name='g1')})


def study_run(grid_name: str, run_path: str) -> StudyRun:
    return StudyRun(grid=grid_name, values={}, run_path=run_path, settings={})


def task_record(index: int, run_path: str) -> TaskRecord:
    return TaskRecord(
        index=index,
        grid=run_path.split('/', maxsplit=1)[0],
        run_path=run_path,
        command=(),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )


def job_record(job_id: str, tasks: tuple[tuple[int, str], ...]) -> JobRecord:
    return JobRecord(
        kind='run',
        submission_id='s1',
        time=utc_now_iso(),
        launcher='slurm',
        job_id=job_id,
        job_file=1,
        tasks=tuple(task_record(index=i, run_path=path) for i, path in tasks),
    )


def test_groups_by_grid_and_resources(
    study: Study, machine: MachineProfile, tmp_path: Path
) -> None:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    indices = assign_indices(runs=runs, jobs=[])
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=3,
    )
    assert [g.job_file for g in groups] == [3, 4]
    assert groups[0].slurm is not None
    assert groups[0].slurm.gpus == 0
    assert groups[1].slurm is not None
    assert groups[1].slurm.gpus == 1
    assert [[t.index for t in g.tasks] for g in groups] == [[0, 1], [2, 3]]
    assert [g.job_name for g in groups] == ['g1', 'g1']
    task = groups[1].tasks[1]
    assert task.index == 3
    assert task.log_file == machine.output_root / 's' / '.study' / 'logs' / 'g1_3.out'
    assert task.command[:4] == study.grids['g'].command
    assert '--run-path' in task.command
    assert task.attempt == 1


def test_indices_are_stable_when_the_grid_changes() -> None:
    jobs = [
        job_record('7', ((0, 'g/a/1'), (1, 'g/a/2'), (2, 'g/b/1'))),
        job_record('8', ((0, 'h/x/1'),)),
    ]
    runs = [
        study_run('g', 'g/a0/1'),
        study_run('g', 'g/a/1'),
        study_run('g', 'g/a/2'),
        study_run('g', 'g/b/1'),
        study_run('g', 'g/a0/2'),
    ]
    assert assign_indices(runs=runs, jobs=jobs) == {
        'g/a0/1': 3,
        'g/a/1': 0,
        'g/a/2': 1,
        'g/b/1': 2,
        'g/a0/2': 4,
    }


def test_index_at_max_array_size_is_a_task_error(study: Study, machine: MachineProfile) -> None:
    machine = dataclasses.replace(
        machine,
        slurm=SlurmSettings(partition='gpu', time='01:00:00', max_array_size=3),
        overrides=(),
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    indices = {run.run_path: i for i, run in enumerate(runs)}
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=1,
    )
    assert len(groups) == 1
    assert [t.index for t in groups[0].tasks] == [0, 1, 2, 3]
    problems = validate_jobs(study=study, groups=groups)
    assert [path for path, _ in problems] == [runs[3].run_path]
    message = problems[0][1]
    assert 'grid g' in message
    assert 'max_array_size is 3' in message


def test_next_job_file(tmp_path: Path) -> None:
    assert next_job_file(study_folder=tmp_path) == 1
    (tmp_path / 'jobs').mkdir()
    (tmp_path / 'jobs' / '7.sbatch').touch()
    assert next_job_file(study_folder=tmp_path) == 8


def test_extra_args_and_attempts(study: Study, machine: MachineProfile) -> None:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    run_path = runs[0].run_path
    indices = {run.run_path: i for i, run in enumerate(runs)}
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={run_path: ('--run.rerun-from', 'evaluate')},
        attempts={run_path: 2},
        first_job_file=1,
    )
    task = groups[0].tasks[0]
    assert task.command[-2:] == ('--run.rerun-from', 'evaluate')
    assert task.extra_args == ('--run.rerun-from', 'evaluate')
    assert task.attempt == 3
    other = groups[0].tasks[1]
    assert other.extra_args == ()
    assert other.attempt == 1


def test_validate_jobs_reports_run_path_and_message(machine: MachineProfile) -> None:
    study = Study(
        name='s',
        grids={
            'g': grid(
                short_name='g1',
                axes={
                    'recipe': from_presets(PRESETS, names=('ts2vec-paper',), without=('data',)),
                    'dataset': {'Coffee': {'data': 'ucr'}},
                },
            )
        },
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs)
    indices = {run.run_path: i for i, run in enumerate(runs)}
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=1,
    )
    problems = validate_jobs(study=study, groups=groups)
    assert [path for path, _ in problems] == [run.run_path for run in runs]
    assert all('--data.dataset-folder-path' in message for _, message in problems)
