"""Grouping runs into numbered job arrays: indices, groups, job files."""

import dataclasses
from pathlib import Path, PurePosixPath

import pytest

from tests.lab.unit.study.configs import grid, PRESETS
from tslab.lab.core.status import utc_now_iso
from tslab.lab.enums import LauncherKind
from tslab.lab.study import from_presets, Study, StudyRun
from tslab.lab.study.code_snapshot import CodeSnapshot
from tslab.lab.study.definition import expand_runs, setting_axis
from tslab.lab.study.launchers.tables import JOB_RECORDS
from tslab.lab.study.machine import MachineOverride, MachineProfile, SlurmSettings
from tslab.lab.study.paths import next_job_file
from tslab.lab.study.plan import assign_indices, JobGroup, plan_jobs, resolve_runs, validate_jobs
from tslab.lab.study.records import JobKind, JobRecord, ProgressSnapshot, TaskRecord


@pytest.fixture
def machine(tmp_path: Path) -> MachineProfile:
    return MachineProfile(
        path=tmp_path / 'machine.toml',
        launcher=LauncherKind.SLURM,
        output_root=tmp_path / 'out',
        slurm=SlurmSettings(partition='gpu', time='01:00:00'),
        overrides=(MachineOverride(match={'recipe': ('cost-paper',)}, slurm={'gpus': 1}),),
    )


@pytest.fixture
def study() -> Study:
    return Study(name='s', grids={'g': grid(short_name='g1')})


def study_run(grid_name: str, run_path: str) -> StudyRun:
    return StudyRun(grid=grid_name, values={}, run_path=PurePosixPath(run_path), settings={})


def task_record(index: int, run_path: str) -> TaskRecord:
    return TaskRecord(
        index=index,
        grid=run_path.split('/', maxsplit=1)[0],
        run_path=PurePosixPath(run_path),
        command=(),
        extra_args=(),
        resources={},
        attempt=1,
        progress=ProgressSnapshot(done_instances=0, newest_checkpoint_mtime=None),
    )


def job_record(job_id: str, tasks: tuple[tuple[int, str], ...]) -> JobRecord:
    return JOB_RECORDS[LauncherKind.SLURM](
        kind=JobKind.RUN,
        submission_id='s1',
        time=utc_now_iso(),
        launcher=LauncherKind.SLURM,
        job_id=job_id,
        job_file=1,
        tasks=tuple(task_record(index=i, run_path=path) for i, path in tasks),
    )


def test_groups_by_grid_and_resources(
    study: Study, machine: MachineProfile, tmp_path: Path
) -> None:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1, environment={})
    indices = assign_indices(study=study, runs=runs, jobs=[])
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=3,
        environment={},
    )
    assert [g.job_file for g in groups] == [3, 4]
    assert groups[0].resources.describe() == 'partition=gpu, time=01:00:00, cpus=1'
    assert groups[0].resources.max_tasks_per_job == 1000
    assert groups[1].resources.describe() == 'partition=gpu, time=01:00:00, cpus=1, gpus=1'
    assert [[t.index for t in g.tasks] for g in groups] == [[0, 1], [2, 3]]
    assert [g.job_name for g in groups] == ['s_g1', 's_g1']
    task = groups[1].tasks[1]
    assert task.index == 3
    assert task.log_file == machine.output_root / 's' / '.study' / 'logs' / 's_g1_3.out'
    assert task.command[:4] == study.grids['g'].command
    assert '--run-path' in task.command
    assert task.attempt == 1


def test_plan_jobs_splits_groups_by_code_snapshot(
    study: Study, machine: MachineProfile, tmp_path: Path
) -> None:
    plain = dataclasses.replace(machine, overrides=())
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=plain, runs=runs, local_cpus=1, environment={})
    indices = assign_indices(study=study, runs=runs, jobs=[])

    def snapshot(folder: Path) -> CodeSnapshot:
        return CodeSnapshot(
            folder=folder,
            hash='0123456789abcdef',
            commit='a' * 40,
            dirty=False,
            untracked_copied=False,
        )

    first = snapshot(tmp_path / 'out' / '.tslab-code-snapshots' / '0123456789abcdef')
    second = snapshot(tmp_path / 'out' / '.tslab-code-snapshots' / 'fedcba9876543210')
    code_snapshots = {runs[0].run_path: first, runs[1].run_path: second, runs[3].run_path: second}
    groups = plan_jobs(
        study=study,
        machine=plain,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
        code_snapshots=code_snapshots,
    )
    assert [group.code_snapshot for group in groups] == [first, second, None]
    assert [group.job_file for group in groups] == [1, 2, 3]
    assert [[task.index for task in group.tasks] for group in groups] == [[0], [1, 3], [2]]


def test_plan_jobs_without_code_snapshots_groups_as_before(
    study: Study, machine: MachineProfile
) -> None:
    plain = dataclasses.replace(machine, overrides=())
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=plain, runs=runs, local_cpus=1, environment={})
    indices = assign_indices(study=study, runs=runs, jobs=[])
    groups = plan_jobs(
        study=study,
        machine=plain,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )
    assert len(groups) == 1
    assert groups[0].code_snapshot is None
    assert [task.index for task in groups[0].tasks] == [0, 1, 2, 3]


def test_indices_are_stable_when_the_grid_changes(study: Study) -> None:
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
    assert assign_indices(study=study, runs=runs, jobs=jobs) == {
        PurePosixPath('g/a0/1'): 3,
        PurePosixPath('g/a/1'): 0,
        PurePosixPath('g/a/2'): 1,
        PurePosixPath('g/b/1'): 2,
        PurePosixPath('g/a0/2'): 4,
    }


def test_index_at_max_array_size_is_a_task_error(study: Study, machine: MachineProfile) -> None:
    machine = dataclasses.replace(
        machine,
        slurm=SlurmSettings(partition='gpu', time='01:00:00', max_array_size=3),
        overrides=(),
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1, environment={})
    indices = {run.run_path: i for i, run in enumerate(runs)}
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
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
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1, environment={})
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
        environment={},
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
                    'recipe': from_presets(
                        presets=PRESETS, names=('ts2vec-paper',), without=('data',)
                    ),
                    'dataset': {'Coffee': {'data': 'ucr'}},
                },
            )
        },
    )
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1, environment={})
    indices = {run.run_path: i for i, run in enumerate(runs)}
    groups = plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=indices,
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )
    problems = validate_jobs(study=study, groups=groups)
    assert [path for path, _ in problems] == [run.run_path for run in runs]
    assert all('--data.dataset-folder-path' in message for _, message in problems)


def _plan(*, study: Study, machine: MachineProfile) -> list[JobGroup]:
    runs = expand_runs(study=study)
    resolved = resolve_runs(study=study, machine=machine, runs=runs, local_cpus=1, environment={})
    return plan_jobs(
        study=study,
        machine=machine,
        resolved=resolved,
        indices=assign_indices(study=study, runs=runs, jobs=[]),
        extra_args={},
        attempts={},
        first_job_file=1,
        environment={},
    )


def test_default_job_name_keeps_one_group_per_grid(machine: MachineProfile) -> None:
    study = Study(name='s', grids={'g': grid(short_name='g1')})
    groups = _plan(study=study, machine=machine)
    assert [g.job_name for g in groups] == ['s_g1', 's_g1']  # split by resources only, as today
    assert all(t.log_file.name == f's_g1_{t.index}.out' for g in groups for t in g.tasks)


def test_job_name_template_splits_groups(machine: MachineProfile) -> None:
    study = Study(name='s', grids={'g': grid(short_name='g1', job_name='{recipe}_{short_name}')})
    groups = _plan(study=study, machine=machine)
    assert sorted(g.job_name for g in groups) == ['cost-paper_g1', 'ts2vec-paper_g1']
    assert [[t.index for t in g.tasks] for g in groups] == [[0, 1], [0, 1]]
    assert all(t.log_file.name == f'{g.job_name}_{t.index}.out' for g in groups for t in g.tasks)


def _small_arrays(
    *, machine: MachineProfile, job_name: str = '{recipe}_{short_name}'
) -> tuple[Study, MachineProfile]:
    """Six runs in a grid with max_array_size 3: two recipes times three seeds."""
    seeds = setting_axis(path='seed', values=(1, 2, 3))
    study = Study(
        name='s',
        grids={'g': grid(short_name='g1', job_name=job_name, axes={**grid().axes, 'seed': seeds})},
    )
    slurm = SlurmSettings(partition='gpu', time='01:00:00', max_array_size=3)
    return study, dataclasses.replace(machine, slurm=slurm, overrides=())


def test_grid_past_max_array_size_plans_when_job_name_splits_it(machine: MachineProfile) -> None:
    study, small = _small_arrays(machine=machine)
    groups = _plan(study=study, machine=small)
    assert [[t.index for t in g.tasks] for g in groups] == [[0, 1, 2], [0, 1, 2]]
    assert validate_jobs(study=study, groups=groups) == []
    log_files = [t.log_file for g in groups for t in g.tasks]
    assert len(set(log_files)) == len(log_files)


def test_one_array_past_max_array_size_still_fails(machine: MachineProfile) -> None:
    study, small = _small_arrays(machine=machine, job_name='{short_name}')
    groups = _plan(study=study, machine=small)
    problems = validate_jobs(study=study, groups=groups)
    assert len(problems) == 3
    assert 'needs array index 3, but max_array_size is 3' in problems[0][1]


def test_indices_per_job_name_stay_stable_across_submits(machine: MachineProfile) -> None:
    study, _ = _small_arrays(machine=machine)
    runs = expand_runs(study=study)
    first = assign_indices(study=study, runs=runs[:4], jobs=[])
    recorded = [job_record('7', tuple((first[r.run_path], str(r.run_path)) for r in runs[:4]))]
    again = assign_indices(study=study, runs=runs, jobs=recorded)
    assert {path: again[path] for path in first} == first
    assert sorted(again.values()) == [0, 0, 1, 1, 2, 2]
    one = runs[2]
    assert assign_indices(study=study, runs=[one], jobs=recorded) == {
        one.run_path: first[one.run_path]
    }


def test_removed_run_keeps_its_index_reserved(machine: MachineProfile) -> None:
    study, _ = _small_arrays(machine=machine)
    recorded = [job_record('7', ((5, 'g/gone/1'),))]
    indices = assign_indices(study=study, runs=expand_runs(study=study), jobs=recorded)
    assert min(indices.values()) == 6
