"""Resolving runs on a machine: settings and resources, merged per run."""

from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
import dataclasses
from dataclasses import dataclass, replace
from io import StringIO
import logging
import os
from pathlib import Path
import sys

import tslab
from tslab.study.definition import (
    Grid,
    match_problems,
    matching_in_order,
    merge_overrides,
    merge_settings,
    Study,
    StudyRun,
)
from tslab.study.expansion import expand_setting
from tslab.study.machine import MachineOverride, MachineProfile, SlurmSettings
from tslab.study.records import JobRecord, latest_tasks
from tslab.study.settings import check_paths, settings_to_args, SettingValue

__all__ = [
    'JobGroup',
    'PlannedTask',
    'ResolvedRun',
    'assign_indices',
    'check_machine',
    'expansion_env',
    'next_job_file',
    'plan_jobs',
    'resolve_runs',
    'run_arguments',
    'study_folder',
    'validate_arguments',
    'validate_jobs',
]

_LOGGER = logging.getLogger(__name__)

# why: commands check the machine once, then resolve; a warning is worth saying once
_WARNED: set[tuple[Path, int, str]] = set()


@dataclass(frozen=True, kw_only=True)
class ResolvedRun:
    """One run with the settings and resources of its machine applied.

    Args:
        run: The study run this resolves.
        settings: The settings of the run, then the machine settings, then the
            settings of the matching machine overrides; not yet expanded.
        slurm: The base ``[slurm]`` table with the matching overrides'
            ``slurm`` applied field by field, or ``None`` off SLURM.
    """

    run: StudyRun
    settings: Mapping[str, SettingValue]
    slurm: SlurmSettings | None


def resolve_runs(
    *, study: Study, machine: MachineProfile, runs: Sequence[StudyRun]
) -> list[ResolvedRun]:
    """Every run with the settings and resources of the machine applied.

    Machine settings paths are checked against every grid they can apply to;
    an override that names an axis, grid or value this study lacks is logged
    once as a warning and still applies wherever it matches.

    Args:
        study: The study the runs came from.
        machine: The machine the runs run on.
        runs: The runs to resolve, e.g. ``expand_runs``' output.

    Returns:
        The resolved runs, in the order of ``runs``.

    Raises:
        ValueError: If a machine setting path is not a field of a grid it
            applies to; the error names the profile file, the override index
            or ``settings``, the grid and the paths.
    """
    check_machine(study=study, machine=machine)
    return [
        ResolvedRun(
            run=run,
            settings=merge_overrides(
                base=merge_settings(base=run.settings, update=machine.settings),
                overrides=[(override.match, override.settings) for override in machine.overrides],
                grid_name=run.grid,
                values=run.values,
            ),
            slurm=_slurm_for(
                base=machine.slurm,
                overrides=machine.overrides,
                grid_name=run.grid,
                values=run.values,
            ),
        )
        for run in runs
    ]


def check_machine(*, study: Study, machine: MachineProfile) -> None:
    """Check the machine settings against the grids; warn about dead overrides.

    Every command that reads a machine should call this, not only the ones
    that submit: a profile that does not fit the study is a mistake to hear
    about early. A warning is logged once per override.

    Args:
        study: The study the machine runs.
        machine: The machine profile to check.

    Raises:
        ValueError: If a machine setting path is not a field of a grid it
            applies to; the error names the profile file, the override index
            or ``settings``, the grid and the paths.
    """
    _settings_problems(study=study, machine=machine)
    _override_warnings(study=study, machine=machine)


def _settings_problems(*, study: Study, machine: MachineProfile) -> None:
    for grid_name, grid in study.grids.items():
        bad = check_paths(config_class=grid.config, settings=machine.settings)
        if bad:
            message = (
                f'{machine.path}: settings set paths that are not fields of '
                f'{grid.config.__name__} (grid {grid_name!r}): {", ".join(bad)}'
            )
            raise ValueError(message)
    for position, override in enumerate(machine.overrides):
        for grid_name, grid in _override_grids(
            study=study, grid_names=override.match.get('grid', ()), match=override.match
        ):
            bad = check_paths(config_class=grid.config, settings=override.settings)
            if bad:
                message = (
                    f'{machine.path}: override {position} sets paths that are not fields '
                    f'of {grid.config.__name__} (grid {grid_name!r}): {", ".join(bad)}'
                )
                raise ValueError(message)


def _override_grids(
    *, study: Study, grid_names: Sequence[str], match: Mapping[str, object]
) -> list[tuple[str, Grid]]:
    """The grids the settings of an override apply to, for path checking."""
    axes = [key for key in match if key != 'grid']
    return [
        (name, grid)
        for name, grid in study.grids.items()
        if (not grid_names or name in grid_names) and all(axis in grid.axes for axis in axes)
    ]


def _override_warnings(*, study: Study, machine: MachineProfile) -> None:
    for position, override in enumerate(machine.overrides):
        problems = match_problems(grids=study.grids, match=override.match)
        text = '; '.join(problems)
        if problems and (machine.path, position, text) not in _WARNED:
            _WARNED.add((machine.path, position, text))
            _LOGGER.warning(
                'machine override %d of %s matches nothing in this study: %s',
                position,
                machine.path,
                text,
            )


def _slurm_for(
    *,
    base: SlurmSettings | None,
    overrides: Sequence[MachineOverride],
    grid_name: str,
    values: Mapping[str, str],
) -> SlurmSettings | None:
    if base is None:
        return None  # why: no [slurm] table means no scheduler; overrides cannot add one
    merged: dict[str, object] = {
        item.name: getattr(base, item.name) for item in dataclasses.fields(base)
    }
    order = matching_in_order(
        matches=[override.match for override in overrides], grid_name=grid_name, values=values
    )
    for position in order:
        merged.update(overrides[position].slurm)
    return SlurmSettings(**merged)  # ty: ignore[invalid-argument-type]


def run_arguments(
    *,
    grid: Grid,
    settings: Mapping[str, SettingValue],
    env: Mapping[str, str],
    placeholders: Mapping[str, str],
) -> list[str]:
    """The flags of one run: every setting expanded, then turned to tyro flags.

    Args:
        grid: The grid of the run; its ``config`` class the flags are read into.
        settings: The resolved settings of the run, not yet expanded.
        env: The variable names to values for expansion.
        placeholders: The placeholder names to values for this run.

    Returns:
        The flags, in the order of ``settings_to_args``.

    Raises:
        ValueError: If a value names an unset variable or an unknown
            placeholder.
        ValueError: If a setting path is not a field of the config of the grid.
    """
    expanded = {
        path: expand_setting(value=value, env=env, values=placeholders)
        for path, value in settings.items()
    }
    return settings_to_args(config_class=grid.config, settings=expanded)


def validate_arguments(*, config_class: type, arguments: Sequence[str]) -> str | None:
    """Whether tyro reads the arguments back into the config class.

    Args:
        config_class: The config dataclass the arguments are for.
        arguments: The flags plus any pass-through arguments.

    Returns:
        ``None`` when the arguments parse, else the error text of tyro or of
        the config class.
    """
    buffer = StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            tslab.parse_config(config_class=config_class, args=list(arguments))
    except SystemExit:
        return buffer.getvalue()
    except (ValueError, TypeError) as error:
        # why: the __post_init__ of a config class can reject a value tyro accepted
        return f'{buffer.getvalue()}{type(error).__name__}: {error}'
    return None


def expansion_env(*, machine: MachineProfile) -> dict[str, str]:
    """The variables a run expands: the process environment, profile wins."""
    return {**os.environ, **machine.env}


def _usable_cpus() -> int:
    """The CPUs this process may run on; a local launch runs its tasks here."""
    # why: the affinity mask honours taskset and container limits; macOS has no such call
    if sys.platform == 'linux':
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass(frozen=True, kw_only=True)
class PlannedTask:
    """One run prepared for submission.

    Args:
        resolved: The resolved run this is a task for.
        index: The array index of the run, the same in every attempt.
        log_file: The file the output of the task is appended to.
        command: The full command: the one of the grid, the flags, the extra args.
        extra_args: The pass-through arguments, last in ``command``.
        attempt: The attempt number of this submission.
        error: A planning problem, e.g. an index past ``max_array_size``;
            the task is still planned, but ``validate_jobs`` reports it.
        uncounted_attempts: Earlier attempts that ended ``PREEMPTED`` or ``NODE_FAIL``.
        uncounted_in_row: How many of the latest attempts in a row ended that way.
    """

    resolved: ResolvedRun
    index: int
    log_file: Path
    command: tuple[str, ...]
    extra_args: tuple[str, ...]
    attempt: int
    error: str | None = None
    uncounted_attempts: int = 0
    uncounted_in_row: int = 0


@dataclass(frozen=True, kw_only=True)
class JobGroup:
    """The runs submitted as one job: the runs of one grid with one set of resources.

    Args:
        grid: The grid name the runs of the group came from.
        job_name: The ``short_name`` of the grid; it names the job and its logs.
        slurm: The resources of the job; ``None`` when the machine is local.
        job_file: The ``jobs/<n>.tasks`` / ``jobs/<n>.sbatch`` number.
        tasks: The tasks of the group, sorted by index.
    """

    grid: str
    job_name: str
    slurm: SlurmSettings | None
    job_file: int
    tasks: tuple[PlannedTask, ...]


def assign_indices(*, runs: Sequence[StudyRun], jobs: Sequence[JobRecord]) -> dict[str, int]:
    """The array index of every run: recorded ones keep theirs, new ones append.

    Args:
        runs: The runs of the study, in expansion order.
        jobs: The job records of the study, in submission order.

    Returns:
        The run path to index. A run with a task record keeps the index of
        that record; the others get, in the order of ``runs``, one plus the largest
        index their grid already uses in ``jobs`` or was assigned earlier.
    """
    latest = latest_tasks(jobs=jobs)
    highest: dict[str, int] = {}
    for _, task in latest.values():
        highest[task.grid] = max(highest.get(task.grid, -1), task.index)
    indices: dict[str, int] = {}
    for run in runs:
        recorded = latest.get(run.run_path)
        if recorded is not None:
            indices[run.run_path] = recorded[1].index
            continue
        indices[run.run_path] = highest.get(run.grid, -1) + 1
        highest[run.grid] = indices[run.run_path]
    return indices


def study_folder(*, machine: MachineProfile, study: Study) -> Path:
    """The hidden folder of the study: ``<output_root>/<study>/.study``."""
    return machine.output_root / study.name / '.study'


def next_job_file(*, study_folder: Path) -> int:
    """The next job file number: one past the largest on disk, else 1."""
    numbers = 1
    for suffix in ('.sbatch', '.tasks'):
        for path in (study_folder / 'jobs').glob(f'*{suffix}'):
            try:
                number = int(path.name.removesuffix(suffix))
            except ValueError:
                continue
            numbers = max(numbers, number + 1)
    return numbers


def _task_error(*, grid_name: str, run_path: str, index: int, max_array_size: int) -> str:
    return (
        f'grid {grid_name}: run {run_path} needs array index {index}, '
        f'but max_array_size is {max_array_size} (the MaxArraySize of the cluster)'
    )


def plan_jobs(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study: Study,
    machine: MachineProfile,
    resolved: Sequence[ResolvedRun],
    indices: Mapping[str, int],
    extra_args: Mapping[str, tuple[str, ...]],
    attempts: Mapping[str, int],
    first_job_file: int,
    uncounted: Mapping[str, tuple[int, int]] | None = None,
) -> list[JobGroup]:
    """The runs grouped into numbered jobs, ready for validation and launch.

    Runs group by (grid, slurm) in first-appearance order; a group is never
    split by count, so a group of tasks past ``max_array_size`` is reported
    by ``validate_jobs`` instead of planned around.

    Args:
        study: The study the runs came from.
        machine: The machine the runs run on.
        resolved: The runs resolved on ``machine``, in run order.
        indices: The run path to array index, e.g. from ``assign_indices``.
        extra_args: The run path to pass-through arguments, last in the
            command.
        attempts: The run path to attempts already made; 0 when absent.
        first_job_file: The first job file number to number groups from.
        uncounted: The run path to ``(uncounted attempts, uncounted in a row)``,
            see ``TaskRecord``; ``(0, 0)`` when absent.

    Returns:
        The groups, numbered from ``first_job_file`` in first-appearance
        order.
    """
    folder = study_folder(machine=machine, study=study)
    local_cpus = _usable_cpus()
    groups: list[JobGroup] = []
    lookup: dict[tuple[str, SlurmSettings | None], int] = {}
    next_file = first_job_file
    tasks_of: dict[int, list[PlannedTask]] = {}
    env = expansion_env(machine=machine)
    counters = uncounted or {}
    for run in resolved:
        grid = study.grids[run.run.grid]
        key = (run.run.grid, run.slurm)
        if key not in lookup:
            lookup[key] = len(groups)
            tasks_of[lookup[key]] = []
            groups.append(
                JobGroup(
                    grid=run.run.grid,
                    job_name=grid.short_name,
                    slurm=run.slurm,
                    job_file=next_file,
                    tasks=(),
                )
            )
            next_file += 1
        index = indices[run.run.run_path]
        args = extra_args.get(run.run.run_path, ())
        log_file = folder / 'logs' / f'{grid.short_name}_{index}.out'
        try:
            arguments = run_arguments(
                grid=grid,
                settings=run.settings,
                env=env,
                placeholders={
                    'output_root': str(machine.output_root),
                    'study': study.name,
                    'run_path': run.run.run_path,
                    'log_file': str(log_file),
                    'cpus': str(local_cpus if run.slurm is None else run.slurm.cpus_per_task),
                },
            )
        except (ValueError, TypeError) as error:
            arguments = []
            error_text = str(error)
        else:
            error_text = None
        if run.slurm is not None and index >= run.slurm.max_array_size:
            size_error = _task_error(
                grid_name=run.run.grid,
                run_path=run.run.run_path,
                index=index,
                max_array_size=run.slurm.max_array_size,
            )
            error_text = size_error if error_text is None else f'{error_text}\n{size_error}'
        task = PlannedTask(
            resolved=run,
            index=index,
            log_file=log_file,
            command=(*grid.command, *arguments, *args),
            extra_args=args,
            attempt=attempts.get(run.run.run_path, 0) + 1,
            error=error_text,
            uncounted_attempts=counters.get(run.run.run_path, (0, 0))[0],
            uncounted_in_row=counters.get(run.run.run_path, (0, 0))[1],
        )
        tasks_of[lookup[key]].append(task)
    return [
        replace(group, tasks=tuple(sorted(tasks_of[position], key=lambda task: task.index)))
        for position, group in enumerate(groups)
    ]


def validate_jobs(*, study: Study, groups: Sequence[JobGroup]) -> list[tuple[str, str]]:
    """The runs whose arguments do not parse, as (run path, message).

    Args:
        study: The study the groups came from.
        groups: The planned groups.

    Returns:
        One pair per problem, in group and task order; a task whose
        arguments could not even be built reports its building error.
    """
    problems: list[tuple[str, str]] = []
    for group in groups:
        for task in group.tasks:
            run_path = task.resolved.run.run_path
            if task.error is not None:
                problems.append((run_path, task.error))
                continue
            error = validate_arguments(
                config_class=study.grids[group.grid].config,
                arguments=task.command[len(study.grids[group.grid].command) :],
            )
            if error is not None:
                problems.append((run_path, error.strip()))
    return problems
