"""Resolving runs on a machine: settings and resources, merged per run."""

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
import logging
import os
from pathlib import Path, PurePosixPath
import pickle
from typing import TYPE_CHECKING

import tslab.lab
from tslab.lab.enums import AfterCondition, Satisfy
from tslab.lab.study.code_snapshot import CodeSnapshot
from tslab.lab.study.definition import (
    chained_grids,
    check_paths,
    expand_runs,
    expand_setting,
    Grid,
    grid_order,
    match_axes,
    match_problems,
    merge_overrides,
    merge_settings,
    settings_to_args,
    SettingValue,
    Study,
    StudyRun,
    upstream_runs,
)
from tslab.lab.study.definition.settings import non_machine_paths
from tslab.lab.study.machine import MachineOverride, MachineProfile
from tslab.lab.study.paths import log_file_of, study_folder
from tslab.lab.study.plan_after import merge_corr_groups
from tslab.lab.study.records import JobRecord, latest_tasks, TaskRecord

if TYPE_CHECKING:
    from tslab.lab.study.launchers.common.types import Resources
    from tslab.lab.study.plan_after import AfterTarget

__all__ = [
    'JobGroup',
    'PlannedTask',
    'ResolvedRun',
    'assign_indices',
    'check_machine',
    'expansion_env',
    'pin_conflicts',
    'plan_jobs',
    'resolve_runs',
    'run_arguments',
    'validate_arguments',
    'validate_jobs',
]

_LOGGER = logging.getLogger(__name__)

PARALLEL_VALIDATION_FROM = 700
"""From this many tasks on, ``validate_jobs`` checks the arguments in worker processes.

Each worker imports the config classes on start, a few seconds in all; below
this count one process is faster.
"""

_CPU_SHARE = 0.5
"""The share of the usable CPUs the argument checks may take; a login node is shared."""

_CHUNKS_PER_WORKER = 4
"""Argument checks are handed to each worker in about this many batches."""


@dataclass(frozen=True, kw_only=True)
class ResolvedRun:
    """One run with the settings and resources of its machine applied.

    Args:
        run: The study run this resolves.
        settings: The settings of the run, then the machine settings, then the
            settings of the matching machine overrides; not yet expanded.
        resources: The resources of the run, from the backend of its machine.
    """

    run: StudyRun
    settings: Mapping[str, SettingValue]
    resources: 'Resources'


def resolve_runs(  # noqa: PLR0913 # why: keyword-only; each input is documented below
    *,
    study: Study,
    machine: MachineProfile,
    runs: Sequence[StudyRun],
    local_cpus: int,
    environment: Mapping[str, str],
    warned: set[str] | None = None,
) -> list[ResolvedRun]:
    """Every run with the settings and resources of the machine applied.

    Machine settings paths are checked against every grid they can apply to;
    an override that names an axis, grid or value this study lacks is logged
    once as a warning and still applies wherever it matches.

    Args:
        study: The study the runs came from.
        machine: The machine the runs run on.
        runs: The runs to resolve, e.g. ``expand_runs``' output.
        local_cpus: The CPUs available to a local launch.
        environment: The process environment, read for ``CUDA_VISIBLE_DEVICES`` by a
            local profile that discovers its GPUs.
        warned: Forwarded to ``check_machine`` so a command that already
            checked the machine does not hear the same warning twice.

    Returns:
        The resolved runs, in the order of ``runs``.

    Raises:
        ValueError: If a machine setting path is not a field of a grid it
            applies to; the error names the profile file, the override index
            or ``settings``, the grid and the paths.
    """
    check_machine(study=study, machine=machine, warned=warned)
    # why: the launcher door imports JobGroup from this module
    from tslab.lab.study.launchers.tables import backend_for  # noqa: PLC0415

    backend = backend_for(machine=machine)
    return [
        ResolvedRun(
            run=run,
            settings=merge_overrides(
                base=merge_settings(base=run.settings, update=machine.settings),
                overrides=[(override.match, override.settings) for override in machine.overrides],
                grid_name=run.grid,
                values=run.values,
            ),
            resources=backend.resources(
                machine=machine,
                overrides=machine.overrides,
                grid_name=run.grid,
                values=run.values,
                local_cpus=local_cpus,
                environment=environment,
            ),
        )
        for run in runs
    ]


def check_machine(*, study: Study, machine: MachineProfile, warned: set[str] | None = None) -> None:
    """Check the machine settings against the grids; warn about dead overrides.

    Every command that reads a machine should call this, not only the ones
    that submit: a profile that does not fit the study is a mistake to hear
    about early. A warning is logged once per override per call; a command
    that checks the machine twice passes the same ``warned`` set, so the
    warning is said once.

    Args:
        study: The study the machine runs.
        machine: The machine profile to check.
        warned: Texts already warned about in this command, so a repeated
            check does not repeat the warning.

    Raises:
        ValueError: If a machine setting path is not a field of a grid it
            applies to; the error names the profile file, the override index
            or ``settings``, the grid and the paths.
    """
    _settings_problems(study=study, machine=machine)
    seen = warned if warned is not None else set()
    for text in _override_warnings(study=study, machine=machine):
        if text in seen:
            continue
        seen.add(text)
        _LOGGER.warning('%s', text)


def _settings_problems(*, study: Study, machine: MachineProfile) -> None:
    for grid_name, grid in study.grids.items():
        bad = check_paths(config_class=grid.config, settings=machine.settings)
        if bad:
            message = (
                f'{machine.path}: settings set paths that are not fields of '
                f'{grid.config.__name__} (grid {grid_name!r}): {", ".join(bad)}'
            )
            raise ValueError(message)
        _check_machine_settings(
            grid_name=grid_name,
            grid=grid,
            machine=machine,
            settings=machine.settings,
            location='[settings]',
        )
    for position, override in enumerate(machine.overrides):
        _override_problems(study=study, machine=machine, position=position, override=override)


def _override_problems(
    *, study: Study, machine: MachineProfile, position: int, override: MachineOverride
) -> None:
    """Raise for the first grid an override sets a path on that is not a field of it."""
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
        _check_machine_settings(
            grid_name=grid_name,
            grid=grid,
            machine=machine,
            settings=override.settings,
            location=f'overrides[{position}].settings',
        )


def _check_machine_settings(
    *,
    grid_name: str,
    grid: Grid,
    machine: MachineProfile,
    settings: Mapping[str, SettingValue],
    location: str,
) -> None:
    bad = non_machine_paths(config_class=grid.config, settings=settings)
    if not bad:
        return
    message = (
        f'{machine.path}: {location} sets {", ".join(bad)}, which '
        f'{"is" if len(bad) == 1 else "are"} not a machine setting of '
        f'{grid.config.__name__} (grid {grid_name!r}); '
        f'set {"it" if len(bad) == 1 else "them"} in the study instead'
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


def _override_warnings(*, study: Study, machine: MachineProfile) -> tuple[str, ...]:
    """The warning text of each override that matches nothing in this study."""
    warnings: list[str] = []
    for position, override in enumerate(machine.overrides):
        problems = match_problems(grids=study.grids, match=override.match)
        if problems:
            warnings.append(
                f'machine override {position} of {machine.path} '
                f'matches nothing in this study: {"; ".join(problems)}'
            )
    return tuple(warnings)


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


def pin_conflicts(
    *,
    before: Mapping[str, SettingValue],
    after: Mapping[str, SettingValue],
    env: Mapping[str, str],
    placeholders: Mapping[str, str],
) -> list[tuple[str, SettingValue, SettingValue]]:
    """The study-set paths changed by machine settings, after expansion.

    Args:
        before: Settings after study and grid merges, before machine settings.
        after: Settings after machine settings and machine overrides.
        env: The environment used to expand values.
        placeholders: The run placeholders used to expand values.

    Returns:
        Conflicts as ``(path, study value, machine value)`` in ``before`` order.
    """
    conflicts: list[tuple[str, SettingValue, SettingValue]] = []
    for path, study_value in before.items():
        if path not in after:
            continue
        machine_value = after[path]
        expanded_study = expand_setting(value=study_value, env=env, values=placeholders)
        expanded_machine = expand_setting(value=machine_value, env=env, values=placeholders)
        if str(expanded_study) == str(expanded_machine):
            continue
        conflicts.append((path, study_value, machine_value))
    return conflicts


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
            tslab.lab.parse_config(config_class=config_class, args=list(arguments))
    except SystemExit:
        return buffer.getvalue()
    except (ValueError, TypeError) as error:
        # why: the __post_init__ of a config class can reject a value tyro accepted
        return f'{buffer.getvalue()}{type(error).__name__}: {error}'
    return None


def expansion_env(*, machine: MachineProfile, environment: Mapping[str, str]) -> dict[str, str]:
    """The variables a run expands: the process environment, profile wins."""
    return {**environment, **machine.env}


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
    """The runs submitted as one job: runs of one grid with one job name and resources.

    Args:
        grid: The grid name the runs of the group came from.
        job_name: The name of the job, the grid ``job_name`` template filled in.
        resources: The resources of the job, as declared by its backend.
        job_file: The ``jobs/<n>.tasks`` / ``jobs/<n>.sbatch`` number.
        tasks: The tasks of the group, sorted by index.
        after: The upstream dependencies of each task in the group.
        chained: Whether the grid participates in a dependency chain.
        code_snapshot: The code snapshot the jobs of the group run from;
            ``None`` means the live code. A group has at most one, so one
            job file carries one ``PYTHONPATH``.
    """

    grid: str
    job_name: str
    resources: 'Resources'
    job_file: int
    tasks: tuple[PlannedTask, ...]
    after: tuple['AfterTarget', ...] = ()
    chained: bool = False
    code_snapshot: CodeSnapshot | None = None


def assign_indices(
    *, study: Study, runs: Sequence[StudyRun], jobs: Sequence[JobRecord]
) -> dict[PurePosixPath, int]:
    """The array index of every run: recorded ones keep theirs, new ones append.

    Indices are numbered per job name, not per grid, so ``max_array_size``
    limits the size of one array and a grid split by ``job_name`` can hold
    more runs than that.

    Args:
        study: The study the runs came from; it gives every run its job name.
        runs: The runs to index, in expansion order.
        jobs: The job records of the study, in submission order.

    Returns:
        The run path to index. A run with a task record keeps the index of
        that record; the others get, in the order of ``runs``, one plus the largest
        index their job name already uses in ``jobs`` or was assigned earlier.
        A recorded run that left the study holds its index in every job name
        of its grid, so no new run reuses its log file.
    """
    names = {run.run_path: study.job_name_of(run=run) for run in [*expand_runs(study=study), *runs]}
    latest = latest_tasks(jobs=jobs)
    taken: dict[str, set[int]] = {}
    removed: dict[str, set[int]] = {}
    for run_path, (_, task) in latest.items():
        name = names.get(run_path)
        if name is None:
            removed.setdefault(task.grid, set()).add(task.index)
        else:
            taken.setdefault(name, set()).add(task.index)
    ordered = _upstream_first(study=study, runs=runs)
    indices: dict[PurePosixPath, int] = {}
    for run in ordered:
        recorded = latest.get(run.run_path)
        if recorded is not None:
            indices[run.run_path] = recorded[1].index
            continue
        name = names[run.run_path]
        blocked = taken.setdefault(name, set()) | removed.get(run.grid, set())
        index = _upstream_index(study=study, run=run, latest=latest, indices=indices)
        if index is None or index in blocked:
            index = max(blocked, default=-1) + 1
        taken[name].add(index)
        indices[run.run_path] = index
    return {run.run_path: indices[run.run_path] for run in runs}


def _upstream_first(*, study: Study, runs: Sequence[StudyRun]) -> list[StudyRun]:
    """The runs with upstream grids first; unchanged for a study without chained grids."""
    if not chained_grids(study=study):
        return list(runs)
    rank = {name: position for position, name in enumerate(grid_order(study=study))}
    return sorted(runs, key=lambda run: rank[run.grid])


def _upstream_index(
    *,
    study: Study,
    run: StudyRun,
    latest: Mapping[PurePosixPath, tuple[JobRecord, TaskRecord]],
    indices: Mapping[PurePosixPath, int],
) -> int | None:
    """The array index of the one upstream run a one-to-one downstream run pairs with."""
    grid = study.grids[run.grid]
    if len(grid.after) != 1:
        return None
    (after,) = grid.after
    paired = match_axes(study=study, grid_name=run.grid, after=after)
    plain = after.condition is AfterCondition.OK and after.satisfy is Satisfy.ALL
    if not plain or not set(grid.axes) <= set(paired):
        return None
    upstream = upstream_runs(study=study, run=run, after=after)
    if len(upstream) != 1:
        return None
    path = upstream[0].run_path
    if path in indices:
        return indices[path]
    return latest[path][1].index if path in latest else None


def _task_error(*, grid_name: str, run_path: PurePosixPath, index: int, max_array_size: int) -> str:
    return (
        f'grid {grid_name}: run {run_path} needs array index {index}, '
        f'but max_array_size is {max_array_size} (the MaxArraySize of the cluster)'
    )


def _after_placeholders(
    *, study: Study, machine: MachineProfile, grid: Grid, run: StudyRun, project_root: Path | None
) -> tuple[dict[str, str], dict[str, tuple[StudyRun, ...]]]:
    """The single upstream run paths a downstream run can use as placeholders."""
    upstream_by_grid = {
        after.grid: upstream_runs(study=study, run=run, after=after) for after in grid.after
    }
    output_root = _absolute_output_root(output_root=machine.output_root, project_root=project_root)
    values = {
        f'after:{grid_name}': str(output_root / study.name / runs[0].run_path)
        for grid_name, runs in upstream_by_grid.items()
        if len(runs) == 1
    }
    if len(grid.after) == 1 and f'after:{grid.after[0].grid}' in values:
        values['after'] = values[f'after:{grid.after[0].grid}']
    return values, upstream_by_grid


def _absolute_output_root(*, output_root: Path, project_root: Path | None) -> Path:
    """Resolve a relative output root against the directory of the project."""
    if output_root.is_absolute():
        return output_root
    if project_root is None:
        message = 'project_root is required for {after} with a relative output_root'
        raise ValueError(message)
    return project_root / output_root


def _after_placeholder_error(
    *, error: str, run: StudyRun, upstream_by_grid: Mapping[str, tuple[StudyRun, ...]]
) -> str:
    """Name a missing or multi-run after placeholder as a planning problem."""
    if error == "'after' is not a placeholder value for this run":
        return _bare_after_error(run=run, upstream_by_grid=upstream_by_grid)
    return next(
        (
            f'run {run.run_path} matches {len(runs)} runs of grid {grid_name!r}; '
            f'{{after:{grid_name}}} needs exactly one'
            for grid_name, runs in upstream_by_grid.items()
            if error == f"'after:{grid_name}' is not a placeholder value for this run"
        ),
        error,
    )


def _bare_after_error(
    *, run: StudyRun, upstream_by_grid: Mapping[str, tuple[StudyRun, ...]]
) -> str:
    """Why ``{after}`` has no value: several upstream grids or several matched runs."""
    if len(upstream_by_grid) != 1:
        return f'run {run.run_path}: {{after}} needs exactly one after grid; use {{after:<grid>}}'
    ((grid_name, runs),) = upstream_by_grid.items()
    return (
        f'run {run.run_path} matches {len(runs)} runs of grid {grid_name!r}; '
        '{after} needs exactly one'
    )


def plan_jobs(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    study: Study,
    machine: MachineProfile,
    resolved: Sequence[ResolvedRun],
    indices: Mapping[PurePosixPath, int],
    extra_args: Mapping[PurePosixPath, tuple[str, ...]],
    attempts: Mapping[PurePosixPath, int],
    first_job_file: int,
    environment: Mapping[str, str],
    project_root: Path | None = None,
    uncounted: Mapping[PurePosixPath, tuple[int, int]] | None = None,
    code_snapshots: Mapping[PurePosixPath, CodeSnapshot | None] | None = None,
    after: Mapping[PurePosixPath, tuple['AfterTarget', ...]] | None = None,
) -> list[JobGroup]:
    """The runs grouped into numbered jobs, ready for validation and launch.

    Runs group by (grid, job name, resources, code snapshot) in first-appearance order;
    a group is never split by count, so a group of tasks past
    ``max_array_size`` is reported by ``validate_jobs`` instead of planned
    around. The code snapshot is part of the key, so a resubmit of a mixed
    study gives each job file exactly one ``PYTHONPATH``.

    Args:
        study: The study the runs came from.
        machine: The machine the runs run on.
        resolved: The runs resolved on ``machine``, in run order.
        indices: The run path to array index, e.g. from ``assign_indices``.
        extra_args: The run path to pass-through arguments, last in the
            command.
        attempts: The run path to attempts already made; 0 when absent.
        first_job_file: The first job file number to number groups from.
        environment: The process environment used to expand run settings.
        project_root: The directory jobs run in, needed to make a relative
            ``output_root`` absolute for an ``{after}`` placeholder.
        uncounted: The run path to ``(uncounted attempts, uncounted in a row)``,
            see ``TaskRecord``; ``(0, 0)`` when absent.
        code_snapshots: The run path to code snapshot; ``None`` when absent,
            which means the run submits with the live code.
        after: The upstream targets for each run path.

    Returns:
        The groups, numbered from ``first_job_file`` in first-appearance
        order.
    """
    folder = study_folder(machine=machine, study=study)
    groups: list[JobGroup] = []
    lookup: dict[tuple[str, str, Resources, CodeSnapshot | None, tuple[AfterTarget, ...]], int] = {}
    next_file = first_job_file
    tasks_of: dict[int, list[PlannedTask]] = {}
    pin_conflicts_found: list[tuple[str, SettingValue, SettingValue]] = []
    env = expansion_env(machine=machine, environment=environment)
    counters = uncounted or {}
    snapshots = code_snapshots or {}
    targets = after or {}
    chained = chained_grids(study=study)
    for run in resolved:
        grid = study.grids[run.run.grid]
        code_snapshot = snapshots.get(run.run.run_path)
        run_after = targets.get(run.run.run_path, ())
        job_name = study.job_name_of(run=run.run)
        key = (run.run.grid, job_name, run.resources, code_snapshot, run_after)
        if key not in lookup:
            lookup[key] = len(groups)
            tasks_of[lookup[key]] = []
            groups.append(
                JobGroup(
                    grid=run.run.grid,
                    job_name=job_name,
                    resources=run.resources,
                    job_file=next_file,
                    tasks=(),
                    after=run_after,
                    chained=run.run.grid in chained,
                    code_snapshot=code_snapshot,
                )
            )
            next_file += 1
        index = indices[run.run.run_path]
        args = extra_args.get(run.run.run_path, ())
        log_file = log_file_of(folder=folder, job_name=job_name, index=index)
        placeholders = {
            'output_root': str(machine.output_root),
            'study': study.name,
            'run_path': run.run.run_path.as_posix(),
            'log_file': str(log_file),
            'cpus': str(run.resources.cpus),
            'cpus_per_run': str(run.resources.cpus_per_run),
        }
        after_placeholders, upstream_by_grid = _after_placeholders(
            study=study, machine=machine, grid=grid, run=run.run, project_root=project_root
        )
        placeholders.update(after_placeholders)
        try:
            run_pin_conflicts = pin_conflicts(
                before=run.run.settings, after=run.settings, env=env, placeholders=placeholders
            )
            arguments = run_arguments(
                grid=grid, settings=run.settings, env=env, placeholders=placeholders
            )
        except (ValueError, TypeError) as error:
            arguments = []
            error_text = _after_placeholder_error(
                error=str(error), run=run.run, upstream_by_grid=upstream_by_grid
            )
        else:
            pin_conflicts_found.extend(run_pin_conflicts)
            error_text = None
        max_tasks_per_job = run.resources.max_tasks_per_job
        if max_tasks_per_job is not None and index >= max_tasks_per_job:
            size_error = _task_error(
                grid_name=run.run.grid,
                run_path=run.run.run_path,
                index=index,
                max_array_size=max_tasks_per_job,
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
    unique_conflicts = [
        conflict
        for position, conflict in enumerate(pin_conflicts_found)
        if conflict not in pin_conflicts_found[:position]
    ]
    if unique_conflicts:
        message = '; '.join(
            f'study {study.name!r} pins {path} = {study_value!r}, but '
            f'{machine.path} sets {machine_value!r}; remove it from one of them'
            for path, study_value, machine_value in unique_conflicts
        )
        raise ValueError(message)
    planned = [
        replace(group, tasks=tuple(sorted(tasks_of[position], key=lambda task: task.index)))
        for position, group in enumerate(groups)
    ]
    return merge_corr_groups(groups=planned, indices=indices)


def validate_jobs(
    *, study: Study, groups: Sequence[JobGroup], parallel_from: int = PARALLEL_VALIDATION_FROM
) -> list[tuple[PurePosixPath, str]]:
    """The run paths and messages of task argument and group resource problems.

    Args:
        study: The study the groups came from.
        groups: The planned groups.
        parallel_from: The task count from which the arguments are checked in
            worker processes.

    Returns:
        One pair per task or group problem, in group and task order. A task
        whose arguments could not even be built reports its building error.
    """
    checks = [
        (study.grids[group.grid].config, task.command[len(study.grids[group.grid].command) :])
        for group in groups
        for task in group.tasks
        if task.error is None
    ]
    errors = iter(_argument_errors(checks=checks, parallel=len(checks) >= parallel_from))
    problems: list[tuple[PurePosixPath, str]] = []
    for group in groups:
        problems.extend(_task_problems(group=group, errors=errors))
        group_problem = _group_signal_problem(group=group)
        if group_problem is not None:
            problems.append(group_problem)
    return problems


def _task_problems(
    *, group: JobGroup, errors: Iterator[str | None]
) -> list[tuple[PurePosixPath, str]]:
    """The problems of the tasks of one group; tasks that were checked consume ``errors``."""
    problems = []
    for task in group.tasks:
        message = task.error if task.error is not None else _stripped(error=next(errors))
        if message is not None:
            problems.append((task.resolved.run.run_path, message))
    return problems


def _stripped(*, error: str | None) -> str | None:
    """The error without surrounding whitespace, or ``None``."""
    return None if error is None else error.strip()


def _argument_errors(
    *, checks: Sequence[tuple[type, tuple[str, ...]]], parallel: bool
) -> list[str | None]:
    """The ``validate_arguments`` result of each (config class, arguments) pair, in order."""
    if parallel:
        try:
            return _errors_in_workers(checks=checks)
        except (OSError, BrokenProcessPool, pickle.PicklingError, AttributeError) as error:
            # why: a config class the workers cannot import (a local class) or a
            # pool that cannot start still validates, one process slower
            _LOGGER.warning('checking the run arguments in one process: %s', error)
    return [_check(check) for check in checks]


def _errors_in_workers(*, checks: Sequence[tuple[type, tuple[str, ...]]]) -> list[str | None]:
    """The ``validate_arguments`` result of each pair, computed in worker processes."""
    workers = _worker_count()
    chunksize = max(1, len(checks) // (workers * _CHUNKS_PER_WORKER))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_check, checks, chunksize=chunksize))


def _worker_count() -> int:
    """The ``_CPU_SHARE`` of the CPUs this process may use, at least one."""
    # why: on Linux the affinity mask (cgroups, taskset) is what this process may use
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return max(1, int(cpus * _CPU_SHARE))


def _check(check: tuple[type, tuple[str, ...]]) -> str | None:
    """``validate_arguments`` on one (config class, arguments) pair; a worker entry point."""
    config_class, arguments = check
    return validate_arguments(config_class=config_class, arguments=arguments)


def _group_signal_problem(*, group: JobGroup) -> tuple[PurePosixPath, str] | None:
    """The time-limit signal problem for one job group, or ``None`` when valid."""
    problem = group.resources.signal_problem()
    if problem is None:
        return None
    run_path = group.tasks[0].resolved.run.run_path
    return run_path, f'job file {group.job_file}: {problem}'
