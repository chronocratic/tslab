"""Helpers shared by the ``tslab study`` command dataclasses."""

import builtins
from collections.abc import Generator, Sequence
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shlex

from tslab.lab.cli.common import Request, UsageError
from tslab.lab.enums import StudyState
from tslab.lab.study.code_snapshot import CodeSnapshot, CodeSnapshotError, take_code_snapshot
from tslab.lab.study.commands import parse_states
from tslab.lab.study.context import Context, load_context, records_context, Selection
from tslab.lab.study.launchers.cpus import available_cpus
from tslab.lab.study.launchers.tables import backend_for
from tslab.lab.study.machine import resolve_machine
from tslab.lab.study.project import find_studies, load_project_settings, ProjectSettings

_LOGGER = logging.getLogger('tslab.lab.cli.study')


def _context_of(  # noqa: PLR0913 # why: keyword-only; each option is one flag of the command
    *,
    study: str,
    machine: str,
    grid: tuple[str, ...],
    only: tuple[str, ...],
    read_only: bool = False,
    ended_states: dict[str, str] | None = None,
) -> Context:
    """The context of a study reference; a bad reference or profile is a usage error."""
    with _usage_errors(study=study):
        location = resolve_machine(machine=machine, start=Path.cwd(), environment=os.environ)
        return load_context(
            reference=study,
            machine_path=location.path,
            selection=Selection(grids=grid, only=only),
            project_root=location.project_root,
            read_only=read_only,
            ended_states=ended_states,
            local_cpus=available_cpus(),
            environment=os.environ,
        )


def _records_context_of(*, study: str, machine: str) -> Context:
    """The study, machine and job records only; a bad reference or profile is a usage error."""
    with _usage_errors(study=study):
        location = resolve_machine(machine=machine, start=Path.cwd(), environment=os.environ)
        return records_context(
            reference=study,
            machine_path=location.path,
            project_root=location.project_root,
            environment=os.environ,
        )


def take_requested_code_snapshot(
    *, request: Request, context: Context, wanted: bool, include_untracked: bool, dry_run: bool
) -> CodeSnapshot | None:
    """The code snapshot the launch runs from; ``None`` when not wanted.

    Args:
        request: The resolved request; its ``tslab.toml`` names the project root
            and the untracked check.
        context: The context of the launch, for the output root of the machine.
        wanted: Whether the command wants a code snapshot at all.
        include_untracked: Copy untracked files under ``src/`` too.
        dry_run: Run every check and compute the hash, copy nothing.

    Raises:
        UsageError: If the snapshot cannot be taken; the message names the way out.
    """
    if not wanted:
        return None
    settings = request.settings
    project_root = settings.path.parent if settings.path is not None else context.project_root
    try:
        taken = take_code_snapshot(
            project_root=project_root,
            output_root=context.machine.output_root,
            include_untracked=include_untracked,
            check_untracked=settings.code_snapshot_check_untracked,
            dry_run=dry_run,
        )
    except CodeSnapshotError as error:
        raise UsageError(str(error)) from error
    _LOGGER.info('code snapshot: %s', taken.folder)
    return taken


@contextmanager
def _usage_errors(*, study: str) -> Generator[None]:
    try:
        yield
    except ModuleNotFoundError as error:
        module_name = study.partition(':')[0]
        message = str(error)
        # why: only a missing study module is a usage error worth a hint; a missing
        # dependency inside it keeps its own message
        if error.name is not None and f'{module_name}.'.startswith(f'{error.name}.'):
            message = _missing_study_message(error=error, module_name=module_name)
        raise UsageError(message) from error
    except (ValueError, TypeError, FileNotFoundError) as error:
        message = _missing_study_attribute_message(study=study, error=error)
        raise UsageError(message) from error


def _missing_study_attribute_message(
    *, study: str, error: ValueError | TypeError | FileNotFoundError
) -> str:
    """Add available names when a short-name module lacks its ``study`` attribute."""
    if not isinstance(error, ValueError):
        return str(error)
    module_name, separator, attribute_name = study.partition(':')
    if not separator or attribute_name != 'study':
        return str(error)
    if str(error) != f'{module_name} has no attribute {attribute_name!r}':
        return str(error)
    return _study_lookup_error(module_name=module_name, problem=str(error), fallback=str(error))


def _missing_study_message(*, error: ModuleNotFoundError, module_name: str) -> str:
    """The useful usage error for a short name whose module cannot import."""
    fallback = (
        f'cannot import the study module {module_name!r} ({error}); run tslab from '
        'the project folder, or check STUDY (module:attribute)'
    )
    return _study_lookup_error(
        module_name=module_name, problem=f'cannot import {module_name}', fallback=fallback
    )


def _study_lookup_error(*, module_name: str, problem: str, fallback: str) -> str:
    """The short-name error with discovered studies, or its fallback."""
    settings: ProjectSettings = load_project_settings(start=Path.cwd())
    if settings.studies is None:
        return fallback
    prefix = f'{settings.studies}.'
    if not module_name.startswith(prefix):
        return fallback
    short_name = module_name.removeprefix(prefix)
    if not short_name or '.' in short_name:
        return fallback
    try:
        found = find_studies(settings=settings)
    except (ModuleNotFoundError, ValueError):
        return fallback
    names = ', '.join(found) if found else 'none'
    return f'no study {short_name!r}: {problem}; studies found: {names}'


def _states_of(*, values: tuple[str, ...]) -> tuple[StudyState, ...]:
    try:
        return parse_states(values=values)
    except ValueError as error:
        raise UsageError(str(error)) from error


def _selection_args(
    *, study: str, machine: str, grid: tuple[str, ...], only: tuple[str, ...]
) -> str:
    return shlex.join([study, '--machine', machine, *_selection_options(grid=grid, only=only)])


def _log_next_step(*, dry_run: bool, selection: str) -> None:
    """Log what to run next after a launch that exited cleanly."""
    if dry_run:
        _LOGGER.info('dry run: nothing submitted; run again without --dry-run to submit')
        return
    _LOGGER.info('follow progress: tslab study status %s', selection)


def _selection_options(*, grid: tuple[str, ...], only: tuple[str, ...]) -> builtins.list[str]:
    options: builtins.list[str] = []
    for grid_name in grid:
        options.extend(('--grid', grid_name))
    for selection in only:
        options.extend(('--only', selection))
    return options


def _detach(  # noqa: PLR0913 # why: keyword-only private helper; one keyword per part of the command line
    *,
    context: Context,
    command: str,
    selection_options: Sequence[str],
    options: Sequence[str],
    flags: Sequence[str],
    extra_args: Sequence[str],
    request: tuple[str, str],
) -> int:
    """Start the command again in the background; the arguments are the original ones."""
    study, machine = request
    arguments = [
        'study',
        command,
        study,
        '--machine',
        machine,
        *selection_options,
        *options,
        *flags,
    ]
    if extra_args:
        arguments.extend(('--', *extra_args))
    backend = backend_for(machine=context.machine)
    try:
        return backend.detach(
            context=context,
            arguments=arguments,
            wait_command=shlex.join(
                ['tslab', 'study', 'wait', study, '--machine', machine, *selection_options]
            ),
        )
    except ValueError as error:
        raise UsageError(str(error)) from error


def _warn_no_runs(*, context: Context) -> None:
    if not context.reports:
        _LOGGER.warning('no runs match the selection; drop --grid/--only or check their values')


def _status_hints(
    *, context: Context, study: str, machine: str, grid: tuple[str, ...], only: tuple[str, ...]
) -> None:
    if not context.reports:
        _warn_no_runs(context=context)
        return
    selection = _selection_args(study=study, machine=machine, grid=grid, only=only)
    states = {report.state for report in context.reports}
    if states == {StudyState.MISSING}:
        _LOGGER.info('nothing ran yet; check the plan: tslab study submit %s --dry-run', selection)
        return
    if StudyState.FAILED in states:
        _LOGGER.info(
            'fix the failed runs, then: tslab study resubmit %s --states failed', selection
        )
    if StudyState.INTERRUPTED in states or StudyState.MISSING in states:
        _LOGGER.info('retry interrupted and missing runs: tslab study resubmit %s', selection)
    if StudyState.QUEUED in states or StudyState.RUNNING in states:
        _LOGGER.info('follow progress: tslab study status %s --watch 60', selection)
    if all(report.state is StudyState.DONE for report in context.reports):
        _LOGGER.info(
            'all %d runs done; collect metrics: tslab study results %s --out results.csv',
            len(context.reports),
            selection,
        )
