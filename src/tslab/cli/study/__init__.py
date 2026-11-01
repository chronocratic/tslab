"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses in the submodules hold only the flags of one command;
the behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shlex

from tslab.cli.usage import UsageError
from tslab.study.commands import parse_states
from tslab.study.context import Context, load_context, Selection
from tslab.study.launchers import available_cpus, backend_for
from tslab.study.machine import resolve_machine
from tslab.study.state import StudyState

__all__ = [
    'STUDY_COMMANDS',
    'Cancel',
    'Failures',
    'Resubmit',
    'Results',
    'RunOne',
    'Show',
    'Status',
    'Submit',
    'Usage',
    'Wait',
]

_LOGGER = logging.getLogger(__name__)


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
            message = (
                f'cannot import the study module {module_name!r} ({error}); run tslab from '
                'the project folder, or check STUDY (module:attribute)'
            )
        raise UsageError(message) from error
    except (ValueError, TypeError, FileNotFoundError) as error:
        raise UsageError(str(error)) from error


def _states_of(*, values: tuple[str, ...]) -> tuple[StudyState, ...]:
    try:
        return parse_states(values=values)
    except ValueError as error:
        raise UsageError(str(error)) from error


def _selection_args(
    *, study: str, machine: str, grid: tuple[str, ...], only: tuple[str, ...]
) -> str:
    arguments = [study, '--machine', machine]
    for grid_name in grid:
        arguments.extend(('--grid', grid_name))
    for selection in only:
        arguments.extend(('--only', selection))
    return shlex.join(arguments)


def _selection_options(*, grid: tuple[str, ...], only: tuple[str, ...]) -> list[str]:
    options: list[str] = []
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


# why: the submodules import the shared helpers from this door
from tslab.cli.study.cancel import Cancel  # noqa: E402
from tslab.cli.study.failures import Failures, Usage  # noqa: E402
from tslab.cli.study.resubmit import Resubmit, RunOne  # noqa: E402
from tslab.cli.study.results import Results  # noqa: E402
from tslab.cli.study.status import Show, Status, Wait  # noqa: E402
from tslab.cli.study.submit import Submit  # noqa: E402

STUDY_COMMANDS = {
    'submit': Submit,
    'status': Status,
    'resubmit': Resubmit,
    'results': Results,
    'show': Show,
    'wait': Wait,
    'failures': Failures,
    'cancel': Cancel,
    'usage': Usage,
    'run': RunOne,
}
