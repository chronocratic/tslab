"""The ``tslab study`` subcommands: thin tyro dataclasses over ``tslab.study.commands``.

The argument dataclasses here hold only the flags of one command; the
behavior lives in ``tslab.study.commands``, which the ``run`` methods call.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import shlex
import sys
from typing import Annotated

import tyro

from tslab.cli.console import use_stderr, watch
from tslab.cli.output import print_json
from tslab.cli.usage import UsageError
from tslab.jobs.slurm import SlurmError
from tslab.study.commands import cancel, parse_states, resubmit, results, run_one, status, submit
from tslab.study.context import Context, expected_starts, load_context, Selection
from tslab.study.documents import cancel_document, launch_document, show_document, status_document
from tslab.study.failures import collect_failures, log_failures
from tslab.study.launchers import backend_for, LaunchReport
from tslab.study.loading import load_study
from tslab.study.machine import resolve_machine
from tslab.study.outcome import EXIT_TIMEOUT, study_exit_code
from tslab.study.plan import JobGroup
from tslab.study.report import log_show
from tslab.study.state import StudyState
from tslab.study.usage import log_usage, usage_document
from tslab.study.wait import wait_for_study

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
_MIN_WAIT_INTERVAL_SECONDS = 10


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
        location = resolve_machine(machine=machine, start=Path.cwd())
        return load_context(
            reference=study,
            machine_path=location.path,
            selection=Selection(grids=grid, only=only),
            project_root=location.project_root,
            read_only=read_only,
            ended_states=ended_states,
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


@dataclass(frozen=True, kw_only=True)
class Submit:
    """Submit the runs of a study to a machine, or dry-run the plan.

    For example:
        tslab study submit my_paper.studies:example --machine local --dry-run
        tslab study submit my_paper.studies:example --machine local -- --trainer.max-epochs 5

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        include_done: Submit the done runs as well.
        dry_run: Write the job files and print the plan, submit nothing.
        detach: Local machines only: start the launch in the background and return once
            it runs; follow it with tslab study wait.
        extra_args: The arguments after --, appended to every run.
        json: Print one study-launch JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    include_done: bool = False
    dry_run: bool = False
    detach: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()
    json: bool = False

    def run(self) -> int:
        """Submit and return the exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        if self.detach:
            flags = [
                flag
                for flag, on in (
                    ('--include-done', self.include_done),
                    ('--dry-run', self.dry_run),
                    ('--json', self.json),
                )
                if on
            ]
            return _detach(
                context=context,
                command='submit',
                selection_options=_selection_options(grid=self.grid, only=self.only),
                options=[],
                flags=flags,
                extra_args=self.extra_args,
                request=(self.study, self.machine),
            )
        _warn_no_runs(context=context)

        launches: list[tuple[LaunchReport, Sequence[JobGroup]]] = []

        def remember_launch(*, report: LaunchReport, groups: Sequence[JobGroup]) -> None:
            launches.append((report, groups))

        code = submit(
            context=context,
            include_done=self.include_done,
            dry_run=self.dry_run,
            extra_args=self.extra_args,
            on_launch=remember_launch if self.json else None,
            task_stream=sys.stderr if self.json else None,
        )
        if code == 0:
            if self.dry_run:
                _LOGGER.info('dry run: nothing submitted; run again without --dry-run to submit')
            else:
                selection = _selection_args(
                    study=self.study, machine=self.machine, grid=self.grid, only=self.only
                )
                _LOGGER.info('follow progress: tslab study status %s', selection)
        if self.json and (code == 0 or launches):
            report, groups = launches[0] if launches else (None, ())
            print_json(
                document=launch_document(
                    report=report, groups=groups, dry_run=self.dry_run, cancelled=(), exit_code=code
                )
            )
        return code


@dataclass(frozen=True, kw_only=True)
class Status:
    """Show the state of every run of a study, and the runs not done.

    For example:
        tslab study status my_paper.studies:example --machine local
        tslab study status my_paper.studies:example --machine local --watch 60

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        states: Show only the runs in these states; comma separated.
        watch: Refresh every this many seconds until Ctrl-C.
        json: Print one study-status JSON document.
        exit_code: Return the code for the run states.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    states: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='STATE'), tyro.conf.UseAppendAction
    ] = ()
    watch: Annotated[float | None, tyro.conf.arg(metavar='SECONDS', help_behavior_hint='')] = None
    json: bool = False
    exit_code: bool = False

    def run(self) -> int:
        """Print the table and return the exit code."""
        if self.watch is not None and self.json:
            message = '--watch does not combine with --json; use tslab study wait'
            raise UsageError(message)
        states = _states_of(values=self.states)

        def show() -> int:
            context = _context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
            )
            if self.exit_code and not context.reports:
                message = 'no runs match the selection; drop --grid/--only or check values'
                raise UsageError(message)
            starts = expected_starts(context=context)
            status(context=context, states=states, starts=starts)
            _status_hints(
                context=context,
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
            )
            code = study_exit_code(context=context)
            if self.json:
                print_json(document=status_document(context=context, exit_code=code, starts=starts))
            return code if self.exit_code else 0

        return watch(seconds=self.watch, show=show)


@dataclass(frozen=True, kw_only=True)
class Show:
    """Show the grids, axes, and run counts of a study.

    For example:
        tslab study show my_paper.studies:example

    Attributes:
        study: The study reference, module:attribute.
        json: Print one study-show JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    json: bool = False

    def run(self) -> int:
        """Show a study without loading a machine profile."""
        with _usage_errors(study=self.study):
            study = load_study(reference=self.study)
        if self.json:
            print_json(document=show_document(study=study))
        else:
            log_show(study=study)
        return 0


@dataclass(frozen=True, kw_only=True)
class Wait:
    """Wait for selected runs to finish or need attention.

    For example:
        tslab study wait my_paper.studies:example --machine local

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        interval: Longest gap between checks in seconds (at least 10); checks start
            30 seconds apart and slow down while nothing changes.
        timeout: Stop after this many seconds.
        keep_going: Wait until no run is live, even after a failure.
        json: Print one study-wait JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    interval: Annotated[float, tyro.conf.arg(metavar='SECONDS')] = 300.0
    timeout: Annotated[float | None, tyro.conf.arg(metavar='SECONDS')] = None
    keep_going: bool = False
    json: bool = False

    def run(self) -> int:
        """Wait for the study and return its final state code."""
        if self.interval < _MIN_WAIT_INTERVAL_SECONDS:
            message = '--interval takes at least 10 seconds'
            raise UsageError(message)

        ended_states: dict[str, str] = {}

        def load() -> Context:
            context = _context_of(
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
                read_only=True,
                ended_states=ended_states,
            )
            if not context.reports:
                message = 'no runs match the selection; drop --grid/--only or check values'
                raise UsageError(message)
            return context

        code, context, waited = wait_for_study(
            load=load, interval=self.interval, timeout=self.timeout, keep_going=self.keep_going
        )
        if self.json:
            document = status_document(context=context, exit_code=code)
            document['schema'] = 'tslab.study-wait/1'
            document['waited_seconds'] = waited
            document['timed_out'] = code == EXIT_TIMEOUT
            print_json(document=document)
        else:
            status(context=context)
            _status_hints(
                context=context,
                study=self.study,
                machine=self.machine,
                grid=self.grid,
                only=self.only,
            )
        return code


@dataclass(frozen=True, kw_only=True)
class Failures:
    """Show error details and recent logs for failed or interrupted runs.

    For example:
        tslab study failures my_paper.studies:example --machine local

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        states: Show only runs in these states; failed and interrupted by default.
        lines: Number of trailing lines to show from each log.
        json: Print one study-failures JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    states: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='STATE'), tyro.conf.UseAppendAction
    ] = ()
    lines: int = 30
    json: bool = False

    def run(self) -> int:
        """Show failure details and return the process exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        failures = collect_failures(
            context=context, states=_states_of(values=self.states), lines=self.lines
        )
        if self.json:
            print_json(document={'schema': 'tslab.study-failures/1', 'runs': failures})
        elif failures:
            log_failures(failures=failures)
        else:
            _LOGGER.info('no failed or interrupted runs')
        return 0


@dataclass(frozen=True, kw_only=True)
class Cancel:
    """Cancel queued and running SLURM jobs, or stop the live local launch.

    For example:
        tslab study cancel my_paper.studies:example --machine cluster

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        dry_run: Show job ids without cancelling them.
        signal: Send USR2 to the running jobs instead of cancelling them, to test
            that runs save and resubmit; the jobs then end on their own.
        json: Print one study-cancel JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    dry_run: bool = False
    signal: bool = False
    json: bool = False

    def run(self) -> int:
        """Cancel selected jobs and return the process exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        signal = 'USR2' if self.signal else None
        code, cancelled = cancel(context=context, dry_run=self.dry_run, signal=signal)
        if code == 0 and cancelled and not self.dry_run:
            selection = _selection_args(
                study=self.study, machine=self.machine, grid=self.grid, only=self.only
            )
            _LOGGER.info('follow progress: tslab study status %s', selection)
        if self.json:
            print_json(
                document=cancel_document(
                    cancelled=cancelled, dry_run=self.dry_run, exit_code=code, signal=signal
                )
            )
        return code


@dataclass(frozen=True, kw_only=True)
class Usage:
    """Show what each run used, and the sizes to suggest for the next jobs.

    For example:
        tslab study usage my_paper.studies:example --machine cluster

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        json: Print one study-usage JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    json: bool = False

    def run(self) -> int:
        """Show the usage of the selected runs and return the process exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        try:
            document = usage_document(context=context)
        except SlurmError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: scheduler errors are shown, not raised
            return 1
        if self.json:
            print_json(document=document)
        else:
            log_usage(
                document=document,
                suggests_memory=backend_for(machine=context.machine).suggests_memory,
            )
        return 0


@dataclass(frozen=True, kw_only=True)
class RunOne:
    """Run one selected run in the foreground of this terminal.

    The command of the run is built the same way as for a submission, but
    nothing is submitted or recorded: SLURM is never called and the study
    folder is untouched. Inside a ``salloc`` allocation, ``job_time_share``
    still pauses the training; pass ``-- --trainer.job-time-share 1`` to
    train for the whole allocation.

    For example:
        tslab study run my_paper.studies:example --machine cluster --only model=small,seed=1

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name or profile path.
        grid: Keep only this grid.
        only: Keep only runs that match axis=value.
        extra_args: The arguments after --; taken from the last
            submission when omitted.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()

    def run(self) -> int:
        """Run the one selected run and return its exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        try:
            return run_one(context=context, extra_args=self.extra_args or None)
        except ValueError as error:
            raise UsageError(str(error)) from error


@dataclass(frozen=True, kw_only=True)
class Resubmit:
    """Submit the interrupted, missing or failed runs of a study again.

    For example:
        tslab study resubmit my_paper.studies:example --machine local
        tslab study resubmit my_paper.studies:example --machine local --dry-run

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        states: The states to resubmit; comma separated. Interrupted and
            missing when omitted.
        cancel_live: Cancel the queued and running runs first and
            resubmit them a little later.
        dry_run: Write the job files and print the plan, submit nothing.
        auto: Retry only the interrupted runs that made progress, up to
            followup.max_attempts each; ignores --states and --cancel-live.
            The follow-up jobs pass this.
        detach: Local machines only: start the launch in the background and return once
            it runs; follow it with tslab study wait.
        extra_args: The arguments after --; taken from the last
            submission when omitted.
        json: Print one study-launch JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    states: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='STATE'), tyro.conf.UseAppendAction
    ] = ()
    cancel_live: bool = False
    dry_run: bool = False
    auto: bool = False
    detach: bool = False
    extra_args: tyro.conf.Suppress[tuple[str, ...]] = ()
    json: bool = False

    def run(self) -> int:
        """Resubmit and return the exit code."""
        states = _states_of(values=self.states)
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only
        )
        if self.detach:
            flags = [
                flag
                for flag, on in (
                    ('--cancel-live', self.cancel_live),
                    ('--dry-run', self.dry_run),
                    ('--auto', self.auto),
                    ('--json', self.json),
                )
                if on
            ]
            options: list[str] = []
            for state in self.states:
                options.extend(('--states', state))
            return _detach(
                context=context,
                command='resubmit',
                selection_options=_selection_options(grid=self.grid, only=self.only),
                options=options,
                flags=flags,
                extra_args=self.extra_args,
                request=(self.study, self.machine),
            )
        _warn_no_runs(context=context)
        launches: list[tuple[LaunchReport, Sequence[JobGroup]]] = []

        def remember_launch(*, report: LaunchReport, groups: Sequence[JobGroup]) -> None:
            launches.append((report, groups))

        backend = backend_for(machine=context.machine)
        cancelled = (
            backend.cancellable_job_ids(reports=context.reports)
            if self.cancel_live and not self.auto
            else []
        )
        code = resubmit(
            context=context,
            states=states,
            cancel_live=self.cancel_live,
            dry_run=self.dry_run,
            extra_args=self.extra_args or None,
            auto=self.auto,
            on_launch=remember_launch if self.json else None,
            task_stream=sys.stderr if self.json else None,
        )
        if code == 0:
            if self.dry_run:
                _LOGGER.info('dry run: nothing submitted; run again without --dry-run to submit')
            else:
                selection = _selection_args(
                    study=self.study, machine=self.machine, grid=self.grid, only=self.only
                )
                _LOGGER.info('follow progress: tslab study status %s', selection)
        if self.json and (code == 0 or launches):
            report, groups = launches[0] if launches else (None, ())
            print_json(
                document=launch_document(
                    report=report,
                    groups=groups,
                    dry_run=self.dry_run,
                    cancelled=cancelled,
                    exit_code=code,
                )
            )
        return code


@dataclass(frozen=True, kw_only=True)
class Results:
    """Collect the metrics of every run of a study into one CSV file.

    For example:
        tslab study results my_paper.studies:example --machine local --out results.csv

    Attributes:
        study: The study reference, module:attribute.
        machine: A machine name (machines/<name>.toml above the working folder, or
            registered with tslab machine add) or a profile path.
        grid: Keep only this grid.
        only: Keep only the runs that match axis=value; comma separated for
            several values of one axis.
        out: The CSV file to write.
        json: Write rows as a versioned JSON document.
    """

    study: Annotated[str, tyro.conf.arg(metavar='STUDY'), tyro.conf.Positional]
    machine: Annotated[str, tyro.conf.arg(metavar='MACHINE')]
    grid: Annotated[tuple[str, ...], tyro.conf.arg(metavar='GRID'), tyro.conf.UseAppendAction] = ()
    only: Annotated[
        tuple[str, ...], tyro.conf.arg(metavar='AXIS=VALUES'), tyro.conf.UseAppendAction
    ] = ()
    out: Annotated[Path, tyro.conf.arg(metavar='FILE')]
    json: bool = False

    def run(self) -> int:
        """Write the table and return the exit code."""
        context = _context_of(
            study=self.study, machine=self.machine, grid=self.grid, only=self.only, read_only=True
        )
        _warn_no_runs(context=context)
        if self.out == Path('-'):
            use_stderr()
        return results(context=context, out=self.out, as_json=self.json)


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
