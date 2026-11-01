"""The resubmit and run-one commands: retry interrupted runs, run one in the foreground."""

from collections.abc import Callable, Sequence
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import TextIO, TYPE_CHECKING

from tslab.core.paths import LOCK_FILE
from tslab.study.commands import _finish_launch, _uncounted, _UNCOUNTED_JOB_STATES, LIVE_STATES
from tslab.study.commands.submit import submit_runs
from tslab.study.context import Context
from tslab.study.launchers import backend_for, JobPhase, OnLaunch
from tslab.study.plan import assign_indices, plan_jobs, validate_jobs
from tslab.study.records import latest_tasks, take_progress
from tslab.study.reporting.report import log_problems
from tslab.study.reporting.state import RunReport, StudyState

if TYPE_CHECKING:
    from tslab.study.launchers import Backend

__all__ = ['resubmit', 'run_one']

# why: a selection error lists this many run paths before counting the rest
_SELECTION_SHOWN = 20

# why: a cancelled job leaves its lock until the heartbeat is stale; the restart
# waits this many seconds (the stale-lock limit of the run engine plus a minute)
_RESTART_DELAY_SECONDS = 660

#: Preempted or node-failed attempts in a row after which the run is left alone.
_MAX_UNCOUNTED_IN_ROW = 20

_LOGGER = logging.getLogger(__name__)


def resubmit(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    states: tuple[StudyState, ...] = (StudyState.INTERRUPTED, StudyState.MISSING),
    cancel_live: bool = False,
    dry_run: bool = False,
    extra_args: tuple[str, ...] | None = None,
    auto: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    on_launch: OnLaunch | None = None,
    task_stream: TextIO | None = None,
) -> int:
    """Submit the runs in the given states again, one attempt further on.

    Queued and running runs are never selected by state. With
    ``cancel_live``, the queued and running runs that hold a SLURM job id
    are cancelled first and submitted a little later, so they restart on a
    fresh machine. The cancelling happens after the plan is validated, and
    never on a dry run.

    Args:
        context: The context to resubmit from.
        states: The states to resubmit; interrupted and missing by default.
        cancel_live: Cancel the live runs of the selection first.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The pass-through arguments; taken from the last
            submission when ``None``.
        auto: The mode of the follow-up job: only interrupted runs that made
            progress since their last attempt, up to
            ``followup.max_attempts`` of the machine; the rest are logged and left alone,
            and ``cancel_live`` is ignored.
        sleep: What waits out the submission pause of the cluster.
        on_launch: Called once with the launch result and groups, including dry runs.
        task_stream: Stream for task output on a local machine.

    Returns:
        ``0`` when everything submitted, or nothing needed submitting;
        ``1`` when a run does not parse or a submission failed;
        ``128 + N`` when the local launch was stopped by signal N.
    """
    if auto:
        # why: auto is the path of the follow-up job: it retries interrupted runs
        # with progress only, and never touches the live ones
        states = (StudyState.INTERRUPTED,)
        cancel_live = False
    if not states:
        states = (StudyState.INTERRUPTED, StudyState.MISSING)
    selected = [
        report
        for report in context.reports
        if report.state in states and report.state not in LIVE_STATES
    ]
    if auto:
        selected = _with_progress(reports=selected, context=context)
    live = _live_reports(reports=context.reports, cancel_live=cancel_live)
    selected += live
    if not selected:
        _LOGGER.info('nothing to resubmit')
        return 0
    reuse = latest_tasks(jobs=context.jobs)
    args: dict[str, tuple[str, ...]] = {}
    for report in selected:
        path = report.run.run_path
        if extra_args is None and (task := reuse.get(path)) is not None:
            args[path] = task[1].extra_args
        else:
            args[path] = extra_args or ()
    begin, before_launch = _before_resubmit(
        backend=backend_for(machine=context.machine),
        context=context,
        live=live,
        selected=selected,
        dry_run=dry_run,
    )

    code, submitted = submit_runs(
        context=context,
        reports=selected,
        dry_run=dry_run,
        extra_args=args,
        sleep=sleep,
        begin=begin,
        before_launch=before_launch,
        on_launch=on_launch,
        task_stream=task_stream,
    )
    return _finish_launch(
        code=code, submitted=submitted, context=context, sleep=sleep, task_stream=task_stream
    )


def _live_reports(*, reports: Sequence[RunReport], cancel_live: bool) -> list[RunReport]:
    if not cancel_live:
        return []
    return [
        report for report in reports if report.state in LIVE_STATES and report.job_id is not None
    ]


def _before_resubmit(
    *,
    backend: 'Backend',
    context: Context,
    live: Sequence[RunReport],
    selected: Sequence[RunReport],
    dry_run: bool,
) -> tuple[str | None, Callable[[], int]]:
    cancellable = backend.cancellable_job_ids(reports=live)
    if dry_run and cancellable:
        _LOGGER.info('would cancel %s', ', '.join(cancellable))

    def before_launch() -> int:
        if cancellable:
            code, _ = backend.cancel(context=context, dry_run=False, signal=None)
            if code != 0:
                return code
        _remove_ended_locks(reports=selected, study_directory=context.study_directory)
        return 0

    begin = f'now+{_RESTART_DELAY_SECONDS}' if cancellable else None
    return begin, before_launch


def _run_in_terminal(*, args: list[str]) -> int:
    """The exit code of the command with the inherited standard streams."""
    # why: inherited streams, so breakpoint() and prompts reach the terminal
    return subprocess.run(args, check=False).returncode  # noqa: S603 # why: the run command, typed by the study


def _run_process(args: list[str]) -> int:
    # why: the run_process of run_one is called with one positional argument
    return _run_in_terminal(args=args)


def run_one(
    *,
    context: Context,
    extra_args: tuple[str, ...] | None,
    run_process: Callable[[list[str]], int] = _run_process,
) -> int:
    """Run the one selected run in the foreground of this terminal.

    SLURM is never called, and nothing is recorded: the study folder is left
    untouched, so the run of the foreground does not become an attempt of the
    study. Inside a ``salloc`` allocation, ``job_time_share`` still pauses the
    training; pass ``-- --trainer.job-time-share 1`` to train for the whole
    allocation.

    Args:
        context: The context with the runs selected; exactly one is needed.
        extra_args: The pass-through arguments; taken from the latest
            submission of the run when ``None``.
        run_process: What runs the script; by default a subprocess with the
            inherited standard streams, so ``breakpoint()`` reaches the
            terminal.

    Returns:
        The exit code of the command; ``1`` when the run is live or the plan
        has problems, without starting anything.

    Raises:
        ValueError: When zero or more than one run is selected.
    """
    reports = list(context.reports)
    if not reports:
        message = 'no runs match the selection; drop --grid/--only or check values'
        raise ValueError(message)
    if len(reports) > 1:
        paths = [report.run.run_path for report in reports]
        shown = ', '.join(paths[:_SELECTION_SHOWN])
        more = (
            f', and {len(paths) - _SELECTION_SHOWN} more' if len(paths) > _SELECTION_SHOWN else ''
        )
        message = f'{len(paths)} runs match the selection: {shown}{more}; narrow with --only'
        raise ValueError(message)
    (report,) = reports
    if report.state in LIVE_STATES:
        message = (
            f'{report.run.run_path} is {report.state.value} (job {report.job_id}); '
            'wait for it or cancel it first'
        )
        _LOGGER.error('%s', message)
        return 1
    run = report.run
    resolved = context.resolved[run.run_path]
    if extra_args is not None:
        args = extra_args
    else:
        args = _reused_args(path=run.run_path, context=context)
    groups = plan_jobs(
        study=context.study,
        machine=context.machine,
        resolved=[resolved],
        indices=assign_indices(runs=[run], jobs=context.jobs),
        extra_args={run.run_path: args},
        attempts={run.run_path: report.attempt},
        first_job_file=0,
        environment=os.environ,
    )
    problems = validate_jobs(study=context.study, groups=groups)
    if problems:
        log_problems(problems=problems)
        return 1
    (task,) = groups[0].tasks
    script = backend_for(machine=context.machine).foreground_script(
        context=context, resources=resolved.resources, command=task.command
    )
    return run_process(['bash', '-c', script])


def _reused_args(*, path: str, context: Context) -> tuple[str, ...]:
    """The extra args of the latest submission of one run, or none."""
    task = latest_tasks(jobs=context.jobs).get(path)
    return () if task is None else task[1].extra_args


def _with_progress(*, reports: Sequence[RunReport], context: Context) -> list[RunReport]:
    """The reports that may retry: under the cap, and one step further on.

    An attempt that ended ``PREEMPTED`` or ``NODE_FAIL`` does not count toward the
    cap and needs no progress, until 20 of them follow one another.
    """
    max_attempts = context.machine.followup.max_attempts
    retried: list[RunReport] = []
    for report in reports:
        if report.job_state in _UNCOUNTED_JOB_STATES:
            if _uncounted(report=report)[1] > _MAX_UNCOUNTED_IN_ROW:
                _LOGGER.info(
                    'gave up: %s preempted or node failed %d times in a row',
                    report.run.run_path,
                    _MAX_UNCOUNTED_IN_ROW,
                )
                continue
            retried.append(report)
            continue
        counted = report.attempt - (report.task.uncounted_attempts if report.task else 0)
        if counted >= max_attempts:
            _LOGGER.info('gave up: %s after %d attempts', report.run.run_path, counted)
            continue
        if not _made_progress(report=report, context=context):
            _LOGGER.info('stuck: %s (attempt %d, no progress)', report.run.run_path, report.attempt)
            continue
        retried.append(report)
    return retried


def _made_progress(*, report: RunReport, context: Context) -> bool:
    """Whether the run moved since its latest submission.

    More done instances than the snapshot of the latest task, or a newer
    checkpoint than the one of the snapshot (a snapshot without one counts as old).
    """
    progress = take_progress(run_directory=context.study_directory / report.run.run_path)
    before = report.task.progress if report.task is not None else None
    if before is None:
        return progress.done_instances > 0 or progress.newest_checkpoint_mtime is not None
    if progress.done_instances > before.done_instances:
        return True
    return progress.newest_checkpoint_mtime is not None and (
        before.newest_checkpoint_mtime is None
        or progress.newest_checkpoint_mtime > before.newest_checkpoint_mtime
    )


def _remove_ended_locks(*, reports: Sequence[RunReport], study_directory: Path) -> None:
    for report in reports:
        job_state = report.job_state
        if report.job_phase in (JobPhase.QUEUED, JobPhase.RUNNING, JobPhase.UNKNOWN):
            continue
        if not report.folder.lock_live:
            continue
        (study_directory / report.run.run_path / LOCK_FILE).unlink(missing_ok=True)
        _LOGGER.info(
            'removed the lock of %s: its job %s ended (%s)',
            report.run.run_path,
            report.job_id or '-',
            job_state,
        )
