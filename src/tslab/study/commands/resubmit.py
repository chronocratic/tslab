"""The resubmit and run-one commands: retry interrupted runs, run one in the foreground."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import TextIO, TYPE_CHECKING

from tslab.core.lock import read_lock_owner
from tslab.core.meta import hostname
from tslab.core.paths import LOCK_FILE
from tslab.enums import AfterVerdict, ChainedRetry, JobFlag, JobPhase, Retry, StudyState
from tslab.study.chain import after_verdict, downstream_of
from tslab.study.code_snapshot import CodeSnapshot, CodeSnapshotError, SRC_FOLDER
from tslab.study.commands.submit import submit_runs
from tslab.study.commands.support import _finish_launch, _uncounted, LIVE_STATES
from tslab.study.context import Context, study_reports, widen_context
from tslab.study.definition import chained_grids, StudyRun, upstream_runs
from tslab.study.launchers.common.types import OnLaunch
from tslab.study.launchers.tables import backend_for
from tslab.study.machine.profile import FollowupSettings
from tslab.study.plan import assign_indices, plan_jobs, validate_jobs
from tslab.study.records import latest_tasks, take_progress
from tslab.study.reporting.report import log_problems
from tslab.study.reporting.state import RunReport

if TYPE_CHECKING:
    from tslab.study.launchers.common.protocols import Backend
    from tslab.study.records import JobRecord, TaskRecord

__all__ = ['may_retry', 'resubmit', 'run_one']

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
    states: tuple[StudyState, ...] = (
        StudyState.INTERRUPTED,
        StudyState.MISSING,
        StudyState.DEPENDENCY_FAILED,
    ),
    cancel_live: bool = False,
    dry_run: bool = False,
    extra_args: tuple[str, ...] | None = None,
    auto: bool = False,
    new_code_snapshot: CodeSnapshot | None = None,
    drop_code_snapshot: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    on_launch: OnLaunch | None = None,
    task_stream: TextIO | None = None,
    ignore_after: bool = False,
) -> int:
    """Submit the runs in the given states again, one attempt further on.

    Queued and running runs are never selected by state. With
    ``cancel_live``, the queued and running runs that hold a SLURM job id
    are cancelled first and submitted a little later, so they restart on a
    fresh machine. The cancelling happens after the plan is validated, and
    never on a dry run.

    Every run resubmits from the code snapshot of its latest job; a run
    without one runs the live code. ``new_code_snapshot`` switches every
    selected run to one fresh snapshot, and ``drop_code_snapshot`` to the
    live code. A reused snapshot folder without a ``src/`` copy is an error
    before anything is planned.

    Args:
        context: The context to resubmit from.
        states: The states to resubmit; interrupted, missing and dependency failed
            by default. Waiting runs that depend on a selected run come along, and
            their jobs are cancelled first.
        cancel_live: Cancel the live runs of the selection first.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The pass-through arguments; taken from the last
            submission when ``None``.
        auto: The mode of the follow-up job: only interrupted runs that made
            progress since their last attempt, up to
            ``followup.max_attempts`` of the machine; the rest are logged and left alone,
            and ``cancel_live`` is ignored. Runs of grids that are not chained need
            ``retry = "resubmit"``, runs of chained grids ``chained_retry`` other than
            ``"manual"``; dependency failed runs of chained grids are taken when their
            upstream runs are met or resubmitted in the same call.
        new_code_snapshot: The snapshot every selected run switches to,
            replacing the recorded one; ``None`` keeps the recorded one.
        drop_code_snapshot: Run the live code, even when the runs recorded a
            code snapshot.
        sleep: What waits out the submission pause of the cluster.
        on_launch: Called once with the launch result and groups, including dry runs.
        task_stream: Stream for task output on a local machine.
        ignore_after: Submit without dependency targets or the local dependency filter.

    Returns:
        ``0`` when everything submitted, or nothing needed submitting;
        ``1`` when a run does not parse or a submission failed;
        ``128 + N`` when the local launch was stopped by signal N.

    Raises:
        CodeSnapshotError: When a reused snapshot folder has no ``src/``
            left; the message names the folder and the way out.
    """
    if not states:
        states = (StudyState.INTERRUPTED, StudyState.MISSING, StudyState.DEPENDENCY_FAILED)
    if auto:
        # why: auto is the path of the follow-up job: it retries interrupted runs
        # with progress only, and never touches the live ones
        states = (StudyState.INTERRUPTED, StudyState.DEPENDENCY_FAILED)
        cancel_live = False
    selected = [
        report
        for report in context.reports
        if report.state in states and report.state not in LIVE_STATES
    ]
    if auto:
        selected = _auto_selection(reports=selected, context=context)
    else:
        context, selected = _with_cancelled_upstream(context=context, selected=selected)
    live = _live_reports(reports=context.reports, cancel_live=cancel_live)
    selected += live
    # why: SLURM holds a job at MaxBatchRequeue; it shows interrupted but is still queued
    live += [
        report
        for report in selected
        if JobFlag.REQUEUE_HELD in report.job_flags and report.job_id is not None
    ]
    context, selected, waiting = _with_downstream(context=context, selected=selected)
    live += waiting
    selected += waiting
    if not selected:
        _LOGGER.info('nothing to resubmit')
        return 0
    reuse = latest_tasks(jobs=context.jobs)
    code_snapshots = _code_snapshots_of(
        reports=selected, reuse=reuse, new=new_code_snapshot, drop=drop_code_snapshot, states=states
    )
    args: dict[PurePosixPath, tuple[str, ...]] = {}
    for report in selected:
        path = report.run.run_path
        if extra_args is None and (task := reuse.get(path)) is not None:
            args[path] = task[1].extra_args
        else:
            args[path] = extra_args or ()
    start_delay_seconds, before_launch = _before_resubmit(
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
        code_snapshots=code_snapshots,
        sleep=sleep,
        start_delay_seconds=start_delay_seconds,
        before_launch=before_launch,
        on_launch=on_launch,
        task_stream=task_stream,
        ignore_after=ignore_after,
    )
    return _finish_launch(
        code=code, submitted=submitted, context=context, sleep=sleep, task_stream=task_stream
    )


def _auto_selection(*, reports: Sequence[RunReport], context: Context) -> list[RunReport]:
    """The reports the follow-up job may take, by the retry mode of their grid."""
    followup = context.machine.followup
    chained = chained_grids(study=context.study)
    interrupted = [
        report
        for report in reports
        if report.state is StudyState.INTERRUPTED
        and _retry_allowed(grid=report.run.grid, chained=chained, followup=followup)
    ]
    taken = _with_progress(reports=interrupted, context=context)
    if not chained or followup.chained_retry is ChainedRetry.MANUAL:
        return taken
    everything = study_reports(context=context)
    # why: a dependency failed run is taken when its upstream is met; one whose upstream
    # is resubmitted in this call comes along as downstream of it
    taken += [
        report
        for report in reports
        if report.state is StudyState.DEPENDENCY_FAILED
        and report.run.grid in chained
        and after_verdict(
            study=context.study, run=report.run, reports=everything, followup=followup
        )
        is AfterVerdict.MET
    ]
    return taken


def _verdict_met(
    *, context: Context, run: StudyRun, reports: Mapping[PurePosixPath, RunReport]
) -> bool:
    verdict = after_verdict(
        study=context.study, run=run, reports=reports, followup=context.machine.followup
    )
    return verdict is AfterVerdict.MET


def _retry_allowed(*, grid: str, chained: frozenset[str], followup: FollowupSettings) -> bool:
    """Whether the retry mode of the grid lets the follow-up resubmit its runs."""
    if grid in chained:
        return followup.chained_retry is not ChainedRetry.MANUAL
    return followup.retry is Retry.RESUBMIT


def _with_downstream(
    *, context: Context, selected: list[RunReport]
) -> tuple[Context, list[RunReport], list[RunReport]]:
    """Add the waiting and dependency failed runs that hang on the selected ones.

    Returns:
        The context widened to hold them, the selection with the dependency failed
        ones added, and the waiting ones, whose jobs the caller cancels.
    """
    if not chained_grids(study=context.study):
        return context, selected, []
    everything = study_reports(context=context)
    chosen = {report.run.run_path for report in selected}
    extra = downstream_of(study=context.study, run_paths=chosen, reports=list(everything.values()))
    if not extra:
        return context, selected, []
    context = widen_context(context=context, reports=extra)
    waiting = [report for report in extra if report.state in LIVE_STATES]
    failed = [report for report in extra if report.state not in LIVE_STATES]
    return context, [*selected, *failed], waiting


def _with_cancelled_upstream(
    *, context: Context, selected: list[RunReport]
) -> tuple[Context, list[RunReport]]:
    """Add the cancelled upstream runs that keep selected runs ``dependency_failed``.

    A run cancelled by the user (``study cancel``, ``scancel``) reads ``failed``, so the
    default states leave it out, and its downstream runs could never be submitted.

    Returns:
        The context widened to hold them and the selection with them added.
    """
    if not chained_grids(study=context.study):
        return context, selected
    everything = study_reports(context=context)
    chosen = {report.run.run_path: report for report in selected}
    pending = [report for report in selected if report.state is StudyState.DEPENDENCY_FAILED]
    while pending:
        found = _cancelled_upstream(
            context=context, report=pending.pop(), everything=everything, chosen=chosen
        )
        chosen.update({report.run.run_path: report for report in found})
        pending += found
    given = {report.run.run_path for report in selected}
    extra = [report for path, report in chosen.items() if path not in given]
    if not extra:
        return context, selected
    for report in extra:
        _LOGGER.info('also resubmitting %s: it was cancelled', report.run.run_path)
    return widen_context(context=context, reports=extra), [*selected, *extra]


def _cancelled_upstream(
    *,
    context: Context,
    report: RunReport,
    everything: Mapping[PurePosixPath, RunReport],
    chosen: Mapping[PurePosixPath, RunReport],
) -> list[RunReport]:
    """The cancelled upstream runs of one report that are not chosen yet."""
    paths = dict.fromkeys(
        upstream.run_path
        for after in context.study.grids[report.run.grid].after
        for upstream in upstream_runs(study=context.study, run=report.run, after=after)
        if upstream.run_path not in chosen
    )
    return [
        everything[path]
        for path in paths
        if path in everything and _cancelled(report=everything[path])
    ]


def _cancelled(*, report: RunReport) -> bool:
    """Whether a failed run ended because its job was cancelled."""
    return report.state is StudyState.FAILED and JobFlag.CANCELLED in report.job_flags


def _code_snapshots_of(
    *,
    reports: Sequence[RunReport],
    reuse: Mapping[PurePosixPath, tuple['JobRecord', 'TaskRecord']],
    new: CodeSnapshot | None,
    drop: bool,
    states: tuple[StudyState, ...],
) -> dict[PurePosixPath, CodeSnapshot | None]:
    """The code snapshot each selected run resubmits from.

    Args:
        reports: The selected run reports.
        reuse: The latest job and task per run path.
        new: The snapshot every run switches to; ``None`` keeps the
            recorded one of each run.
        drop: Run the live code, even when the runs recorded a snapshot.
        states: The states of the selection; failed runs warn on reuse.

    Returns:
        The run path to code snapshot; ``None`` runs the live code.

    Raises:
        CodeSnapshotError: When a reused snapshot folder has no ``src/``
            left.
    """
    if new is not None:
        return {report.run.run_path: new for report in reports}
    if drop:
        return {report.run.run_path: None for report in reports}
    snapshots: dict[PurePosixPath, CodeSnapshot | None] = {}
    for report in reports:
        latest = reuse.get(report.run.run_path)
        snapshots[report.run.run_path] = latest[0].code_snapshot if latest is not None else None
    _check_reused_folders(snapshots=snapshots)
    _warn_reused_snapshots(reuse=reuse, snapshots=snapshots, states=states)
    return snapshots


def _check_reused_folders(*, snapshots: Mapping[PurePosixPath, CodeSnapshot | None]) -> None:
    """Every reused snapshot folder must still hold a ``src/`` copy."""
    for snapshot in dict.fromkeys(snapshots.values()):
        if snapshot is None or (snapshot.folder / SRC_FOLDER).is_dir():
            continue
        message = (
            f'the code snapshot folder {snapshot.folder} has no src/ left; the runs '
            'cannot resubmit from it; pass --new-code-snapshot to copy the code again'
        )
        raise CodeSnapshotError(message)


def _warn_reused_snapshots(
    *,
    reuse: Mapping[PurePosixPath, tuple['JobRecord', 'TaskRecord']],
    snapshots: Mapping[PurePosixPath, CodeSnapshot | None],
    states: tuple[StudyState, ...],
) -> None:
    """One warning per reused snapshot, when failed runs are resubmitted."""
    if StudyState.FAILED not in states:
        return
    warned: set[CodeSnapshot] = set()
    for path, snapshot in snapshots.items():
        if snapshot is None or snapshot in warned:
            continue
        warned.add(snapshot)
        latest = reuse.get(path)
        if latest is None:
            continue
        _LOGGER.warning(
            'reusing the code snapshot from %s; pass --new-code-snapshot if you fixed the code',
            latest[0].time,
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
) -> tuple[int | None, Callable[[], int]]:
    cancellable = backend.cancellable_job_ids(reports=live)
    if dry_run and cancellable:
        _LOGGER.info('would cancel %s', ', '.join(cancellable))

    def before_launch() -> int:
        code = (
            backend.cancel(
                context=dataclasses.replace(context, reports=list(live)), dry_run=False, signal=None
            )[0]
            if cancellable
            else 0
        )
        if code != 0:
            return code
        _remove_ended_locks(
            reports=selected,
            study_directory=context.study_directory,
            knows_other_hosts=backend.knows_other_hosts,
        )
        return 0

    delay = _RESTART_DELAY_SECONDS if cancellable else None
    return delay, before_launch


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
        shown = ', '.join(path.as_posix() for path in paths[:_SELECTION_SHOWN])
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
        indices=assign_indices(study=context.study, runs=[run], jobs=context.jobs),
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


def _reused_args(*, path: PurePosixPath, context: Context) -> tuple[str, ...]:
    """The extra args of the latest submission of one run, or none."""
    task = latest_tasks(jobs=context.jobs).get(path)
    return () if task is None else task[1].extra_args


def _with_progress(*, reports: Sequence[RunReport], context: Context) -> list[RunReport]:
    """The reports that may retry: under the cap, and one step further on.

    An attempt that ended ``PREEMPTED`` or ``NODE_FAIL`` does not count toward the
    cap and needs no progress, until 20 of them follow one another.
    """
    return [report for report in reports if may_retry(report=report, context=context)]


def may_retry(*, report: RunReport, context: Context) -> bool:
    """Whether one report is under the cap and one step further on; logs why not."""
    if JobFlag.NOT_COUNTED in report.job_flags:
        return _under_uncounted_limit(report=report)
    counted = report.attempt - (report.task.uncounted_attempts if report.task else 0)
    if counted >= context.machine.followup.max_attempts:
        _LOGGER.info('gave up: %s after %d attempts', report.run.run_path, counted)
        return False
    if not _made_progress(report=report, context=context):
        _LOGGER.info('stuck: %s (attempt %d, no progress)', report.run.run_path, report.attempt)
        return False
    return True


def _under_uncounted_limit(*, report: RunReport) -> bool:
    """Whether the run has not yet failed too many times in a row without counting."""
    if _uncounted(report=report)[1] <= _MAX_UNCOUNTED_IN_ROW:
        return True
    _LOGGER.info(
        'gave up: %s preempted or node failed %d times in a row',
        report.run.run_path,
        _MAX_UNCOUNTED_IN_ROW,
    )
    return False


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


def _remove_ended_locks(
    *, reports: Sequence[RunReport], study_directory: Path, knows_other_hosts: bool
) -> None:
    for report in reports:
        job_state = report.job_state
        if report.job_phase in (JobPhase.QUEUED, JobPhase.RUNNING, JobPhase.UNKNOWN):
            continue
        if not report.folder.lock_live:
            continue
        lock = study_directory / report.run.run_path / LOCK_FILE
        owner = read_lock_owner(path=lock)
        if owner is None:
            continue
        if owner.host == hostname():
            continue
        # why: local job state cannot see live processes on other hosts
        if not knows_other_hosts:
            _LOGGER.info(
                'kept the lock of %s: it is live on host %s', report.run.run_path, owner.host
            )
            continue
        lock.unlink(missing_ok=True)
        _LOGGER.info(
            'removed the lock of %s: its job %s ended (%s)',
            report.run.run_path,
            report.job_id or '-',
            job_state,
        )
