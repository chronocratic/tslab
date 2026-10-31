"""The ``tslab study`` commands: submit, status, resubmit and results.

Each command is a plain function over a ``Context`` (see ``tslab.study.context``).
The CLI dataclasses in ``tslab.cli.study`` parse the arguments and call these
functions, so the behavior is testable without the command line.
"""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import json
import logging
import os
from pathlib import Path
import signal as signal_module
import subprocess
import sys
import time
from typing import TextIO

from tslab.core.paths import atomic_write_text, LOCK_FILE
from tslab.jobs.local import start_keep_awake, stop_keep_awake
from tslab.jobs.slurm import scancel, SlurmError
from tslab.study.context import Context, refresh_context
from tslab.study.followups import queue_followups
from tslab.study.job_script import foreground_script
from tslab.study.launch import (
    launch,
    Launcher,
    LaunchReport,
    local_job_id,
    LocalLauncher,
    resolve_local_gpus,
    SIGNAL_EXIT_BASE,
    SlurmLauncher,
)
from tslab.study.local_launch import hold_launch, launch_is_live, LocalLaunchError, read_launch
from tslab.study.machine import LocalSettings, MachineProfile
from tslab.study.plan import (
    assign_indices,
    JobGroup,
    next_job_file,
    plan_jobs,
    study_folder,
    validate_jobs,
)
from tslab.study.records import JobRecord, latest_tasks, read_jobs, submission_lock, take_progress
from tslab.study.report import log_launch, log_problems, log_status
from tslab.study.result_table import collect_results, format_results_csv, write_results_csv
from tslab.study.state import job_ended, RunReport, StudyState

__all__ = [
    'LIVE_STATES',
    'cancel',
    'parse_states',
    'resubmit',
    'results',
    'run_one',
    'status',
    'submit',
    'submit_runs',
]

#: The states of a run that is queued or running; submit and resubmit leave them alone.
LIVE_STATES = (StudyState.QUEUED, StudyState.RUNNING)

# why: a selection error lists this many run paths before counting the rest
_SELECTION_SHOWN = 20

_LOGGER = logging.getLogger(__name__)


def _finish_launch(
    *,
    code: int,
    submitted: Sequence[JobRecord],
    context: Context,
    sleep: Callable[[float], None],
    task_stream: TextIO | None,
) -> int:
    """The exit code of a launch, with its follow-ups queued when it was not stopped by a signal.

    On a local machine with ``followup.auto_retry`` the interrupted runs that made progress
    are submitted again, in this process; that call ends here too, so the chain runs until a
    pass retries nothing.
    """
    if code >= SIGNAL_EXIT_BASE:
        # why: the launch was stopped by a signal; resubmit continues it
        return code
    followup_code, _ = queue_followups(
        context=context, job_ids=[record.job_id for record in submitted], sleep=sleep
    )
    code = max(code, followup_code)
    if context.machine.launcher == 'local' and context.machine.followup.auto_retry and submitted:
        retry_code = resubmit(
            context=refresh_context(context=context),
            auto=True,
            sleep=sleep,
            task_stream=task_stream,
        )
        return max(code, retry_code)
    return code


# why: a cancelled job leaves its lock until the heartbeat is stale; the restart
# waits this many seconds (the stale-lock limit of the run engine plus a minute)
_RESTART_DELAY_SECONDS = 660


def submit(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    include_done: bool,
    dry_run: bool,
    extra_args: tuple[str, ...],
    sleep: Callable[[float], None] = time.sleep,
    on_launch: Callable[[LaunchReport, Sequence[JobGroup]], None] | None = None,
    task_stream: TextIO | None = None,
) -> int:
    """Submit the runs that need submitting, on the machine of the context.

    Queued and running runs are never submitted; done runs only when
    ``include_done``. Nothing is written when the arguments of one run do not
    parse.

    Args:
        context: The context to submit from.
        include_done: Submit the done runs as well.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The pass-through arguments, appended to every run.
        sleep: What waits out the submission pause of the cluster.
        on_launch: Called once with the launch result and groups, including dry runs.
        task_stream: Stream for task output on a local machine.

    Returns:
        ``0`` when everything submitted, or nothing needed submitting;
        ``1`` when a run does not parse or a submission failed;
        ``128 + N`` when the local launch was stopped by signal N.
    """
    to_submit = [
        report
        for report in context.reports
        if report.state not in LIVE_STATES and (report.state is not StudyState.DONE or include_done)
    ]
    if not to_submit:
        _LOGGER.info('nothing to submit')
        return 0
    code, submitted = submit_runs(
        context=context,
        reports=to_submit,
        dry_run=dry_run,
        extra_args={report.run.run_path: extra_args for report in to_submit},
        sleep=sleep,
        on_launch=on_launch,
        task_stream=task_stream,
    )
    return _finish_launch(
        code=code, submitted=submitted, context=context, sleep=sleep, task_stream=task_stream
    )


def parse_states(*, values: Sequence[str]) -> tuple[StudyState, ...]:
    """The state names of the ``--states`` flags, comma separated per flag.

    Args:
        values: The flag values, one per flag.

    Returns:
        The states in the order the names were given.

    Raises:
        ValueError: If a name is not a state name; the six names are named.
    """
    states: list[StudyState] = []
    for value in values:
        for raw in value.split(','):
            name = raw.strip()
            try:
                states.append(StudyState(name))
            except ValueError:
                allowed = ', '.join(state.value for state in StudyState)
                message = f'unknown state {name!r}; use one of: {allowed}'
                raise ValueError(message) from None
    return tuple(states)


def status(
    *,
    context: Context,
    states: tuple[StudyState, ...] = (),
    starts: Mapping[str, str | None] | None = None,
) -> int:
    """Print the state table of the study and one line per run not done.

    Args:
        context: The context to report on.
        states: When given, the listed lines are limited to these states.
        starts: When given, the lines of queued runs end with their expected
            start time.

    Returns:
        ``0``; the command only reads (spec §7).
    """
    log_status(context=context, states=states, starts=starts)
    return 0


def _cancel_local(*, context: Context, dry_run: bool, signal: str | None) -> tuple[int, list[str]]:
    if signal is not None:
        _LOGGER.error('cancel --signal is SLURM only')
        return 2, []
    launch = read_launch(study_folder=study_folder(machine=context.machine, study=context.study))
    if launch is None or not launch_is_live(launch=launch):
        _LOGGER.info('nothing to cancel')
        return 0, []
    selected = {report.job_id for report in context.reports}
    missed = [key for key, state in launch.tasks if state != 'ended' and key not in selected]
    if missed:
        _LOGGER.error(
            'a local launch is cancelled as one unit; the selection misses its tasks: %s',
            ', '.join(missed),
        )
        return 2, []
    if dry_run:
        _LOGGER.info('would stop local launch %s', launch.pid)
        return 0, [str(launch.pid)]
    os.kill(launch.pid, signal_module.SIGTERM)
    _LOGGER.info('sent SIGTERM to local launch %s', launch.pid)
    return 0, [str(launch.pid)]


def cancel(*, context: Context, dry_run: bool, signal: str | None = None) -> tuple[int, list[str]]:
    """Cancel or signal the queued and running SLURM jobs of selected runs.

    On a local machine it stops the live local launch with a SIGTERM; the
    selection must cover every live task, because the launch is one unit.

    Args:
        context: The study context containing selected run reports.
        dry_run: Log the job ids without cancelling or signalling them.
        signal: Send this signal to the running jobs instead of cancelling
            them, so the runs save and end on their own; queued runs are
            skipped because a pending job has no step to signal.

    Returns:
        The process exit code and the job ids cancelled, signalled, or
        selected for either.
    """
    if context.machine.launcher != 'slurm':
        return _cancel_local(context=context, dry_run=dry_run, signal=signal)
    live = [
        report
        for report in context.reports
        if report.state in LIVE_STATES and report.job_id is not None
    ]
    if signal is not None:
        queued = [
            report.job_id
            for report in live
            if report.state is StudyState.QUEUED and report.job_id is not None
        ]
        if queued:
            _LOGGER.info('skipped queued runs: %s', ', '.join(queued))
        live = [report for report in live if report.state is StudyState.RUNNING]
    job_ids = [report.job_id for report in live if report.job_id is not None]
    if not job_ids:
        _LOGGER.info('nothing to cancel')
        return 0, []
    if dry_run:
        if signal is not None:
            _LOGGER.info(
                'would signal %s with %s; the jobs save and end on their own',
                ', '.join(job_ids),
                signal,
            )
        else:
            _LOGGER.info('would cancel %s', ', '.join(job_ids))
        return 0, job_ids
    try:
        _cancel_live(reports=live, launcher=context.machine.launcher, signal=signal)
    except SlurmError as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: cancellation failures need no traceback
        return 1, []
    return 0, job_ids


def resubmit(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    states: tuple[StudyState, ...] = (StudyState.INTERRUPTED, StudyState.MISSING),
    cancel_live: bool = False,
    dry_run: bool = False,
    extra_args: tuple[str, ...] | None = None,
    auto: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    on_launch: Callable[[LaunchReport, Sequence[JobGroup]], None] | None = None,
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
    live = []
    if cancel_live:
        live = [
            report
            for report in context.reports
            if report.state in LIVE_STATES and report.job_id is not None
        ]
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
    if dry_run and live:
        _LOGGER.info('would cancel %s', ', '.join(str(report.job_id) for report in live))

    def before_launch() -> None:
        _cancel_live(reports=live, launcher=context.machine.launcher)
        _remove_ended_locks(reports=selected, study_directory=context.study_directory)

    code, submitted = submit_runs(
        context=context,
        reports=selected,
        dry_run=dry_run,
        extra_args=args,
        sleep=sleep,
        begin=f'now+{_RESTART_DELAY_SECONDS}' if live else None,
        before_launch=before_launch,
        on_launch=on_launch,
        task_stream=task_stream,
    )
    return _finish_launch(
        code=code, submitted=submitted, context=context, sleep=sleep, task_stream=task_stream
    )


def _run_in_terminal(args: list[str]) -> int:
    """The exit code of the command with the inherited standard streams."""
    # why: inherited streams, so breakpoint() and prompts reach the terminal
    return subprocess.run(args, check=False).returncode  # noqa: S603 # why: the run command, typed by the study


def run_one(
    *,
    context: Context,
    extra_args: tuple[str, ...] | None,
    run_process: Callable[[list[str]], int] = _run_in_terminal,
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
    )
    problems = validate_jobs(study=context.study, groups=groups)
    if problems:
        log_problems(problems=problems)
        return 1
    (task,) = groups[0].tasks
    if resolved.slurm is not None and resolved.slurm.gpus > 1:
        _LOGGER.info('study run starts one process; the job would start %d', resolved.slurm.gpus)
    script = foreground_script(
        slurm=resolved.slurm,
        project_root=context.project_root,
        env=context.machine.env,
        command=task.command,
    )
    return run_process(['bash', '-c', script])


def _reused_args(*, path: str, context: Context) -> tuple[str, ...]:
    """The extra args of the latest submission of one run, or none."""
    task = latest_tasks(jobs=context.jobs).get(path)
    return () if task is None else task[1].extra_args


def results(*, context: Context, out: Path, as_json: bool = False) -> int:
    """Write the results table of the runs as CSV or JSON.

    Args:
        context: The context of the command.
        out: The output file, or ``Path('-')`` for standard output.
        as_json: Write rows as a versioned JSON document.

    Returns:
        ``0``.
    """
    rows = collect_results(
        study=context.study, study_directory=context.study_directory, reports=context.reports
    )
    if out == Path('-'):
        text = (
            json.dumps({'schema': 'tslab.study-results/1', 'rows': rows}, indent=2)
            if as_json
            else format_results_csv(rows=rows)
        )
        sys.stdout.write(f'{text}\n' if as_json else text)
    elif as_json:
        atomic_write_text(
            path=out, text=json.dumps({'schema': 'tslab.study-results/1', 'rows': rows}, indent=2)
        )
    else:
        write_results_csv(rows=rows, path=out)
    _LOGGER.info('wrote %d rows to %s', len(rows), out.as_posix())
    return 0


#: Ended SLURM states that say nothing about the run: the node or the queue failed it.
_UNCOUNTED_JOB_STATES = frozenset({'PREEMPTED', 'NODE_FAIL'})

#: Preempted or node-failed attempts in a row after which the run is left alone.
_MAX_UNCOUNTED_IN_ROW = 20


def _uncounted(*, report: RunReport) -> tuple[int, int]:
    """The counters of the next attempt: ``(uncounted attempts, uncounted in a row)``."""
    attempts, in_row = (
        (report.task.uncounted_attempts, report.task.uncounted_in_row)
        if report.task is not None
        else (0, 0)
    )
    if report.job_state in _UNCOUNTED_JOB_STATES:
        return attempts + 1, in_row + 1
    return attempts, 0


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


def _cancel_live(*, reports: Sequence[RunReport], launcher: str, signal: str | None = None) -> None:
    """Cancel the SLURM jobs of the live reports, or send their steps a signal.

    Raises:
        SlurmError: If ``scancel`` fails.
    """
    if not reports:
        return
    if launcher != 'slurm':
        _LOGGER.info('skipped the live runs: only SLURM jobs can be cancelled')
        return
    ids = [report.job_id for report in reports if report.job_id is not None]
    if ids:
        scancel(job_ids=ids, signal=signal)
        if signal is not None:
            _LOGGER.info(
                'signalled %s with %s; the jobs save and end on their own', ', '.join(ids), signal
            )
        else:
            _LOGGER.info('cancelled %s', ', '.join(ids))


def _remove_ended_locks(*, reports: Sequence[RunReport], study_directory: Path) -> None:
    for report in reports:
        job_state = report.job_state
        if not job_ended(job_state) or not report.folder.lock_live:
            continue
        (study_directory / report.run.run_path / LOCK_FILE).unlink(missing_ok=True)
        _LOGGER.info(
            'removed the lock of %s: its job %s ended (%s)',
            report.run.run_path,
            report.job_id or '-',
            job_state,
        )


def _launcher_of(  # noqa: PLR0913 # why: keyword-only private helper; one keyword per launcher input
    *,
    machine: MachineProfile,
    project_root: Path,
    sleep: Callable[[float], None],
    task_stream: TextIO | None = None,
    gpu_ids: tuple[str, ...] = (),
    on_task_state: Callable[[str, str], None] | None = None,
) -> Launcher:
    if machine.launcher == 'slurm':
        return SlurmLauncher(sleep=sleep)
    return LocalLauncher(
        project_root=project_root,
        env=machine.env,
        local=machine.local or LocalSettings(),  # why: loaded local profiles always carry the table
        gpu_ids=gpu_ids,
        stream=task_stream,
        on_task_state=on_task_state,
    )


def _local_task_ids(*, groups: Sequence[JobGroup]) -> list[str]:
    return [
        f'{local_job_id(job_file=group.job_file)}_{task.index}'
        for group in groups
        for task in group.tasks
    ]


def _launch_held(  # noqa: PLR0913 # why: keyword-only private helper; one keyword per input of launch
    *,
    context: Context,
    groups: Sequence[JobGroup],
    folder: Path,
    dry_run: bool,
    begin: str | None,
    sleep: Callable[[float], None],
    task_stream: TextIO | None,
) -> LaunchReport | None:
    """Launch the groups; on a local machine hold the pid file and keep the machine awake.

    Returns:
        The launch report; ``None`` when a GPU or launch file problem was logged.
    """
    machine = context.machine
    gpu_ids: tuple[str, ...] = ()
    keep_awake: subprocess.Popen[bytes] | None = None
    is_local = machine.launcher != 'slurm' and not dry_run
    if is_local:
        # why: a loaded local profile always carries the table
        local = machine.local or LocalSettings()
        try:
            gpu_ids = resolve_local_gpus(local=local, environment=os.environ)
        except ValueError as error:
            _LOGGER.error('%s', error)  # noqa: TRY400 # why: a GPU setup error is shown, not raised
            return None
        _LOGGER.info('local GPUs: %s', ','.join(gpu_ids) if gpu_ids else 'none')
        if local.keep_awake:
            keep_awake = start_keep_awake(pid=os.getpid(), reason='tslab study launch')
    try:
        with contextlib.ExitStack() as stack:
            on_task_state = None
            if is_local:
                on_task_state = stack.enter_context(
                    hold_launch(study_folder=folder, task_ids=_local_task_ids(groups=groups))
                )
            return launch(
                launcher=_launcher_of(
                    machine=machine,
                    project_root=context.project_root,
                    sleep=sleep,
                    task_stream=task_stream,
                    gpu_ids=gpu_ids,
                    on_task_state=on_task_state,
                ),
                machine=machine,
                study=context.study,
                groups=groups,
                project_root=context.project_root,
                dry_run=dry_run,
                begin=begin,
            )
    except LocalLaunchError as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: a launch file problem is shown, not raised
        return None
    finally:
        stop_keep_awake(process=keep_awake)


def submit_runs(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    reports: Sequence[RunReport],
    dry_run: bool,
    extra_args: Mapping[str, tuple[str, ...]],
    sleep: Callable[[float], None],
    begin: str | None = None,
    before_launch: Callable[[], None] | None = None,
    on_launch: Callable[[LaunchReport, Sequence[JobGroup]], None] | None = None,
    task_stream: TextIO | None = None,
) -> tuple[int, list[JobRecord]]:
    """Plan, validate and launch the given runs; the shared tail of the commands.

    A run keeps the array index of its first submission; its attempt is one
    past its latest recorded attempt. The planning and the launch hold the
    submission lock of the study, and read the job records again under it.

    Args:
        context: The context to submit from.
        reports: The state reports of the runs to submit.
        dry_run: Write the job files and print the plan, submit nothing.
        extra_args: The run path to pass-through arguments.
        sleep: What waits out the submission pause of the cluster.
        begin: A SLURM ``--begin`` for the submitted jobs.
        before_launch: Called once the plan is valid and before the first job
            is submitted, never on a dry run; a ``SlurmError`` from it stops
            the command with exit code 1.
        on_launch: Called once with the launch result and groups, including dry runs.
        task_stream: Stream for task output on a local machine.

    Returns:
        The exit code and the submitted run jobs; empty records on a dry
        run or when nothing was submitted.
    """
    machine = context.machine
    if machine.launcher == 'slurm' and not context.job_states_known and not dry_run:
        _LOGGER.error('the SLURM job states could not be read; not submitting, to avoid duplicates')
        return 1, []
    runs = [report.run for report in reports]
    folder = study_folder(machine=machine, study=context.study)
    with submission_lock(study_folder=folder):
        jobs = read_jobs(path=folder / 'submissions.jsonl')
        groups = plan_jobs(
            study=context.study,
            machine=machine,
            resolved=[context.resolved[run.run_path] for run in runs],
            indices=assign_indices(runs=runs, jobs=jobs),
            extra_args=extra_args,
            attempts={report.run.run_path: report.attempt for report in reports},
            uncounted={report.run.run_path: _uncounted(report=report) for report in reports},
            first_job_file=next_job_file(study_folder=folder),
        )
        problems = validate_jobs(study=context.study, groups=groups)
        if problems:
            log_problems(problems=problems)
            return 1, []
        failure = None
        if before_launch is not None and not dry_run:
            try:
                before_launch()
            except SlurmError as error:
                failure = str(error)
        if failure is not None:
            _LOGGER.error('%s', failure)
            return 1, []
        report = _launch_held(
            context=context,
            groups=groups,
            folder=folder,
            dry_run=dry_run,
            begin=begin,
            sleep=sleep,
            task_stream=task_stream,
        )
        if report is None:
            return 1, []
        if on_launch is not None:
            on_launch(report, groups)
    return log_launch(report=report, groups=groups)
