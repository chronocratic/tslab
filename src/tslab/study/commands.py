"""The ``tslab study`` commands: submit, status, resubmit and results.

Each command is a plain function over a ``Context`` (see ``tslab.study.context``).
The CLI dataclasses in ``tslab.cli.study`` parse the arguments and call these
functions, so the behavior is testable without the command line.
"""

from collections.abc import Callable, Mapping, Sequence
import logging
from pathlib import Path
import time

from tslab.core.paths import LOCK_FILE
from tslab.study.context import Context
from tslab.study.followups import queue_followups
from tslab.study.launch import launch, Launcher, LocalLauncher, SlurmLauncher
from tslab.study.machine import MachineProfile
from tslab.study.plan import assign_indices, next_job_file, plan_jobs, study_folder, validate_jobs
from tslab.study.records import JobRecord, latest_tasks, read_jobs, submission_lock, take_progress
from tslab.study.report import log_launch, log_problems, log_status
from tslab.study.result_table import collect_results, write_results_csv
from tslab.study.slurm import scancel, SlurmError
from tslab.study.state import job_ended, RunReport, StudyState

__all__ = ['LIVE_STATES', 'parse_states', 'resubmit', 'results', 'status', 'submit', 'submit_runs']

#: The states of a run that is queued or running; submit and resubmit leave them alone.
LIVE_STATES = (StudyState.QUEUED, StudyState.RUNNING)

_LOGGER = logging.getLogger(__name__)

# why: a cancelled job leaves its lock until the heartbeat is stale; the restart
# waits this many seconds (the stale-lock limit of the run engine plus a minute)
_RESTART_DELAY_SECONDS = 660


def submit(
    *,
    context: Context,
    include_done: bool,
    dry_run: bool,
    extra_args: tuple[str, ...],
    sleep: Callable[[float], None] = time.sleep,
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

    Returns:
        ``0`` when everything submitted, or nothing needed submitting;
        ``1`` when a run does not parse or a submission failed.
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
    )
    followup_code, _ = queue_followups(
        context=context, job_ids=[record.job_id for record in submitted], sleep=sleep
    )
    return max(code, followup_code)


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


def status(*, context: Context, states: tuple[StudyState, ...] = ()) -> int:
    """Print the state table of the study and one line per run not done.

    Args:
        context: The context to report on.
        states: When given, the listed lines are limited to these states.

    Returns:
        ``0``; the command only reads (spec §7).
    """
    log_status(context=context, states=states)
    return 0


def resubmit(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    states: tuple[StudyState, ...] = (StudyState.INTERRUPTED, StudyState.MISSING),
    cancel_live: bool = False,
    dry_run: bool = False,
    extra_args: tuple[str, ...] | None = None,
    auto: bool = False,
    sleep: Callable[[float], None] = time.sleep,
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

    Returns:
        ``0`` when everything submitted, or nothing needed submitting;
        ``1`` when a run does not parse or a submission failed.
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
    )
    followup_code, _ = queue_followups(
        context=context, job_ids=[record.job_id for record in submitted], sleep=sleep
    )
    return max(code, followup_code)


def results(*, context: Context, out: Path) -> int:
    """Write the results table of the runs of the context to ``out`` as CSV.

    Args:
        context: The context of the command.
        out: The CSV file to write.

    Returns:
        ``0``.
    """
    rows = collect_results(
        study=context.study, study_directory=context.study_directory, reports=context.reports
    )
    write_results_csv(rows=rows, path=out)
    _LOGGER.info('wrote %d rows to %s', len(rows), out.as_posix())
    return 0


def _with_progress(*, reports: Sequence[RunReport], context: Context) -> list[RunReport]:
    """The reports that may retry: under the cap, and one step further on."""
    max_attempts = context.machine.followup.max_attempts
    retried: list[RunReport] = []
    for report in reports:
        if report.attempt >= max_attempts:
            _LOGGER.info('gave up: %s after %d attempts', report.run.run_path, report.attempt)
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


def _cancel_live(*, reports: Sequence[RunReport], launcher: str) -> None:
    """Cancel the SLURM jobs of the live reports.

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
        scancel(job_ids=ids)
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


def _launcher_of(
    *, machine: MachineProfile, project_root: Path, sleep: Callable[[float], None]
) -> Launcher:
    if machine.launcher == 'slurm':
        return SlurmLauncher(sleep=sleep)
    return LocalLauncher(project_root=project_root, env=machine.env)


def submit_runs(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    reports: Sequence[RunReport],
    dry_run: bool,
    extra_args: Mapping[str, tuple[str, ...]],
    sleep: Callable[[float], None],
    begin: str | None = None,
    before_launch: Callable[[], None] | None = None,
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
        report = launch(
            launcher=_launcher_of(machine=machine, project_root=context.project_root, sleep=sleep),
            machine=machine,
            study=context.study,
            groups=groups,
            project_root=context.project_root,
            dry_run=dry_run,
            begin=begin,
        )
    return log_launch(report=report, groups=groups)
