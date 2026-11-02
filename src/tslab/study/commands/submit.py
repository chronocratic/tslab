"""The submit command: plan, validate and launch the runs that need submitting."""

from collections.abc import Callable, Mapping, Sequence
import logging
import os
import socket
import time
from typing import TextIO

from tslab.core.status import utc_now_iso
from tslab.study.code_snapshot import CodeSnapshot
from tslab.study.commands import _finish_launch, _uncounted, LIVE_STATES
from tslab.study.context import Context
from tslab.study.launchers import backend_for, LaunchError, OnLaunch
from tslab.study.paths import next_job_file, study_folder
from tslab.study.plan import assign_indices, plan_jobs, validate_jobs
from tslab.study.records import JobRecord, read_jobs, submission_lock
from tslab.study.reporting.report import log_launch, log_problems
from tslab.study.reporting.state import RunReport, StudyState

__all__ = ['submit', 'submit_runs']

_LOGGER = logging.getLogger(__name__)


def submit(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    include_done: bool,
    dry_run: bool,
    extra_args: tuple[str, ...],
    code_snapshot: CodeSnapshot | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_launch: OnLaunch | None = None,
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
        code_snapshot: The code snapshot every submitted run runs from;
            ``None`` runs the live code.
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
        code_snapshots=(
            {report.run.run_path: code_snapshot for report in to_submit}
            if code_snapshot is not None
            else None
        ),
        sleep=sleep,
        on_launch=on_launch,
        task_stream=task_stream,
    )
    return _finish_launch(
        code=code, submitted=submitted, context=context, sleep=sleep, task_stream=task_stream
    )


def _before_launch_code(*, before_launch: Callable[[], int] | None, dry_run: bool) -> int:
    """The exit code of the hook that runs before the first submission; 0 to go on."""
    if before_launch is None or dry_run:
        return 0
    try:
        return before_launch()
    except LaunchError as error:
        _LOGGER.error('%s', error)  # noqa: TRY400 # why: a hook failure is shown, not raised
        return 1


def submit_runs(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    context: Context,
    reports: Sequence[RunReport],
    dry_run: bool,
    extra_args: Mapping[str, tuple[str, ...]],
    sleep: Callable[[float], None],
    start_delay_seconds: int | None = None,
    before_launch: Callable[[], int] | None = None,
    on_launch: OnLaunch | None = None,
    task_stream: TextIO | None = None,
    code_snapshots: Mapping[str, CodeSnapshot | None] | None = None,
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
        start_delay_seconds: Seconds the submitted jobs wait before they may start.
        before_launch: Called once the plan is valid and before the first job
            is submitted, never on a dry run; a ``LaunchError`` from it stops
            the command with exit code 1.
        on_launch: Called once with the launch result and groups, including dry runs.
        task_stream: Stream for task output on a local machine.
        code_snapshots: The run path to code snapshot; ``None`` when absent,
            which means the run submits with the live code.

    Returns:
        The exit code and the submitted run jobs; empty records on a dry
        run or when nothing was submitted.
    """
    machine = context.machine
    if not context.job_states.known and not dry_run:
        _LOGGER.error('the job states could not be read; not submitting, to avoid duplicates')
        return 1, []
    backend = backend_for(machine=machine)
    blocker = backend.launch_blocker(context=context)
    if blocker is not None:
        _LOGGER.error('%s', blocker)
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
            environment=os.environ,
            code_snapshots=code_snapshots,
        )
        problems = validate_jobs(study=context.study, groups=groups)
        if problems:
            log_problems(problems=problems)
            return 1, []
        before_code = _before_launch_code(before_launch=before_launch, dry_run=dry_run)
        if before_code != 0:
            return before_code, []
        report = backend.launch(
            context=context,
            groups=groups,
            folder=folder,
            dry_run=dry_run,
            start_delay_seconds=start_delay_seconds,
            sleep=sleep,
            task_stream=task_stream,
            environment=os.environ,
            pid=os.getpid(),
            host=socket.gethostname(),
            now=utc_now_iso,
        )
        if report is None:
            return 1, []
        if on_launch is not None:
            on_launch(report=report, groups=groups)
    return log_launch(report=report, groups=groups)
