"""Process exit codes for study states."""

from collections.abc import Iterable

from tslab.enums import StudyState
from tslab.study.context import Context
from tslab.study.reporting.state import RunReport

__all__ = [
    'EXIT_DONE',
    'EXIT_ERROR',
    'EXIT_FAILED',
    'EXIT_LIVE',
    'EXIT_RETRY',
    'EXIT_TIMEOUT',
    'EXIT_UNKNOWN',
    'EXIT_USAGE',
    'counts_live',
    'states_exit_code',
    'study_exit_code',
]

EXIT_DONE = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_LIVE = 3
EXIT_FAILED = 4
EXIT_RETRY = 5
EXIT_UNKNOWN = 6
EXIT_TIMEOUT = 124


def states_exit_code(*, states: Iterable[StudyState], live: bool) -> int:
    """The exit code for a collection of run states.

    Args:
        states: The states of the selected runs.
        live: Whether any run has pending work not shown by its state.

    Returns:
        The code matching the highest-priority state.
    """
    state_set = set(states)
    if StudyState.FAILED in state_set:
        return EXIT_FAILED
    if live or StudyState.QUEUED in state_set or StudyState.RUNNING in state_set:
        return EXIT_LIVE
    if StudyState.INTERRUPTED in state_set or StudyState.MISSING in state_set:
        return EXIT_RETRY
    return EXIT_DONE


def counts_live(*, report: RunReport, context: Context) -> bool:
    """Whether a run has work pending beyond its displayed state.

    Args:
        report: The state report of one selected run.
        context: The study context containing lock and follow-up state.

    Returns:
        Whether the run is queued, running, being submitted, or awaiting retry.
    """
    if report.state in (StudyState.QUEUED, StudyState.RUNNING):
        return True
    if (
        report.task is not None
        and context.submission_in_progress
        and (
            report.state is StudyState.INTERRUPTED
            or (
                report.state is StudyState.MISSING
                and not (context.study_directory / report.run.run_path).is_dir()
            )
        )
    ):
        return True
    return report.state is StudyState.INTERRUPTED and context.followup_live


def study_exit_code(*, context: Context) -> int:
    """The exit code of the selected runs in one study context.

    Args:
        context: The study state and machine information.

    Returns:
        The code matching the study state and any pending work.
    """
    if not context.job_states.known:
        return EXIT_UNKNOWN
    return states_exit_code(
        states=(report.state for report in context.reports),
        live=any(counts_live(report=report, context=context) for report in context.reports),
    )
