"""The ``tslab study`` commands: submit, status, resubmit and results.

Each command is a plain function over a ``Context`` (see ``tslab.study.context``).
The CLI dataclasses in ``tslab.cli.study`` parse the arguments and call these
functions, so the behavior is testable without the command line.
"""

from collections.abc import Callable, Sequence
from typing import TextIO

from tslab.enums import JobFlag, StudyState
from tslab.study.context import Context, refresh_context
from tslab.study.launchers.common.types import SIGNAL_EXIT_BASE
from tslab.study.launchers.tables import backend_for
from tslab.study.records import JobRecord
from tslab.study.reporting.state import RunReport

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
        backend_for(machine=context.machine).after_stopped_submit(
            context=context, submitted=submitted
        )
        return code
    return max(
        code,
        backend_for(machine=context.machine).after_submit(
            context=context,
            submitted=submitted,
            retry=lambda: resubmit(
                context=refresh_context(context=context),
                auto=True,
                sleep=sleep,
                task_stream=task_stream,
            ),
            sleep=sleep,
        ),
    )


def _uncounted(*, report: RunReport) -> tuple[int, int]:
    """The counters of the next attempt: ``(uncounted attempts, uncounted in a row)``."""
    attempts, in_row = (
        (report.task.uncounted_attempts, report.task.uncounted_in_row)
        if report.task is not None
        else (0, 0)
    )
    if JobFlag.NOT_COUNTED in report.job_flags:
        return attempts + 1, in_row + 1
    return attempts, 0


# why: the submodules import the shared helpers from this door
from tslab.study.commands.cancel import cancel  # noqa: E402
from tslab.study.commands.resubmit import resubmit, run_one  # noqa: E402
from tslab.study.commands.results import results  # noqa: E402
from tslab.study.commands.status import parse_states, status  # noqa: E402
from tslab.study.commands.submit import submit, submit_runs  # noqa: E402
