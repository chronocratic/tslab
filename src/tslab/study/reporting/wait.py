"""Wait for a study to finish or reach a state that needs attention."""

from collections.abc import Callable
import logging
import time

from tslab.study.context import Context
from tslab.study.reporting.outcome import (
    counts_live,
    EXIT_LIVE,
    EXIT_TIMEOUT,
    EXIT_UNKNOWN,
    study_exit_code,
)
from tslab.study.reporting.state import StudyState

__all__ = ['wait_for_study']

_LOGGER = logging.getLogger(__name__)
_FIRST_GAP_SECONDS = 30
_COUNT_ORDER = (
    StudyState.DONE,
    StudyState.RUNNING,
    StudyState.QUEUED,
    StudyState.INTERRUPTED,
    StudyState.FAILED,
    StudyState.MISSING,
)


def wait_for_study(  # noqa: PLR0913 # why: each keyword-only control is documented below
    *,
    load: Callable[[], Context],
    interval: float,
    timeout: float | None,
    keep_going: bool,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[int, Context, float]:
    """Reload a study until it stops or reaches the timeout.

    Args:
        load: Load the latest context of the selected study.
        interval: Longest gap between checks in seconds; checks start 30 seconds
            apart and slow down while nothing changes.
        timeout: Maximum seconds to wait, or ``None`` for no limit.
        keep_going: Wait for every live run instead of stopping on failure.
        sleep: Wait between checks.
        clock: Monotonic clock used to measure the timeout.

    Returns:
        The final exit code, last context, and elapsed seconds.
    """
    start = clock()
    previous_counts: tuple[int, ...] | None = None
    gap: float = min(_FIRST_GAP_SECONDS, interval)
    warned_unknown = False
    last_context: Context | None = None
    last_code: int | None = None
    while True:
        try:
            context = load()
            code = study_exit_code(context=context)
            last_context, last_code = context, code
            counts = _counts(context=context)
            if counts != previous_counts:
                _LOGGER.info('%s %s', time.strftime('%H:%M'), _counts_line(counts=counts))
                previous_counts = counts
                gap = min(_FIRST_GAP_SECONDS, interval)
            else:
                gap = min(gap * 2, interval)
            if code == EXIT_UNKNOWN and not warned_unknown:
                _LOGGER.warning('SLURM job states unknown; still waiting')
                warned_unknown = True
            elif code != EXIT_UNKNOWN:
                warned_unknown = False

            waited = clock() - start
            if _should_stop(context=context, code=code, keep_going=keep_going):
                return code, context, waited

            if timeout is not None and waited + gap > timeout:
                return EXIT_TIMEOUT, context, waited
            sleep(gap)
        except KeyboardInterrupt:
            if last_context is None or last_code is None:
                raise
            return last_code, last_context, clock() - start


def _should_stop(*, context: Context, code: int, keep_going: bool) -> bool:
    if keep_going:
        live = any(counts_live(report=report, context=context) for report in context.reports)
        return code != EXIT_UNKNOWN and not live
    return code not in (EXIT_LIVE, EXIT_UNKNOWN)


def _counts(*, context: Context) -> tuple[int, ...]:
    counts = dict.fromkeys(_COUNT_ORDER, 0)
    for report in context.reports:
        counts[report.state] += 1
    return tuple(counts[state] for state in _COUNT_ORDER)


def _counts_line(*, counts: tuple[int, ...]) -> str:
    return ' '.join(
        f'{state.value} {count}' for state, count in zip(_COUNT_ORDER, counts, strict=True)
    )
