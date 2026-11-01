"""Wait for a study to finish or reach a state that needs attention."""

from collections.abc import Callable
from dataclasses import dataclass, replace
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
    poll = _Poll(gap=min(_FIRST_GAP_SECONDS, interval))
    while True:
        result, poll = _step(
            poll=poll,
            load=load,
            interval=interval,
            timeout=timeout,
            keep_going=keep_going,
            sleep=sleep,
            elapsed=lambda: clock() - start,
        )
        if result is not None:
            return result


@dataclass(frozen=True, kw_only=True)
class _Poll:
    """What the wait loop remembers from one check to the next."""

    gap: float
    previous_counts: tuple[int, ...] | None = None
    warned_unknown: bool = False
    last_context: Context | None = None
    last_code: int | None = None


def _step(  # noqa: PLR0913 # why: the wait controls, passed on from wait_for_study
    *,
    poll: _Poll,
    load: Callable[[], Context],
    interval: float,
    timeout: float | None,
    keep_going: bool,
    sleep: Callable[[float], None],
    elapsed: Callable[[], float],
) -> tuple[tuple[int, Context, float] | None, _Poll]:
    """One check: the final result (or ``None`` after the pause) and the next poll state."""
    try:
        context = load()
        code = study_exit_code(context=context)
        poll = _note_progress(
            poll=replace(poll, last_context=context, last_code=code),
            context=context,
            code=code,
            interval=interval,
        )
        waited = elapsed()
        if _should_stop(context=context, code=code, keep_going=keep_going):
            return (code, context, waited), poll
        if timeout is not None and waited + poll.gap > timeout:
            return (EXIT_TIMEOUT, context, waited), poll
        sleep(poll.gap)
    except KeyboardInterrupt:
        if poll.last_context is None or poll.last_code is None:
            raise
        return (poll.last_code, poll.last_context, elapsed()), poll
    return None, poll


def _note_progress(*, poll: _Poll, context: Context, code: int, interval: float) -> _Poll:
    """Log a changed count line and the unknown-state warning once; the poll with its next gap."""
    counts = _counts(context=context)
    if counts != poll.previous_counts:
        _LOGGER.info('%s %s', time.strftime('%H:%M'), _counts_line(counts=counts))
        poll = replace(poll, previous_counts=counts, gap=min(_FIRST_GAP_SECONDS, interval))
    else:
        poll = replace(poll, gap=min(poll.gap * 2, interval))
    if code == EXIT_UNKNOWN and not poll.warned_unknown:
        _LOGGER.warning('job states unknown; still waiting')
        return replace(poll, warned_unknown=True)
    if code != EXIT_UNKNOWN:
        return replace(poll, warned_unknown=False)
    return poll


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
