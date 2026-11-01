"""The status command: the state table of the study, read-only."""

from collections.abc import Mapping, Sequence

from tslab.study.context import Context
from tslab.study.reporting.report import log_status
from tslab.study.reporting.state import StudyState

__all__ = ['parse_states', 'status']


def parse_states(*, values: Sequence[str]) -> tuple[StudyState, ...]:
    """The state names of the ``--states`` flags, comma separated per flag.

    Args:
        values: The flag values, one per flag.

    Returns:
        The states in the order the names were given.

    Raises:
        ValueError: If a name is not a state name; the six names are named.
    """
    return tuple(_parse_state(name=raw.strip()) for value in values for raw in value.split(','))


def _parse_state(*, name: str) -> StudyState:
    """The state of one name, or a ``ValueError`` that lists the valid names."""
    try:
        return StudyState(name)
    except ValueError:
        allowed = ', '.join(state.value for state in StudyState)
        message = f'unknown state {name!r}; use one of: {allowed}'
        raise ValueError(message) from None


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
