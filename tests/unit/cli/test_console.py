from collections.abc import Iterator
from io import StringIO
import logging

import pytest

from tslab.cli.console import ConsoleHandler, make_console, watch
from tslab.cli.usage import UsageError


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    test_logger = logging.getLogger('tslab.test_console')
    original_level = test_logger.level
    original_propagate = test_logger.propagate
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = False
    yield test_logger
    test_logger.setLevel(original_level)
    test_logger.propagate = original_propagate


def _log(
    *,
    logger: logging.Logger,
    message: str,
    level: int,
    force_terminal: bool,
    width: int | None = None,
) -> str:
    buffer = StringIO()
    console = make_console(file=buffer, force_terminal=force_terminal)
    if width is not None:
        console.width = width
    handler = ConsoleHandler(console=console)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    try:
        logger.log(level, '%s', message)
    finally:
        logger.removeHandler(handler)
    return buffer.getvalue()


def test_handler_colors_state_words_on_a_terminal(logger: logging.Logger) -> None:
    output = _log(logger=logger, message='run-a  done', level=logging.INFO, force_terminal=True)

    assert '\x1b[' in output
    assert 'done' in output


def test_handler_writes_plain_text_off_a_terminal(logger: logging.Logger) -> None:
    output = _log(logger=logger, message='run-a  failed', level=logging.ERROR, force_terminal=False)

    assert output == 'error: run-a  failed\n'


def test_handler_prefixes_warnings(logger: logging.Logger) -> None:
    output = _log(logger=logger, message='x', level=logging.WARNING, force_terminal=False)

    assert output == 'warning: x\n'


def test_handler_never_wraps_long_lines(logger: logging.Logger) -> None:
    message = 'x' * 120
    output = _log(
        logger=logger, message=message, level=logging.INFO, force_terminal=False, width=40
    )

    assert output == f'{message}\n'


def test_handler_does_not_read_brackets_as_markup(logger: logging.Logger) -> None:
    output = _log(
        logger=logger, message='scale[f1] [red]x', level=logging.INFO, force_terminal=False
    )

    assert 'scale[f1]' in output
    assert '[red]' in output


def test_state_words_inside_other_words_are_not_highlighted(logger: logging.Logger) -> None:
    output = _log(
        logger=logger, message='--include-done undone', level=logging.INFO, force_terminal=True
    )

    assert '\x1b[' not in output


def test_no_color_env_disables_color(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('NO_COLOR', '1')
    output = _log(logger=logger, message='done', level=logging.INFO, force_terminal=True)

    assert '\x1b[3' not in output


def test_watch_without_seconds_shows_once() -> None:
    calls = 0

    def show() -> int:
        nonlocal calls
        calls += 1
        return 3

    code = watch(
        seconds=None, show=show, console=make_console(file=StringIO(), force_terminal=False)
    )

    assert code == 3
    assert calls == 1


def test_watch_rejects_less_than_one_second() -> None:
    with pytest.raises(UsageError, match='at least 1 second'):
        watch(
            seconds=0.5, show=lambda: 0, console=make_console(file=StringIO(), force_terminal=False)
        )


def test_watch_stops_on_keyboard_interrupt() -> None:
    shown: list[int] = []
    sleeps = 0

    def show() -> int:
        code = len(shown)
        shown.append(code)
        return code

    def sleep(seconds: float) -> None:
        nonlocal sleeps
        assert seconds == 1
        sleeps += 1
        if sleeps == 2:
            raise KeyboardInterrupt

    code = watch(
        seconds=1,
        show=show,
        console=make_console(file=StringIO(), force_terminal=False),
        sleep=sleep,
    )

    assert code == 1
    assert shown == [0, 1]
