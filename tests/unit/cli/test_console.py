from collections.abc import Iterator
from io import StringIO
import logging
import sys
from typing import override

import pytest

import tslab.cli.console as console_module
from tslab.cli.console import ConsoleHandler, make_console, watch
from tslab.cli.usage import UsageError
from tslab.helpers.text import log_table


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


def test_default_handler_rechecks_stdout_after_redirection(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TerminalBuffer(StringIO):
        @override
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv('FORCE_COLOR', raising=False)
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    monkeypatch.setattr(sys, 'stdout', TerminalBuffer())
    console = make_console()
    assert console.is_terminal
    monkeypatch.setattr(console_module, 'CONSOLE', console)
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    handler = ConsoleHandler(console=console)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    try:
        logger.info('done')
    finally:
        logger.removeHandler(handler)

    assert buffer.getvalue() == 'done\n'


def test_force_color_does_not_color_redirected_output(
    logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('FORCE_COLOR', '1')
    buffer = StringIO()
    monkeypatch.setattr(sys, 'stdout', buffer)
    console = make_console()
    handler = ConsoleHandler(console=console)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    try:
        logger.info('done')
    finally:
        logger.removeHandler(handler)

    assert buffer.getvalue() == 'done\n'


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


def _log_table(*, logger: logging.Logger, force_terminal: bool) -> str:
    buffer = StringIO()
    handler = ConsoleHandler(console=make_console(file=buffer, force_terminal=force_terminal))
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    try:
        log_table(
            logger=logger,
            header=('grid', 'done', 'failed'),
            rows=[('a', '2', '0'), ('all', '2', '0')],
            right_align_numbers=True,
            total=True,
        )
    finally:
        logger.removeHandler(handler)
    return buffer.getvalue()


def test_handler_draws_a_table_on_a_terminal(logger: logging.Logger) -> None:
    output = _log_table(logger=logger, force_terminal=True)

    assert '─' in output
    assert '\x1b[' in output


def test_handler_writes_a_table_as_plain_lines_off_a_terminal(logger: logging.Logger) -> None:
    output = _log_table(logger=logger, force_terminal=False)

    assert output == 'grid  done  failed\na        2       0\nall      2       0\n'
