"""Rich output for the command line; library logs stay plain for tests and non-terminal output."""

from collections.abc import Callable, Sequence
import logging
import sys
import time
from typing import ClassVar, IO, override

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.text import Text
from rich.theme import Theme

from tslab.cli.usage import UsageError

__all__ = ['CONSOLE', 'ConsoleHandler', 'configure_logging', 'make_console', 'watch']

_LOGGER = logging.getLogger(__name__)
_THEME = Theme(
    {
        'tslab.done': 'green',
        'tslab.failed': 'bold red',
        'tslab.interrupted': 'yellow',
        'tslab.running': 'cyan',
        'tslab.queued': 'blue',
        'tslab.missing': 'magenta',
        'tslab.skipped': 'dim',
    }
)


class _StateHighlighter(RegexHighlighter):
    base_style = 'tslab.'
    highlights: ClassVar[Sequence[str]] = (
        r'(?<![\w-])(?P<done>done)(?![\w-])',
        r'(?<![\w-])(?P<failed>failed)(?![\w-])',
        r'(?<![\w-])(?P<interrupted>interrupted)(?![\w-])',
        r'(?<![\w-])(?P<running>running)(?![\w-])',
        r'(?<![\w-])(?P<queued>queued)(?![\w-])',
        r'(?<![\w-])(?P<missing>missing)(?![\w-])',
        r'(?<![\w-])(?P<skipped>skipped)(?![\w-])',
    )


def make_console(*, file: IO[str] | None = None, force_terminal: bool | None = None) -> Console:
    """Create a console with TSLab state colors.

    Args:
        file: Output stream, or ``None`` to use standard output.
        force_terminal: Override terminal detection for tests.

    Returns:
        A Rich console configured for TSLab output.
    """
    stream = sys.stdout if file is None else file
    terminal = force_terminal if force_terminal is not None else stream.isatty()
    return Console(file=file, force_terminal=terminal, theme=_THEME, highlight=False)


CONSOLE: Console = make_console()


class ConsoleHandler(logging.Handler):
    """Format plain log records for terminal output."""

    def __init__(self, *, console: Console) -> None:
        """Create a handler that writes through a Rich console.

        Args:
            console: Console used to render each log record.
        """
        super().__init__()
        self._console = console
        self._uses_default_console = console is CONSOLE
        self._stdout: IO[str] | None = None

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._uses_default_console and self._stdout is not sys.stdout:
                self._console = make_console()
                self._stdout = sys.stdout
            message = self.format(record)
            style = None
            if record.levelno >= logging.ERROR:
                message = f'error: {message}'
                style = 'red'
            elif record.levelno >= logging.WARNING:
                message = f'warning: {message}'
                style = 'yellow'
            text = Text(message)
            if style is not None:
                text.stylize(style)
            text = _StateHighlighter()(text)
            self._console.print(text, soft_wrap=True)
        except Exception:  # noqa: BLE001 # why: logging handlers report emit failures through handleError
            self.handleError(record)


def configure_logging() -> None:
    """Install Rich logging unless another handler already owns the root logger."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = ConsoleHandler(console=CONSOLE)
    handler.setFormatter(logging.Formatter('%(message)s'))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def watch(
    *,
    seconds: float | None,
    show: Callable[[], int],
    console: Console = CONSOLE,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """Refresh a command until interrupted.

    Args:
        seconds: Refresh interval, or ``None`` for one display.
        show: Display the command and return its exit code.
        console: Console to clear between refreshes.
        sleep: Delay function, looked up from ``time.sleep`` when omitted.

    Returns:
        The exit code from the last completed display.

    Raises:
        UsageError: If the refresh interval is below one second.
    """
    if seconds is None:
        return show()
    if seconds < 1:
        message = '--watch takes at least 1 second'
        raise UsageError(message)
    wait = time.sleep if sleep is None else sleep
    code = 0
    try:
        while True:
            console.clear()
            code = show()
            _LOGGER.info('refreshing every %ss; press Ctrl-C to stop', format(seconds, 'g'))
            wait(seconds)
    except KeyboardInterrupt:
        return code
