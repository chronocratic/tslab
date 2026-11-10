"""Blocks a notebook shows in order: captions, text, tables and figures."""

from collections.abc import Callable, Mapping, Sequence
import html

from matplotlib.figure import Figure
import pandas as pd

__all__ = ['Block', 'Caption', 'show']


class Caption(str):
    """Text shown right before a figure or table, as an italic paragraph.

    Adding text gives a plain ``str``; apply a prefix as ``Caption(prefix + caption)``.
    """

    __slots__ = ()


type Block = str | Caption | pd.DataFrame | Figure


def show(*, blocks: Sequence[Block]) -> None:
    """Display blocks in order in a notebook.

    A ``Caption`` is an italic paragraph and a plain ``str`` is preformatted text, both
    HTML-escaped; tables and figures go through ``IPython.display.display``.

    Args:
        blocks: What to display.

    Raises:
        TypeError: If a block has none of the block types.
    """
    # why: IPython exists only in a notebook kernel
    from IPython.display import display, HTML  # noqa: PLC0415

    handlers: Mapping[type, Callable[..., object]] = {
        Caption: lambda block: display(HTML(f'<p><em>{html.escape(str(block))}</em></p>')),
        str: lambda block: display(HTML(f'<pre>{html.escape(str(block))}</pre>')),
        pd.DataFrame: display,
        Figure: display,
    }
    for block in blocks:
        handler = next(
            (handle for kind, handle in handlers.items() if isinstance(block, kind)), None
        )
        if handler is None:
            message = f'cannot show a block of type {type(block).__name__}'
            raise TypeError(message)
        handler(block)
