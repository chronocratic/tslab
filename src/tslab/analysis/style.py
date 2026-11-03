"""Figure style for a paper page: widths, one font size and a text colour helper."""

from collections.abc import Generator
import contextlib
from typing import Any

import matplotlib as mpl
from matplotlib.image import AxesImage
import numpy as np

__all__ = [
    'DARK_LUMINANCE',
    'DEFAULT_FONT_SIZE',
    'FULL_WIDTH',
    'PAPER_DPI',
    'is_dark_cell',
    'paper_style',
]

#: Text width of a journal page in inches (13.1 cm), to pass as a figure ``width``.
FULL_WIDTH = 13.1 / 2.54
DEFAULT_FONT_SIZE = 8
PAPER_DPI = 150
#: Perceived luminance below this counts as a dark cell.
DARK_LUMINANCE = 0.5
_FONT_KEYS = (
    'font.size',
    'axes.titlesize',
    'axes.labelsize',
    'xtick.labelsize',
    'ytick.labelsize',
    'legend.fontsize',
    'figure.labelsize',
)
_RED_WEIGHT = 0.299
_GREEN_WEIGHT = 0.587
_BLUE_WEIGHT = 0.114


@contextlib.contextmanager
def paper_style(*, font_size: float = DEFAULT_FONT_SIZE) -> Generator[None]:
    """Use one font size for every text element, so figures read at print size.

    Args:
        font_size: Size in points of every text element.

    Yields:
        Nothing; the settings apply inside the ``with`` block and are restored after.
    """
    settings: dict[Any, Any] = dict.fromkeys(_FONT_KEYS, font_size)
    settings['figure.dpi'] = PAPER_DPI
    with mpl.rc_context():
        mpl.rcParams.update(settings)
        yield


def is_dark_cell(*, image: AxesImage, value: float) -> bool:
    """Tell whether a heatmap cell is dark, to choose a light or dark text colour.

    Args:
        image: The heatmap image, whose colormap and norm give the cell colour.
        value: The cell value.

    Returns:
        ``True`` when the perceived luminance of the cell is below 0.5.
    """
    red, green, blue, _alpha = image.cmap(image.norm(np.asarray(value)))
    return bool(_RED_WEIGHT * red + _GREEN_WEIGHT * green + _BLUE_WEIGHT * blue < DARK_LUMINANCE)
