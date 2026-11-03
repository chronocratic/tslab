"""A critical-difference diagram of average ranks, written from the method.

The Friedman test says whether the methods differ at all. When it does, the Nemenyi
critical difference ``CD`` is the smallest rank gap that counts as significant, and
methods whose average ranks lie closer than ``CD`` are joined by a bar.

The functions never print, never save and never touch the pyplot state.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import matplotlib as mpl
from matplotlib.figure import Figure
import pandas as pd
import scipy.stats

from tslab.analysis.report import Caption
from tslab.analysis.stats import (
    average_ranks,
    DEFAULT_ALPHA,
    Direction,
    friedman,
    FriedmanTest,
    MIN_FRIEDMAN_DATASETS,
    MIN_FRIEDMAN_METHODS,
)
from tslab.analysis.style import FULL_WIDTH

__all__ = ['CdDiagram', 'cd_diagram', 'cd_groups', 'critical_difference']

# Sizes in inches: the figure data coordinates are inches, so spacing is exact.
_LABEL_STEP = 0.2
_GROUP_STEP = 0.09
_TICK = 0.06
_ELBOW_START = 0.25
_TOP = 0.75
_BAR_HEIGHT = 0.4
_GROUP_OVERHANG = 0.04
_LINE_WIDTH = 0.8
_GROUP_WIDTH = 2.5
_SIDE_PADDING_RATIO = 0.5
_MIN_SIDE_PADDING = 1.0
_TEXT_GAP = 0.1
_CHARACTER_WIDTH = 0.07  # inches of one character at the base font size
_BASE_FONT = 8.0
_MIN_CENTER_RATIO = 0.25


@dataclass(frozen=True, kw_only=True)
class CdDiagram:
    """A critical-difference diagram and the numbers behind it.

    Attributes:
        ranks: Average rank of each method over complete rows, best first.
        cd: The Nemenyi critical difference (NaN when it cannot be computed).
        friedman: The Friedman test over the complete rows.
        figure: The diagram, or ``None`` when there is nothing to draw.
        caption: Text for the figure, or the reason there is no figure.
    """

    ranks: pd.Series
    cd: float
    friedman: FriedmanTest
    figure: Figure | None
    caption: Caption


def critical_difference(
    *, method_count: int, dataset_count: int, alpha: float = DEFAULT_ALPHA
) -> float:
    """Return the Nemenyi critical difference of average ranks.

    Args:
        method_count: Number of methods ``k``.
        dataset_count: Number of datasets ``N``.
        alpha: Significance level.

    Returns:
        ``q_alpha * sqrt(k (k + 1) / (6 N))`` with ``q_alpha`` the studentized range
        quantile at ``1 - alpha`` divided by ``sqrt(2)``.
    """
    q_alpha = float(scipy.stats.studentized_range.ppf(1 - alpha, method_count, math.inf))
    q_alpha /= math.sqrt(2)
    return q_alpha * math.sqrt(method_count * (method_count + 1) / (6 * dataset_count))


def cd_groups(*, ranks: Sequence[float], cd: float) -> list[tuple[int, int]]:
    """Find the groups of methods that do not differ significantly.

    Args:
        ranks: Average ranks sorted ascending.
        cd: The critical difference.

    Returns:
        The maximal ``(first, last)`` index spans with ``last > first`` and a rank
        spread below ``cd``; a span inside another is dropped.
    """
    spans = []
    for first in range(len(ranks)):
        last = max(
            (index for index in range(first, len(ranks)) if ranks[index] - ranks[first] < cd),
            default=first,
        )
        if last > first:
            spans.append((first, last))
    return [
        span
        for span in spans
        if not any(other != span and other[0] <= span[0] and span[1] <= other[1] for other in spans)
    ]


def cd_diagram(
    *,
    matrix: pd.DataFrame,
    direction: Direction,
    labels: Mapping[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    width: float = FULL_WIDTH,
) -> CdDiagram:
    """Build a critical-difference diagram from a score matrix.

    Only complete rows are used. No figure is drawn when there are fewer than 2
    complete datasets or 3 methods, or when the Friedman test finds no difference.

    Args:
        matrix: Rows by dataset, columns by method.
        direction: Whether a larger score is better.
        labels: Display names for method columns; a missing one is shown as is.
        alpha: Significance level of the Friedman test and of the critical difference.
        width: Figure width in inches.

    Returns:
        The diagram with its ranks, test and caption.
    """
    test = friedman(matrix=matrix)
    datasets, methods = test.datasets, test.methods
    if datasets:
        ranks = average_ranks(matrix=matrix, direction=direction)
    else:
        ranks = pd.Series(float('nan'), index=matrix.columns, dtype=float)
    if methods < MIN_FRIEDMAN_METHODS or datasets < MIN_FRIEDMAN_DATASETS:
        caption = (
            f'No critical-difference diagram: {datasets} complete datasets and {methods} '
            f'methods; it needs at least {MIN_FRIEDMAN_DATASETS} datasets and '
            f'{MIN_FRIEDMAN_METHODS} methods.'
        )
        return CdDiagram(
            ranks=ranks, cd=float('nan'), friedman=test, figure=None, caption=Caption(caption)
        )
    cd = critical_difference(method_count=methods, dataset_count=datasets, alpha=alpha)
    # why: a NaN p-value (every score tied) is not below alpha, so it must not draw
    if not test.p_value < alpha:
        caption = (
            f'No critical-difference diagram: the Friedman test finds no difference between '
            f'the {methods} methods over {datasets} datasets (p = {test.p_value:.2g}, '
            f'alpha = {alpha}).'
        )
        return CdDiagram(ranks=ranks, cd=cd, friedman=test, figure=None, caption=Caption(caption))
    caption = (
        f'Critical-difference diagram over {datasets} datasets and {methods} methods: '
        f'average rank, 1 = best. Methods joined by a bar do not differ significantly '
        f'(Friedman p = {test.p_value:.2g}, Nemenyi CD = {cd:.2f} at alpha = {alpha}).'
    )
    figure = _draw(ranks=ranks, cd=cd, labels=labels or {}, width=width)
    return CdDiagram(ranks=ranks, cd=cd, friedman=test, figure=figure, caption=Caption(caption))


def _side_padding(*, count: int, names: Sequence[str], width: float) -> float:
    """Rank units of free room on each side so the longest name fits in the figure."""
    label_inches = (
        max(len(name) for name in names)
        * _CHARACTER_WIDTH
        * float(mpl.rcParams['font.size'])
        / _BASE_FONT
        + _TEXT_GAP
    )
    room = max(width - 2 * label_inches, _MIN_CENTER_RATIO * width)
    needed = label_inches * (count - 1) / room
    return float(max(_SIDE_PADDING_RATIO * (count - 1), _MIN_SIDE_PADDING, needed))


def _draw(*, ranks: pd.Series, cd: float, labels: Mapping[str, str], width: float) -> Figure:
    values = [float(value) for value in ranks]
    count = len(values)
    groups = cd_groups(ranks=values, cd=cd)
    left_count = math.ceil(count / 2)
    depth = _ELBOW_START + _GROUP_STEP * len(groups) + _LABEL_STEP * (left_count + 1)
    height = _TOP + depth
    figure = Figure(figsize=(width, height))
    axis = figure.add_axes((0, 0, 1, 1))
    padding = _side_padding(
        count=count, names=[labels.get(str(n), str(n)) for n in ranks.index], width=width
    )
    axis.set_xlim(1 - padding, count + padding)
    axis.set_ylim(-depth, _TOP)
    axis.set_axis_off()
    # figure inches per rank unit, to convert text offsets given in inches
    unit = width / (count - 1 + 2 * padding)
    axis.plot([1, count], [0, 0], color='black', linewidth=_LINE_WIDTH)
    for tick in range(1, count + 1):
        axis.plot([tick, tick], [0, _TICK], color='black', linewidth=_LINE_WIDTH)
        axis.text(tick, _TICK + _TEXT_GAP / 2, str(tick), ha='center', va='bottom')
    axis.plot([1, 1 + cd], [_BAR_HEIGHT, _BAR_HEIGHT], color='black', linewidth=_GROUP_WIDTH)
    axis.text(1 + cd / 2, _BAR_HEIGHT + _TICK, f'CD = {cd:.2f}', ha='center', va='bottom')
    for level, (first, last) in enumerate(groups):
        y = -_ELBOW_START / 2 - _GROUP_STEP * level
        axis.plot(
            [values[first] - _GROUP_OVERHANG, values[last] + _GROUP_OVERHANG],
            [y, y],
            color='black',
            linewidth=_GROUP_WIDTH,
            solid_capstyle='butt',
        )
    first_row = -(_ELBOW_START + _GROUP_STEP * len(groups))
    for position, (name, rank) in enumerate(zip(ranks.index, values, strict=True)):
        on_left = position < left_count
        row = position if on_left else count - 1 - position
        y = first_row - _LABEL_STEP * (row + 1)
        x_end = (1 - 0.05) if on_left else (count + 0.05)
        axis.plot([rank, rank, x_end], [0, y, y], color='black', linewidth=_LINE_WIDTH)
        axis.text(
            x_end + (-1 if on_left else 1) * _TEXT_GAP / unit,
            y,
            labels.get(str(name), str(name)),
            ha='right' if on_left else 'left',
            va='center',
        )
    return figure
