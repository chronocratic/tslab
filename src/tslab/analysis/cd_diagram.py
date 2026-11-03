"""A critical-difference diagram of average ranks, written from the method.

The Friedman test says whether the methods differ at all. When it does, the Nemenyi
critical difference ``CD`` is the smallest rank gap that counts as significant, and
methods whose average ranks lie closer than ``CD`` are joined by a bar.

The functions never print, never save and never touch the pyplot state.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
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

__all__ = ['CdDiagram', 'CdStyle', 'cd_diagram', 'cd_groups', 'critical_difference']


@dataclass(frozen=True, kw_only=True)
class CdStyle:
    """Layout of a critical-difference diagram; every size is in inches.

    Attributes:
        text_space: Room on each side of the rank scale for method names. It grows when
            the longest name needs more, but the scale keeps ``min_center_ratio`` of the
            figure width.
        min_center_ratio: Smallest share of the figure width left for the rank scale.
        top: Height above the rank scale, for the tick numbers and the CD marker.
        cd_height: Height of the CD marker above the rank scale.
        tick: Length of a whole-rank tick; the CD marker end ticks have the same length.
        half_tick: Length of a half-rank tick.
        elbow_start: Drop below the scale before the first no-difference bar.
        group_step: Vertical step between no-difference bars.
        group_overhang: How far a no-difference bar reaches past its outer methods.
        label_step: Vertical step between method names.
        elbow_overhang: How far a name line reaches past the end of the scale.
        text_gap: Gap between a line and its text.
        line_width: Width of the thin lines, in points.
        group_width: Width of the no-difference bars, in points.
    """

    text_space: float = 1.0
    min_center_ratio: float = 0.2
    top: float = 0.75
    cd_height: float = 0.4
    tick: float = 0.06
    half_tick: float = 0.03
    elbow_start: float = 0.25
    group_step: float = 0.09
    group_overhang: float = 0.04
    label_step: float = 0.2
    elbow_overhang: float = 0.05
    text_gap: float = 0.1
    line_width: float = 0.8
    group_width: float = 2.5


DEFAULT_CD_STYLE = CdStyle()
#: Default figure width in inches; pass ``FULL_WIDTH`` to fit the paper text width.
DEFAULT_CD_WIDTH = 6.0


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


def cd_diagram(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
    *,
    matrix: pd.DataFrame,
    direction: Direction,
    labels: Mapping[str, str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    width: float = DEFAULT_CD_WIDTH,
    reverse: bool = False,
    style: CdStyle = DEFAULT_CD_STYLE,
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
        reverse: Put rank 1 on the right instead of the left.
        style: Sizes and line widths of the drawing.

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
    figure = _draw(
        ranks=ranks, cd=cd, labels=labels or {}, width=width, reverse=reverse, style=style
    )
    return CdDiagram(ranks=ranks, cd=cd, friedman=test, figure=figure, caption=Caption(caption))


def _text_space(*, names: Sequence[str], figure: Figure, style: CdStyle) -> float:
    """Inches on each side of the scale: ``style.text_space``, or more for long names."""
    # why: hinting makes text width depend on dpi, so measure at the figure dpi
    measure = RendererAgg(1, 1, figure.dpi).get_text_width_height_descent
    longest = max(float(measure(name, FontProperties(), ismath=False)[0]) for name in names)
    needed = longest / float(figure.dpi) + style.elbow_overhang + style.text_gap
    width = float(figure.get_figwidth())
    widest = width * (1 - style.min_center_ratio) / 2
    return min(max(style.text_space, needed), widest)


def _draw(  # noqa: PLR0913 # why: private helper fed by cd_diagram, all keyword-only
    *,
    ranks: pd.Series,
    cd: float,
    labels: Mapping[str, str],
    width: float,
    reverse: bool,
    style: CdStyle,
) -> Figure:
    values = [float(value) for value in ranks]
    count = len(values)
    names = [labels.get(str(name), str(name)) for name in ranks.index]

    def x(rank: float) -> float:
        # the scale runs from 1 to count either way, so mirroring keeps it in place
        return count + 1 - rank if reverse else rank

    groups = cd_groups(ranks=values, cd=cd)
    left_count = math.ceil(count / 2)
    depth = style.elbow_start + style.group_step * len(groups) + style.label_step * (left_count + 1)
    figure = Figure(figsize=(width, style.top + depth))
    axis = figure.add_axes((0, 0, 1, 1))
    text_space = _text_space(names=names, figure=figure, style=style)
    # figure inches per rank unit, to turn sizes in inches into rank units on the x axis
    unit = (width - 2 * text_space) / (count - 1)
    axis.set_xlim(1 - text_space / unit, count + text_space / unit)
    axis.set_ylim(-depth, style.top)
    axis.set_axis_off()

    def line(xs: Sequence[float], ys: Sequence[float], width: float = style.line_width) -> None:
        axis.plot(xs, ys, color='black', linewidth=width, solid_capstyle='butt')

    line([1, count], [0, 0])
    for tick in range(1, count + 1):
        line([x(tick), x(tick)], [0, style.tick])
        axis.text(x(tick), style.tick + style.text_gap / 2, str(tick), ha='center', va='bottom')
    for half in range(1, count):
        line([half + 0.5] * 2, [0, style.half_tick])
    cd_y, cd_end = style.cd_height, 1 + cd
    line([1, cd_end], [cd_y, cd_y])
    for end_x in (1, cd_end):
        line([end_x, end_x], [cd_y - style.tick / 2, cd_y + style.tick / 2])
    axis.text((1 + cd_end) / 2, cd_y + style.tick, f'CD = {cd:.2f}', ha='center', va='bottom')
    overhang = style.group_overhang / unit
    for level, (first, last) in enumerate(groups):
        y = -style.elbow_start / 2 - style.group_step * level
        low, high = sorted((x(values[first]), x(values[last])))
        line([low - overhang, high + overhang], [y, y], width=style.group_width)
    first_row = -(style.elbow_start + style.group_step * len(groups))
    for position, (name, rank) in enumerate(zip(names, values, strict=True)):
        better_half = position < left_count
        row = position if better_half else count - 1 - position
        on_left = better_half != reverse
        side = -1 if on_left else 1
        y = first_row - style.label_step * (row + 1)
        x_end = (1 if on_left else count) + side * style.elbow_overhang / unit
        line([x(rank), x(rank), x_end], [0, y, y])
        axis.text(
            x_end + side * style.text_gap / unit,
            y,
            name,
            ha='right' if on_left else 'left',
            va='center',
        )
    return figure
