"""An organised output tree for tables and figures.

The layout is enforced in code::

    <root>/
      results_numbers.tex, results_numbers.csv   written by ``write_numbers``
      stats/<name>.csv                           test results
      tables/<name>.csv                          metrics and summaries
      coverage/<name>.csv                        missing and failed runs
      plots/<category>/<format>/<name>.<format>  one subfolder per chosen format
      <child>/                                   a nested tree with the same layout

Names are lowercase slugs, general to specific: ``<what>_<slice>``, for example
``cd_clean_classification_univariate`` or ``pairwise_wilcoxon_accuracy``.
"""

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
import io
import logging
from pathlib import Path
import re

from matplotlib.figure import Figure
import pandas as pd

from tslab.core.files import atomic_write_bytes, atomic_write_text
from tslab.core.paths import require_path

__all__ = [
    'DEFAULT_DPI',
    'DEFAULT_FORMATS',
    'PLOTS_FOLDER',
    'AnalysisOutput',
    'FigureFormat',
    'OutputFolder',
    'require_slug',
]

PLOTS_FOLDER = 'plots'
DEFAULT_DPI = 300
TABLE_SUFFIX = '.csv'
_SLUG_PATTERN = re.compile(r'[a-z0-9][a-z0-9_.-]*')
# why: EPS has no transparency, so matplotlib logs once per artist at this logger
_POSTSCRIPT_LOGGER = 'matplotlib.backends.backend_ps'


class OutputFolder(StrEnum):
    """The folders of the tree that hold tables."""

    STATS = 'stats'
    TABLES = 'tables'
    COVERAGE = 'coverage'


class FigureFormat(StrEnum):
    """The file types a figure can be saved as."""

    PDF = 'pdf'
    PNG = 'png'
    SVG = 'svg'
    EPS = 'eps'


DEFAULT_FORMATS = (FigureFormat.PDF, FigureFormat.PNG)


def require_slug(*, value: str, parameter: str) -> str:
    """Return ``value`` if it is a lowercase slug, otherwise raise ``ValueError``.

    Args:
        value: The name to check; a full match of ``[a-z0-9][a-z0-9_.-]*``.
        parameter: The parameter name, used in the error message.

    Returns:
        The same string.

    Raises:
        ValueError: If ``value`` is not a slug, so no file can escape its folder.
    """
    if not _SLUG_PATTERN.fullmatch(value):
        message = (
            f'{parameter} must be a lowercase slug matching {_SLUG_PATTERN.pattern!r}, '
            f'got {value!r}'
        )
        raise ValueError(message)
    return value


@dataclass(frozen=True, kw_only=True)
class AnalysisOutput:
    """A folder tree that receives tables and figures.

    Attributes:
        root: The folder of this tree.
        formats: The default file types of a saved figure.
        dpi: Resolution of raster formats.
    """

    root: Path
    formats: tuple[FigureFormat, ...] = DEFAULT_FORMATS
    dpi: int = DEFAULT_DPI

    def __post_init__(self) -> None:
        """Check the root and the formats."""
        require_path(value=self.root, parameter='root')
        if not self.formats:
            message = 'formats must name at least one figure format'
            raise ValueError(message)

    def table_path(self, *, folder: OutputFolder, name: str) -> Path:
        """Return the CSV path of a table.

        Args:
            folder: The folder of the tree the table belongs in.
            name: The table name, a lowercase slug.

        Returns:
            ``<root>/<folder>/<name>.csv``.
        """
        require_slug(value=name, parameter='name')
        return self.root / folder.value / f'{name}{TABLE_SUFFIX}'

    def save_table(
        self, *, frame: pd.DataFrame, folder: OutputFolder, name: str, index: bool = False
    ) -> Path:
        """Write a table as CSV, atomically.

        Args:
            frame: The table.
            folder: The folder of the tree the table belongs in.
            name: The table name, a lowercase slug.
            index: Whether to write the row index.

        Returns:
            The file written.
        """
        path = self.table_path(folder=folder, name=name)
        atomic_write_text(path=path, text=frame.to_csv(index=index, lineterminator='\n'))
        return path

    def save_figure(
        self,
        *,
        figure: Figure,
        category: str,
        name: str,
        formats: tuple[FigureFormat, ...] | None = None,
        tight: bool = False,
    ) -> tuple[Path, ...]:
        """Write a figure once per format, atomically.

        Every format is rendered before the first file is written, so a failing
        render leaves no file behind.

        Args:
            figure: The figure to save.
            category: The kind of figure (for example ``critical_difference``), a slug.
            name: The figure name, a slug.
            formats: File types for this figure; the tree default when ``None``.
            tight: Crop to the content and keep long axis labels.

        Returns:
            The files written, one per format.
        """
        require_slug(value=category, parameter='category')
        require_slug(value=name, parameter='name')
        chosen = self.formats if formats is None else formats
        labels = (
            [label for axis in figure.axes for label in (axis.xaxis.label, axis.yaxis.label)]
            if tight
            else None
        )
        logger = logging.getLogger(_POSTSCRIPT_LOGGER)
        previous = logger.level
        logger.setLevel(logging.ERROR)
        try:
            rendered = {}
            for figure_format in chosen:
                buffer = io.BytesIO()
                figure.savefig(
                    buffer,
                    format=figure_format.value,
                    dpi=self.dpi,
                    bbox_inches='tight' if tight else None,
                    bbox_extra_artists=labels,
                )
                rendered[figure_format] = buffer.getvalue()
        finally:
            logger.setLevel(previous)
        paths = tuple(
            self.root / PLOTS_FOLDER / category / fmt.value / f'{name}.{fmt.value}'
            for fmt in chosen
        )
        for path, data in zip(paths, rendered.values(), strict=True):
            atomic_write_bytes(path=path, data=data)
        return paths

    def child(self, *, name: str) -> 'AnalysisOutput':
        """Return a nested tree with the same settings under ``<root>/<name>``.

        Args:
            name: The folder name, a lowercase slug.

        Returns:
            The nested output.
        """
        require_slug(value=name, parameter='name')
        return dataclasses.replace(self, root=self.root / name)
