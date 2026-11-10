r"""Numbers for a paper: LaTeX macros and a CSV that a manuscript reads with ``\res{key}``.

Write the numbers once in the analysis and use them in the text, so a rerun updates the
paper and no number is copied by hand.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
import math
from pathlib import Path
import re

import pandas as pd

from tslab.lab.analysis.output import AnalysisOutput, require_slug
from tslab.lab.core.files import atomic_write_text

__all__ = [
    'NUMBERS_STEM',
    'NumberKind',
    'NumberRecord',
    'format_number',
    'number_key',
    'numbers_frame',
    'write_numbers',
]

NUMBERS_STEM = 'results_numbers'
KEY_COLUMN = 'key'
VALUE_COLUMN = 'value'
RAW_COLUMN = 'raw'
MISSING_TEXT = '--'
# A TeX control sequence name takes no ``$``, ``_``, space or ``.``. Applied in this order:
# the specific mathtext forms first, so ``$\ell_\infty$`` becomes ``Linf`` and
# ``$\ell_2$`` becomes ``L2`` before the general ``$`` and ``_`` rules run.
_KEY_REPLACEMENTS = (
    (r'$\ell_\infty$', 'Linf'),
    (r'$\ell_', 'L'),
    ('$', ''),
    ('_', '-'),
    (' ', '-'),
    ('.', ''),
)
_REPEATED_DASH = re.compile(r'-+')
_UNSAFE_KEY = re.compile(r'[%\\{}#&~^]')
# Text mode needs these escaped; the order does not matter because every key is replaced once.
_TEXT_ESCAPES = str.maketrans(
    {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
)
_ZERO_P_VALUE = r'<10^{-300}'


class NumberKind(StrEnum):
    """How a number is written in the text."""

    SCORE = 'score'
    SIG4 = 'sig4'
    RHO = 'rho'
    P = 'p'
    COUNT = 'count'
    TEXT = 'text'


type NumberRecord = tuple[str, float | str, NumberKind]


def number_key(*, parts: Sequence[str]) -> str:
    r"""Join name parts into a key a TeX control sequence name can hold.

    Args:
        parts: Pieces of the name, general to specific; empty pieces are skipped.

    Returns:
        The pieces joined with ``-``, with ``$``, ``_``, spaces and ``.`` replaced.

    Raises:
        ValueError: If the key still has a character TeX cannot take in a macro
            name (``% \ { } # & ~ ^``); rename the part instead of guessing.
    """
    cleaned = []
    for part in parts:
        for old, new in _KEY_REPLACEMENTS:
            part = part.replace(old, new)  # noqa: PLW2901 # why: each replacement builds on the last
        if part:
            cleaned.append(part)
    key = _REPEATED_DASH.sub('-', '-'.join(cleaned))
    unsafe = _UNSAFE_KEY.search(key)
    if unsafe:
        message = f'key {key!r} has {unsafe.group()!r}, which TeX cannot take in a macro name'
        raise ValueError(message)
    return key


def _ensure_math(*, text: str) -> str:
    return rf'\ensuremath{{{text}}}'


def _score(value: float | str) -> str:
    return _ensure_math(text=f'{float(value):.3f}')


def _significant(value: float | str) -> str:
    return _ensure_math(text=f'{float(value):.4g}')


def _rho(value: float | str) -> str:
    return _ensure_math(text=f'{float(value):.2f}')


def _p_value(value: float | str) -> str:
    number = float(value)
    if number == 0:
        return _ensure_math(text=_ZERO_P_VALUE)
    mantissa, exponent = f'{number:.1e}'.split('e')
    return _ensure_math(text=rf'{mantissa}\times10^{{{int(exponent)}}}')


def _count(value: float | str) -> str:
    return _ensure_math(text=str(round(float(value))))


def _text(value: float | str) -> str:
    return str(value).translate(_TEXT_ESCAPES)


_FORMATTERS: Mapping[NumberKind, Callable[[float | str], str]] = {
    NumberKind.SCORE: _score,
    NumberKind.SIG4: _significant,
    NumberKind.RHO: _rho,
    NumberKind.P: _p_value,
    NumberKind.COUNT: _count,
    NumberKind.TEXT: _text,
}
_ACCEPTED: Mapping[NumberKind, type | tuple[type, ...]] = {
    NumberKind.SCORE: (int, float),
    NumberKind.SIG4: (int, float),
    NumberKind.RHO: (int, float),
    NumberKind.P: (int, float),
    NumberKind.COUNT: (int, float),
    NumberKind.TEXT: str,
}


def format_number(*, value: float | str, kind: NumberKind) -> str:
    r"""Format a value for the text.

    Args:
        value: A number, or a string for ``NumberKind.TEXT``, which is escaped for LaTeX.
        kind: How to write it.

    Returns:
        The LaTeX text; a number in ``\ensuremath{}``, and ``--`` for a non-finite number.

    Raises:
        TypeError: If the value type does not fit the kind.
    """
    if not isinstance(value, _ACCEPTED[kind]):
        message = f'kind {kind.value!r} does not take a {type(value).__name__}'
        raise TypeError(message)
    if isinstance(value, float) and not math.isfinite(value):
        return MISSING_TEXT
    return _FORMATTERS[kind](value)


def numbers_frame(*, records: Iterable[NumberRecord]) -> pd.DataFrame:
    """Build the numbers table from ``(key, value, kind)`` records.

    Args:
        records: One tuple per number.

    Returns:
        A frame with ``key``, ``value`` (the formatted text) and ``raw`` (the number,
        NaN for text).
    """
    rows = [
        {
            KEY_COLUMN: key,
            VALUE_COLUMN: format_number(value=value, kind=kind),
            RAW_COLUMN: float('nan') if isinstance(value, str) else float(value),
        }
        for key, value, kind in records
    ]
    return pd.DataFrame(rows, columns=[KEY_COLUMN, VALUE_COLUMN, RAW_COLUMN])


def write_numbers(  # noqa: PLR0913 # why: one keyword per setting of the file, all independent
    *,
    numbers: pd.DataFrame,
    output: AnalysisOutput,
    source: str,
    now: datetime,
    stem: str = NUMBERS_STEM,
    note: str | None = None,
) -> tuple[Path, Path]:
    r"""Write the numbers as a LaTeX macro file and a CSV at the top of the output tree.

    In LaTeX, ``\input{results_numbers}`` and then ``\res{key}``.

    Args:
        numbers: The frame from :func:`numbers_frame`.
        output: The tree whose root receives both files.
        source: What generated the file, for the header comment.
        now: The time written in the header comment.
        stem: File name without suffix, a lowercase slug.
        note: An extra header line, such as the number of runs.

    Returns:
        The ``.tex`` path and the ``.csv`` path.

    Raises:
        ValueError: If a key is duplicated or ``stem`` is not a slug.
    """
    require_slug(value=stem, parameter='stem')
    repeated = numbers[KEY_COLUMN][numbers[KEY_COLUMN].duplicated()].unique().tolist()
    if repeated:
        message = f'duplicate number keys: {", ".join(repr(key) for key in repeated)}'
        raise ValueError(message)
    header = [f'% Generated by {source} on {now.date().isoformat()}. Do not edit by hand.']
    if note is not None:
        header.append(f'% {note}')
    macros = [
        rf'\expandafter\def\csname res@{key}\endcsname{{{value}}}'
        for key, value in zip(numbers[KEY_COLUMN], numbers[VALUE_COLUMN], strict=True)
    ]
    lines = [*header, r'\newcommand{\res}[1]{\csname res@#1\endcsname}', *macros]
    tex_path = output.root / f'{stem}.tex'
    csv_path = output.root / f'{stem}.csv'
    atomic_write_text(path=tex_path, text='\n'.join(lines) + '\n')
    atomic_write_text(path=csv_path, text=numbers.to_csv(index=False, lineterminator='\n'))
    return tex_path, csv_path
