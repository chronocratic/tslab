"""Load a study from a ``module:attribute`` reference in a user's module."""

import importlib
from pathlib import Path
import sys

from tslab.study.definition import Study

__all__ = ['load_study']


def load_study(*, reference: str) -> Study:
    """The ``Study`` a ``module:attribute`` reference points at.

    The module is imported after the current working directory is on
    ``sys.path``; an ``ImportError`` raised by the user's module propagates
    unchanged, so a broken module breaks the study with its own error.

    Args:
        reference: ``module:attribute``; the attribute must be a ``Study``.

    Returns:
        The ``Study`` the attribute holds.

    Raises:
        ValueError: If the reference does not hold exactly one ``:``, or the
            module has no such attribute.
        TypeError: If the attribute is not a ``Study``.
    """
    module_name, separator, attribute_name = reference.partition(':')
    if not separator or not module_name or not attribute_name:
        message = f'reference {reference!r} is malformed; references are module:attribute'
        raise ValueError(message)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    if not hasattr(module, attribute_name):
        message = f'module {module_name!r} has no attribute {attribute_name!r}'
        raise ValueError(message)
    study = getattr(module, attribute_name)
    if not isinstance(study, Study):
        message = f'{reference!r} does not name a Study, it is a {type(study).__name__}'
        raise TypeError(message)
    return study
