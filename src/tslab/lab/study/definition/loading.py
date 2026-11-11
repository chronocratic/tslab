"""Load a study from a ``module:attribute`` reference in a module of the user."""

import importlib
from pathlib import Path
import sys

from tslab.lab.study.definition.study import Study
from tslab.lab.study.project import _drop_foreign_modules

__all__ = ['load_study']


def load_study(*, reference: str) -> Study:
    """The ``Study`` a ``module:attribute`` reference points at.

    The module is imported while the current working directory is on
    ``sys.path``, which is restored afterwards. A module of the same name
    cached from another folder is dropped first. An ``ImportError`` raised by
    the module of the user propagates unchanged, so a broken module breaks
    the study with its own error.

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
    if not separator or not module_name or not attribute_name or ':' in attribute_name:
        message = f'reference {reference!r} is malformed; references are module:attribute'
        raise ValueError(message)
    cwd = Path.cwd()
    _drop_foreign_modules(module_name=module_name, cwd=cwd)
    inserted = str(cwd) not in sys.path
    if inserted:
        sys.path.insert(0, str(cwd))
    try:
        module = importlib.import_module(module_name)
    except BaseException:
        # why: a half-imported module must not satisfy the next import
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted and str(cwd) in sys.path:
            sys.path.remove(str(cwd))
    if not hasattr(module, attribute_name):
        message = f'{module_name} has no attribute {attribute_name!r}'
        raise ValueError(message)
    study = getattr(module, attribute_name)
    if not isinstance(study, Study):
        message = f'{reference!r} does not name a Study, it is a {type(study).__name__}'
        raise TypeError(message)
    return study
