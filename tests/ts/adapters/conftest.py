"""Skip collecting the adapter tests when the chronocratic extras are not installed.

The ``chronocratic`` marker deselects tests only after their module is imported,
and these modules import chronocratic at the top, so without the extras collection
itself would fail.
"""

from importlib.util import find_spec


def _installed() -> bool:
    try:
        return find_spec('chronocratic.models') is not None and (
            find_spec('chronocratic.datasets') is not None
        )
    except ModuleNotFoundError:
        return False


collect_ignore_glob = [] if _installed() else ['test_*.py']
