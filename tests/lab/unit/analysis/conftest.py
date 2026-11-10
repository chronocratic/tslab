"""The analysis tests need the ``analysis`` extra; the bare base install skips them."""

import importlib.util

_PACKAGES = ('pandas', 'scipy', 'matplotlib')

collect_ignore_glob = (
    ['test_*.py'] if any(importlib.util.find_spec(name) is None for name in _PACKAGES) else []
)
