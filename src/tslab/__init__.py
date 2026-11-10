"""TSLab: resumable, cached research pipelines, in three layers.

``tslab.lab`` is the general lab, ``tslab.ml`` adds torch and Lightning, and
``tslab.ts`` adds time series representation learning.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('tslab')
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = '0.0.0'

__all__ = ['__version__']
