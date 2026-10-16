"""TSLab: resumable, cached research pipelines for time-series representation learning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('tslab')
except PackageNotFoundError:  # pragma: no cover - source tree that was never installed
    __version__ = '0.0.0'

__all__ = ['__version__']
