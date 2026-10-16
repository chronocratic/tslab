"""TSLab: resumable, cached research pipelines for time-series representation learning."""

from importlib.metadata import PackageNotFoundError, version

from tslab.artifact import (
    Artifact,
    is_available,
    Json,
    Metrics,
    Tensors,
    UnavailableArtifact,
    UnavailableArtifactError,
    UnavailableReason,
)
from tslab.config import ConfigFileError, load_config, parse_config, save_config
from tslab.fingerprint import Fingerprintable, NotFingerprinted
from tslab.interfaces.tracker import RunStatus, Tracker
from tslab.paths import InstanceName
from tslab.run import Run, RunLockedError, RunOptions
from tslab.stage import Stage, stage
from tslab.status import InstanceState, RunState

try:
    __version__ = version('tslab')
except (
    PackageNotFoundError
):  # pragma: no cover - only when running from a source tree without install
    __version__ = '0.0.0'

__all__ = [
    'Artifact',
    'ConfigFileError',
    'Fingerprintable',
    'InstanceName',
    'InstanceState',
    'Json',
    'Metrics',
    'NotFingerprinted',
    'Run',
    'RunLockedError',
    'RunOptions',
    'RunState',
    'RunStatus',
    'Stage',
    'Tensors',
    'Tracker',
    'UnavailableArtifact',
    'UnavailableArtifactError',
    'UnavailableReason',
    '__version__',
    'is_available',
    'load_config',
    'parse_config',
    'save_config',
    'stage',
]
