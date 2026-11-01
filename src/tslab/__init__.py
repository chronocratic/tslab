"""TSLab: resumable, cached research pipelines for time series representation learning."""

from importlib.metadata import PackageNotFoundError, version

from tslab import stages
from tslab.core.artifact import (
    Artifact,
    Checkpoint,
    is_available,
    Json,
    Metrics,
    Tensors,
    UnavailableArtifact,
    UnavailableArtifactError,
    UnavailableReason,
)
from tslab.core.config import ConfigFileError, load_config, parse_config, REQUIRED, save_config
from tslab.core.fingerprint import Fingerprintable, NotFingerprinted
from tslab.core.lock import RunLockedError
from tslab.core.paths import InstanceName
from tslab.core.run import JobTimeLimitSignal, Run, RunOptions
from tslab.core.stage import Stage, stage
from tslab.core.status import InstanceState, RunState
from tslab.core.tracker import RunStatus, Tracker
from tslab.stages.encoders import ModelAsEncoder

try:
    __version__ = version('tslab')
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = '0.0.0'

__all__ = [
    'REQUIRED',
    'Artifact',
    'Checkpoint',
    'ConfigFileError',
    'Fingerprintable',
    'InstanceName',
    'InstanceState',
    'JobTimeLimitSignal',
    'Json',
    'Metrics',
    'ModelAsEncoder',
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
    'stages',
]
