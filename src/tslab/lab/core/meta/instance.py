"""Instance metadata: the ``meta.json`` of one executed stage instance.

``meta.json`` is written to the result folder when an instance executes. It
ties the result to the code that made it (stage version, fingerprint,
environment) and to the identities of its inputs, so a later run can tell
whether the result is still current (``stale_reason``) and can load it with
its identity (``identity_from_meta``).
"""

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import cast, TYPE_CHECKING

from tslab.lab.core.artifact import Artifact, ArtifactIdentity
from tslab.lab.core.files import JsonFileError, read_json_object
from tslab.lab.core.paths import META_FILE
from tslab.lab.core.status import InstanceRecord, StatusFileError, utc_now_iso
from tslab.lab.core.text import qualified_name
from tslab.lab.enums import InstanceState

if TYPE_CHECKING:
    from typing import Any

    from tslab.lab.core.stage import _Execution, Stage

__all__ = [
    'can_resume',
    'describe_external_inputs',
    'describe_inputs',
    'identity_from_meta',
    'instance_meta_document',
    'read_instance_meta',
    'stale_reason',
]

_LOGGER = logging.getLogger(__name__)
# When the API is stable, version this format (a format_version key checked on read)
# so old files fail with a clear message.


def instance_meta_document(  # noqa: PLR0913 # why: one keyword per recorded fact
    *,
    stage: 'Stage[Any]',
    execution: '_Execution',
    artifact: Artifact,
    execution_id: str,
    started_at: str,
    seconds: float,
) -> dict[str, object]:
    """Return the content of the ``meta.json`` of one executed instance."""
    return {
        'instance_id': execution.instance_id,
        'stage_version': stage.version,
        'fingerprint': execution.fingerprint,
        'execution_id': execution_id,
        'artifact_type': qualified_name(cls=type(artifact)),
        'settings': dict(execution.settings),
        'inputs': describe_inputs(inputs=execution.inputs),
        'external_inputs': describe_external_inputs(inputs=execution.inputs),
        # The code and package versions that made this result. A resumed run
        # rewrites the run meta.json, so only here do they stay tied to the result.
        'environment': execution.run.environment(),
        'started_at': started_at,
        'finished_at': utc_now_iso(),
        'seconds': seconds,
    }


def stale_reason(
    *,
    record: InstanceRecord | None,
    fingerprint: str,
    directory: Path,
    inputs: Mapping[str, Artifact],
) -> str | None:
    """Return why the cached result cannot be used, or ``None`` if it can."""
    if record is None:
        return 'new'
    if record.fingerprint is not None and record.fingerprint != fingerprint:
        return f'fingerprint changed {record.fingerprint} -> {fingerprint}'
    if record.state is not InstanceState.DONE:
        return f'previous state {record.state.value}'
    return _metadata_reason(directory=directory, inputs=inputs)


def _metadata_reason(*, directory: Path, inputs: Mapping[str, Artifact]) -> str | None:
    """Why the metadata of a finished result cannot be trusted, or ``None``."""
    meta_path = directory / META_FILE
    if not meta_path.exists():
        return 'result folder missing'
    document = read_instance_meta(path=meta_path)
    if document is None:
        return 'cannot read result metadata'
    return _changed_input_reason(document=document, inputs=inputs)


def _changed_input_reason(
    *, document: Mapping[str, object], inputs: Mapping[str, Artifact]
) -> str | None:
    """Why an input was executed again since the result was made, or ``None``."""
    # TSLab writes ``inputs`` as a mapping of dicts; the cast names that shape.
    recorded_inputs = cast('dict[str, dict[str, str | None]]', document.get('inputs') or {})
    for key, value in inputs.items():
        identity = value.tslab_identity
        if identity is None or identity.instance_id is None:
            continue
        if recorded_inputs.get(key, {}).get('execution_id') != identity.execution_id:
            return f'input {identity.instance_id} was executed again'
    return None


def can_resume(*, record: InstanceRecord | None, fingerprint: str) -> bool:
    """Return whether the record describes an execution that may resume with this fingerprint."""
    return (
        record is not None
        and record.state in {InstanceState.RUNNING, InstanceState.FAILED}
        and record.fingerprint == fingerprint
    )


def identity_from_meta(*, directory: Path, instance_id: str) -> ArtifactIdentity:
    """Return the identity recorded in the ``meta.json`` of one instance.

    Raises:
        StatusFileError: If the file is missing or does not name a fingerprint
            and an execution id.
    """
    document = read_instance_meta(path=directory / META_FILE)
    fingerprint: object = None
    execution_id: object = None
    if document is not None:
        fingerprint = document.get('fingerprint')
        execution_id = document.get('execution_id')
    if not isinstance(fingerprint, str) or not isinstance(execution_id, str):
        message = f'cannot read result metadata of {instance_id}'
        raise StatusFileError(message)
    return ArtifactIdentity(
        fingerprint=fingerprint, instance_id=instance_id, execution_id=execution_id
    )


def read_instance_meta(*, path: Path) -> dict[str, object] | None:
    """Parse the ``meta.json`` of one instance; unreadable or non-object files give ``None``.

    A corrupt or partial file must never crash a run: the instance recomputes
    instead (see ``stale_reason`` and ``identity_from_meta``).
    """
    try:
        return read_json_object(path=path)
    except (OSError, JsonFileError):
        _LOGGER.warning('cannot read %s; treating the result as stale', path.as_posix())
        return None


def describe_inputs(*, inputs: Mapping[str, Artifact]) -> dict[str, dict[str, str | None]]:
    """Return the TSLab identity of each input that has one, as plain strings."""
    described: dict[str, dict[str, str | None]] = {}
    for key, value in inputs.items():
        identity = value.tslab_identity
        if identity is not None and identity.instance_id is not None:
            described[key] = {
                'instance_id': identity.instance_id,
                'fingerprint': identity.fingerprint,
                'execution_id': identity.execution_id,
            }
    return described


def describe_external_inputs(*, inputs: Mapping[str, Artifact]) -> dict[str, dict[str, str]]:
    """Return the external path and fingerprint of each input loaded from a file."""
    described: dict[str, dict[str, str]] = {}
    for key, value in inputs.items():
        identity = value.tslab_identity
        if identity is not None and identity.external_path is not None:
            described[key] = {
                'path': identity.external_path.as_posix(),
                'fingerprint': identity.fingerprint,
            }
    return described
