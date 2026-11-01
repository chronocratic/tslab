"""Stages: cached, resumable units of work.

``Stage.__call__`` is a Template Method. It runs one fixed algorithm (claim the
name, fingerprint, decide load/run/skip, execute, save atomically, record the
status) and delegates only the actual work to ``execute``. Users write stages
with the ``@stage`` decorator or by subclassing ``Stage``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from itertools import count
import logging
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, cast, ClassVar, get_type_hints, override
import uuid

from tslab.core.artifact import (
    Artifact,
    ArtifactIdentity,
    bind_identity,
    UnavailableArtifact,
    UnavailableReason,
)
from tslab.core.fingerprint import (
    canonicalize,
    compute_fingerprint,
    FINGERPRINT_FORMAT_VERSION,
    is_not_fingerprinted,
)
from tslab.core.paths import (
    atomic_write_json,
    atomic_write_text,
    ERROR_FILE,
    JsonFileError,
    META_FILE,
    normalize_instance_name,
    read_json_object,
    replace_directory,
    validate_stage_name,
)
from tslab.core.ranks import is_writer_process, RankAgreement, slurm_task_count
from tslab.core.run import JobTimeLimitSignal, Run
from tslab.core.status import (
    format_instance_id,
    InstanceRecord,
    InstanceState,
    StatusFileError,
    utc_now_iso,
)
from tslab.core.text import describe_error, qualified_name

__all__ = ['INSTANCE_META_FORMAT_VERSION', 'FunctionStage', 'Stage', 'stage']

_LOGGER = logging.getLogger(__name__)
_AGREEMENT_SEQUENCE = count()
INSTANCE_META_FORMAT_VERSION = 1
_INCOMPLETE_MARKER = '.incomplete'
_RESERVED_ARGUMENTS = frozenset({'run', 'name'})


@dataclass(frozen=True, kw_only=True)
class _Execution:
    """What ``_invoke`` knows about one instance before executing it."""

    run: Run
    instance_id: str
    directory: Path
    fingerprint: str
    previous_fingerprint: str | None = None
    agreement_key: str
    settings: dict[str, object]
    inputs: Mapping[str, Artifact]
    agreement: RankAgreement | None = None


class Stage[Out: Artifact](ABC):
    """A cached, resumable unit of work. Subclasses implement ``execute`` only.

    Attributes:
        name: Stage name; becomes the folder name. Must match ``[A-Za-z0-9._-]+``.
        version: Bump it when the code of the stage changes in a way that changes
            results. Old cached results are then invalidated.
        output_type: The ``Artifact`` subclass ``execute`` returns.
        in_place: If ``True``, ``execute`` writes directly into the final
            instance folder (``run.instance_directory(...)``) and an unfinished
            folder is kept for resuming. Used by the ready ``train`` stage, so an
            interrupted training resumes from its last checkpoint.
        all_ranks: Run on every rank in a multi-rank job; other stages run only on rank zero.
    """

    name: str
    version: int = 1
    output_type: type[Out]
    in_place: bool = False
    all_ranks: bool = False
    # ``FunctionStage`` sets name and output_type per instance, not on the class.
    _configured_in_init: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401 # why: cooperative super call
        """Reject a concrete subclass that does not define ``name`` and ``output_type``."""
        super().__init_subclass__(**kwargs)
        if cls._configured_in_init or getattr(cls.execute, '__isabstractmethod__', False):
            return
        missing = [
            attribute for attribute in ('name', 'output_type') if not hasattr(cls, attribute)
        ]
        if missing:
            message = f'stage class {cls.__qualname__} must define {" and ".join(missing)}'
            raise TypeError(message)

    @abstractmethod
    # why: each stage declares its own inputs
    def execute(self, **arguments: Any) -> Out:  # noqa: ANN401
        """Do the work and return the artifact.

        Args:
            **arguments: Everything the caller passed, including ``run`` and,
                if given, ``name``.

        Returns:
            The result, an instance of ``output_type``.
        """

    # why: each stage declares its own inputs
    def __call__(self, **arguments: Any) -> Out:  # noqa: ANN401
        """Run the stage instance, or load its cached result. Never override this.

        Args:
            **arguments: Keyword arguments; ``run`` is required, ``name`` is the
                optional instance name, everything else is an input.

        Returns:
            The artifact, or an ``UnavailableArtifact`` if the instance failed
            or was skipped (or during a dry run).
        """
        # A failed, skipped or dry-run instance returns the UnavailableArtifact sentinel
        # instead of an Out, hence the cast.
        return cast('Out', self._invoke(arguments=arguments))

    def setting_annotations(self) -> Mapping[str, object]:
        """Return parameter annotations used to find ``NotFingerprinted`` settings."""
        return get_type_hints(type(self).execute, include_extras=True)

    def _invoke(self, *, arguments: dict[str, object]) -> Artifact:
        run = arguments.get('run')
        if not isinstance(run, Run):
            message = f'stage {self.name!r} must be called with run=<tslab.Run>'
            raise TypeError(message)
        if not is_writer_process() and not self.all_ranks:
            _LOGGER.info(
                'rank %s leaves before %s: only rank 0 runs stages after training',
                os.environ.get('SLURM_PROCID', os.environ.get('LOCAL_RANK', '?')),
                self.name,
            )
            raise SystemExit(0)
        stage_name = validate_stage_name(name=self.name)
        raw_name = arguments.get('name')
        if raw_name is not None and not isinstance(raw_name, str | tuple):
            message = f'name must be str or tuple[str, ...], got {type(raw_name).__name__}'
            raise TypeError(message)
        name_parts = normalize_instance_name(name=raw_name)
        instance_id = format_instance_id(stage_name=stage_name, name_parts=name_parts)
        called_single_rank = run.note_stage_call(all_ranks=self.all_ranks)
        if called_single_rank and self.all_ranks and slurm_task_count() > 1:
            # why: other ranks already exited, so training now would hang waiting for DDP peers
            message = (
                f'{instance_id}: training stages must come before other stages in a multi-GPU '
                'job; rank 1+ already left'
            )
            raise RuntimeError(message)
        # 1. Claim the name (duplicate and nesting checks).
        run.claim_instance(stage_name=stage_name, name_parts=name_parts)
        directory = run.instance_directory(stage_name=stage_name, instance_name=name_parts)
        inputs = {
            key: value
            for key, value in arguments.items()
            if key not in _RESERVED_ARGUMENTS and isinstance(value, Artifact)
        }
        # 2. Skip when an input is unavailable (failed or skipped upstream).
        blocked = _blocking_input(inputs=inputs)
        if blocked is not None and not run.options.dry_run:
            reason = f'input {blocked.unavailable_instance_id} {blocked.unavailable_reason}'
            _LOGGER.warning('skip %s: %s', instance_id, reason)
            run.record_instance(
                instance_id=instance_id,
                record=InstanceRecord(state=InstanceState.SKIPPED, reason=reason),
            )
            return UnavailableArtifact(
                instance_id=instance_id, reason=UnavailableReason.SKIPPED, detail=reason
            )
        # 3. Fingerprint.
        if blocked is not None or _has_unknown_dry_run_input(inputs=inputs):
            run.record_dry_run_action(instance_id=instance_id, action='run (upstream changes)')
            return UnavailableArtifact(
                instance_id=instance_id, reason=UnavailableReason.DRY_RUN, detail='planned to run'
            )
        settings = self._canonical_settings(arguments=arguments, instance_id=instance_id)
        fingerprint = compute_fingerprint(
            document={
                'stage': stage_name,
                'stage_version': self.version,
                'tslab_fingerprint_version': FINGERPRINT_FORMAT_VERSION,
                'seed': run.seed,
                'settings': settings,
                'inputs': {key: value.tslab_fingerprint for key, value in inputs.items()},
            }
        )
        # 4. Decide: load or run.
        invalidation = run.invalidation_reason(instance_id=instance_id)
        record = run.instance_record(instance_id=instance_id)
        stale_reason = invalidation or _stale_reason(
            record=record, fingerprint=fingerprint, directory=directory, inputs=inputs
        )
        if run.options.dry_run:
            return _plan_dry_run(
                run=run, instance_id=instance_id, directory=directory, stale_reason=stale_reason
            )
        agreement = (
            next((value for value in arguments.values() if isinstance(value, RankAgreement)), None)
            if self.all_ranks
            else None
        )
        agreement_key = (
            f'{next(_AGREEMENT_SEQUENCE)}:{instance_id}' if agreement is not None else instance_id
        )
        loaded = self._cached_result(
            directory=directory,
            instance_id=instance_id,
            agreement_key=agreement_key,
            stale_reason=stale_reason,
            agreement=agreement,
        )
        if loaded is not None:
            return loaded
        if stale_reason is not None:
            _LOGGER.info('run %s (%s)', instance_id, stale_reason)
        # 5. Execute, save, record.
        execution = _Execution(
            run=run,
            instance_id=instance_id,
            directory=directory,
            fingerprint=fingerprint,
            previous_fingerprint=None if record is None else record.fingerprint,
            agreement_key=agreement_key,
            settings=settings,
            inputs=inputs,
            agreement=agreement,
        )
        return self._execute_and_save(
            execution=execution,
            arguments=arguments,
            resume=self.in_place
            and invalidation is None
            and _can_resume(record=record, fingerprint=fingerprint),
        )

    def _cached_result(
        self,
        *,
        directory: Path,
        instance_id: str,
        agreement_key: str,
        stale_reason: str | None,
        agreement: RankAgreement | None,
    ) -> Artifact | None:
        loaded = (
            self._try_load(directory=directory, instance_id=instance_id)
            if stale_reason is None and (agreement is None or is_writer_process())
            else None
        )
        if agreement is None:
            return loaded
        use_cache = agreement.agree_across_ranks(
            key=f'{agreement_key}:cache', value=stale_reason is None and loaded is not None
        )
        if use_cache and loaded is None:
            loaded = self._try_load(directory=directory, instance_id=instance_id)
            if loaded is None:
                message = f'{instance_id}: rank 0 chose the cache but this rank cannot load it'
                raise RuntimeError(message)
        return loaded if use_cache else None

    def _canonical_settings(
        self, *, arguments: Mapping[str, object], instance_id: str
    ) -> dict[str, object]:
        annotations = self.setting_annotations()
        settings: dict[str, object] = {}
        for key, value in arguments.items():
            if key in _RESERVED_ARGUMENTS or isinstance(value, Artifact):
                continue
            if is_not_fingerprinted(annotation=annotations.get(key)):
                continue
            try:
                settings[key] = canonicalize(value=value)
            except TypeError as error:
                message = f'{instance_id}: cannot fingerprint argument {key!r}: {error}'
                raise TypeError(message) from error
        return settings

    def _try_load(self, *, directory: Path, instance_id: str) -> Artifact | None:
        try:
            artifact = self.output_type.load(directory=directory)
            identity = _identity_from_meta(directory=directory, instance_id=instance_id)
        except Exception:  # any load failure means "recompute", never "crash"
            _LOGGER.warning('cannot load %s; running it again', instance_id, exc_info=True)
            return None
        bind_identity(artifact=artifact, identity=identity)
        _LOGGER.info('load %s', instance_id)
        return artifact

    def _execute_and_save(
        self, *, execution: _Execution, arguments: dict[str, object], resume: bool
    ) -> Artifact:
        run, instance_id, directory = execution.run, execution.instance_id, execution.directory
        fingerprint = execution.fingerprint
        started_at = utc_now_iso()
        start = time.perf_counter()
        prepared = False
        run.record_instance(
            instance_id=instance_id,
            record=InstanceRecord(
                state=InstanceState.RUNNING, fingerprint=fingerprint, started_at=started_at
            ),
        )
        try:
            run.begin_execution()
            work_directory = self._prepare_work_directory(
                run=run, directory=directory, resume=resume, instance_id=instance_id
            )
            prepared = True
            if execution.agreement is not None:
                execution.agreement.agree_across_ranks(
                    key=f'{execution.agreement_key}:prepared', value=True
                )
            artifact = self.execute(**arguments)
            if not isinstance(artifact, self.output_type):
                message = (
                    f'stage {self.name!r} must return {self.output_type.__name__}, '
                    f'got {type(artifact).__name__}'
                )
                raise TypeError(message)  # noqa: TRY301 # why: handled as a stage failure below
            execution_id = uuid.uuid4().hex
            seconds = time.perf_counter() - start
            if run.writes_files:
                artifact.save(directory=work_directory)
                atomic_write_json(
                    path=work_directory / META_FILE,
                    document=_instance_meta_document(
                        stage=self,
                        execution=execution,
                        artifact=artifact,
                        execution_id=execution_id,
                        started_at=started_at,
                        seconds=seconds,
                    ),
                )
                self._publish(work_directory=work_directory, directory=directory)
        except JobTimeLimitSignal as error:
            failure_fingerprint = execution.fingerprint
            if not prepared and not resume:
                failure_fingerprint = execution.previous_fingerprint
                if failure_fingerprint == execution.fingerprint:
                    failure_fingerprint = None
            self._record_failure(
                execution=execution,
                error=error,
                fingerprint=failure_fingerprint,
                started_at=started_at,
                seconds=time.perf_counter() - start,
            )
            raise
        # why: any stage error is recorded; fail_fast re-raises
        except Exception as error:  # noqa: BLE001
            return self._record_failure(
                execution=execution,
                error=error,
                fingerprint=execution.fingerprint,
                started_at=started_at,
                seconds=time.perf_counter() - start,
            )
        finally:
            run.end_execution()
        bind_identity(
            artifact=artifact,
            identity=ArtifactIdentity(
                fingerprint=fingerprint, instance_id=instance_id, execution_id=execution_id
            ),
        )
        run.record_instance(
            instance_id=instance_id,
            record=InstanceRecord(
                state=InstanceState.DONE,
                fingerprint=fingerprint,
                started_at=started_at,
                finished_at=utc_now_iso(),
                seconds=seconds,
            ),
        )
        run.log_metrics(metrics={f'time/{instance_id.replace("[", "/").rstrip("]")}': seconds})
        _LOGGER.info('done %s in %.1fs', instance_id, seconds)
        return artifact

    def _publish(self, *, work_directory: Path, directory: Path) -> None:
        """Make a fully written result visible: drop the marker or swap the folder in."""
        if self.in_place:
            (directory / _INCOMPLETE_MARKER).unlink(missing_ok=True)
        else:
            replace_directory(source=work_directory, destination=directory)

    def _work_directory(self, *, directory: Path) -> Path:
        """Return where ``execute`` writes: the final folder, or a temporary sibling."""
        if self.in_place:
            return directory
        return directory.with_name(f'{directory.name}.tmp-{os.getpid()}')

    def _prepare_work_directory(
        self, *, run: Run, directory: Path, resume: bool, instance_id: str
    ) -> Path:
        if not run.writes_files:
            return directory
        for leftover in directory.parent.glob(f'{directory.name}.tmp-*'):
            shutil.rmtree(leftover, ignore_errors=True)
        work_directory = self._work_directory(directory=directory)
        if self.in_place:
            if resume:
                _LOGGER.info('resume %s in %s', instance_id, directory.as_posix())
                (directory / ERROR_FILE).unlink(missing_ok=True)
            elif directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / _INCOMPLETE_MARKER).touch()
            return directory
        work_directory.mkdir(parents=True)
        return work_directory

    def _record_failure(
        self,
        *,
        execution: _Execution,
        error: Exception | JobTimeLimitSignal,
        fingerprint: str | None,
        started_at: str,
        seconds: float,
    ) -> Artifact:
        run, instance_id, directory = execution.run, execution.instance_id, execution.directory
        error_line = describe_error(error=error)
        traceback_file: str | None = None
        if run.writes_files:
            if not self.in_place:
                shutil.rmtree(self._work_directory(directory=directory), ignore_errors=True)
            # A result from an earlier successful execution stays in the folder: it is
            # replaced only when a new execution succeeds. ``error.txt`` lands beside it.
            error_path = directory / ERROR_FILE
            try:
                directory.mkdir(parents=True, exist_ok=True)
                atomic_write_text(path=error_path, text=''.join(traceback.format_exception(error)))
            except OSError:
                _LOGGER.warning('cannot write %s', error_path.as_posix(), exc_info=True)
            else:
                traceback_file = error_path.relative_to(run.directory).as_posix()
        run.record_instance(
            instance_id=instance_id,
            record=InstanceRecord(
                state=InstanceState.FAILED,
                fingerprint=fingerprint,
                started_at=started_at,
                finished_at=utc_now_iso(),
                seconds=seconds,
                error=error_line,
                traceback_file=traceback_file,
            ),
        )
        if run.options.fail_fast or isinstance(error, JobTimeLimitSignal):
            raise error
        _LOGGER.error('failed %s: %s', instance_id, error_line, exc_info=error)
        return UnavailableArtifact(
            instance_id=instance_id, reason=UnavailableReason.FAILED, detail=error_line
        )


def _plan_dry_run(
    *, run: Run, instance_id: str, directory: Path, stale_reason: str | None
) -> UnavailableArtifact:
    """Record what a dry run would do and return a placeholder carrying the cached identity."""
    action = 'load' if stale_reason is None else f'run ({stale_reason})'
    run.record_dry_run_action(instance_id=instance_id, action=action)
    placeholder = UnavailableArtifact(
        instance_id=instance_id, reason=UnavailableReason.DRY_RUN, detail=action
    )
    if stale_reason is None:
        try:
            identity = _identity_from_meta(directory=directory, instance_id=instance_id)
        except StatusFileError:
            _LOGGER.warning(
                'cannot read result metadata of %s; its inputs stay unknown', instance_id
            )
        else:
            bind_identity(artifact=placeholder, identity=identity)
    return placeholder


def _instance_meta_document(  # noqa: PLR0913 # why: one keyword per recorded fact
    *,
    stage: Stage[Any],
    execution: _Execution,
    artifact: Artifact,
    execution_id: str,
    started_at: str,
    seconds: float,
) -> dict[str, object]:
    """Return the content of the ``meta.json`` of one executed instance."""
    return {
        'format_version': INSTANCE_META_FORMAT_VERSION,
        'instance_id': execution.instance_id,
        'stage_version': stage.version,
        'fingerprint': execution.fingerprint,
        'execution_id': execution_id,
        'artifact_type': qualified_name(cls=type(artifact)),
        'settings': execution.settings,
        'inputs': _describe_inputs(inputs=execution.inputs),
        'external_inputs': _describe_external_inputs(inputs=execution.inputs),
        # The code and package versions that made this result. A resumed run
        # rewrites the run meta.json, so only here do they stay tied to the result.
        'environment': execution.run.environment(),
        'started_at': started_at,
        'finished_at': utc_now_iso(),
        'seconds': seconds,
    }


def _blocking_input(*, inputs: Mapping[str, Artifact]) -> UnavailableArtifact | None:
    for value in inputs.values():
        if isinstance(value, UnavailableArtifact) and value.unavailable_reason in {
            UnavailableReason.FAILED,
            UnavailableReason.SKIPPED,
        }:
            return value
    return None


def _has_unknown_dry_run_input(*, inputs: Mapping[str, Artifact]) -> bool:
    return any(
        isinstance(value, UnavailableArtifact) and value.tslab_identity is None
        for value in inputs.values()
    )


def _stale_reason(
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
    meta_path = directory / META_FILE
    if not meta_path.exists():
        return 'result folder missing'
    document = _read_instance_meta(path=meta_path)
    if document is None:
        reason = 'cannot read result metadata'
    elif document.get('format_version') != INSTANCE_META_FORMAT_VERSION:
        reason = 'unsupported result metadata format'
    else:
        # TSLab writes ``inputs`` as a mapping of dicts; the cast names that shape.
        recorded_inputs = cast('dict[str, dict[str, str | None]]', document.get('inputs') or {})
        reason = None
        for key, value in inputs.items():
            identity = value.tslab_identity
            if identity is None or identity.instance_id is None:
                continue
            if recorded_inputs.get(key, {}).get('execution_id') != identity.execution_id:
                reason = f'input {identity.instance_id} was executed again'
                break
    return reason


def _can_resume(*, record: InstanceRecord | None, fingerprint: str) -> bool:
    return (
        record is not None
        and record.state in {InstanceState.RUNNING, InstanceState.FAILED}
        and record.fingerprint == fingerprint
    )


def _identity_from_meta(*, directory: Path, instance_id: str) -> ArtifactIdentity:
    document = _read_instance_meta(path=directory / META_FILE)
    fingerprint: object = None
    execution_id: object = None
    if document is not None and document.get('format_version') == INSTANCE_META_FORMAT_VERSION:
        fingerprint = document.get('fingerprint')
        execution_id = document.get('execution_id')
    if not isinstance(fingerprint, str) or not isinstance(execution_id, str):
        message = f'cannot read result metadata of {instance_id}'
        raise StatusFileError(message)
    return ArtifactIdentity(
        fingerprint=fingerprint, instance_id=instance_id, execution_id=execution_id
    )


def _read_instance_meta(*, path: Path) -> dict[str, object] | None:
    """Parse the ``meta.json`` of one instance; unreadable or non-object files give ``None``.

    A corrupt or partial file must never crash a run: the instance recomputes
    instead (see ``_stale_reason`` and ``_identity_from_meta``).
    """
    try:
        return read_json_object(path=path)
    except (OSError, JsonFileError):
        _LOGGER.warning('cannot read %s; treating the result as stale', path.as_posix())
        return None


def _describe_inputs(*, inputs: Mapping[str, Artifact]) -> dict[str, dict[str, str | None]]:
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


def _describe_external_inputs(*, inputs: Mapping[str, Artifact]) -> dict[str, dict[str, str]]:
    described: dict[str, dict[str, str]] = {}
    for key, value in inputs.items():
        identity = value.tslab_identity
        if identity is not None and identity.external_path is not None:
            described[key] = {
                'path': identity.external_path.as_posix(),
                'fingerprint': identity.fingerprint,
            }
    return described


class FunctionStage[**P, Out: Artifact](Stage[Out]):
    """A ``Stage`` built from a plain function by ``@stage``.

    Calls are type-checked against the signature of the wrapped function itself.
    """

    _configured_in_init: ClassVar[bool] = True

    def __init__(  # noqa: PLR0913 # why: one keyword for each stage setting
        self,
        *,
        function: Callable[P, Out],
        name: str,
        output: type[Out],
        version: int,
        in_place: bool,
        all_ranks: bool,
    ) -> None:
        """Wrap ``function``. Use ``@stage`` instead of calling this directly."""
        qualname = getattr(function, '__qualname__', repr(function))
        parameters = inspect.signature(function).parameters
        positional = [
            parameter.name
            for parameter in parameters.values()
            if parameter.kind not in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD}
        ]
        if positional:
            message = (
                f'stage function {qualname} must take keyword-only parameters '
                f'(put * first); positional: {positional}'
            )
            raise TypeError(message)
        if 'run' not in parameters:
            message = f'stage function {qualname} must take a run parameter'
            raise TypeError(message)
        self._function = function
        self._qualname = qualname
        self._accepts_name = 'name' in parameters
        self.name = validate_stage_name(name=name)
        self.version = version
        self.output_type = output
        self.in_place = in_place
        self.all_ranks = all_ranks
        self.__doc__ = function.__doc__
        self.__wrapped__ = function

    @override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Out:
        """Run the stage instance or load its cached result (see ``Stage``)."""
        if args:
            message = f'stage {self.name!r} takes keyword arguments only'
            raise TypeError(message)
        if 'name' in kwargs and not self._accepts_name:
            message = (
                f'stage {self.name!r} is single-instance; to allow name=, add a parameter '
                f'`name: tslab.InstanceName | None = None` to {self._qualname}'
            )
            raise TypeError(message)
        # A failed, skipped or dry-run instance returns the UnavailableArtifact sentinel
        # instead of an Out, hence the cast.
        return cast('Out', self._invoke(arguments=dict(kwargs)))

    @override
    def execute(self, **arguments: Any) -> Out:  # why: forwards to the wrapped function
        """Call the wrapped function."""
        # A ParamSpec-typed callable cannot be called with a plain dict; the dict holds
        # exactly the keyword arguments __call__ received, so the call matches P.
        return cast('Callable[..., Out]', self._function)(**arguments)

    @override
    def setting_annotations(self) -> Mapping[str, object]:
        """Return the annotations of the wrapped function."""
        return get_type_hints(self._function, include_extras=True)


def stage[**P, Out: Artifact](
    *,
    name: str,
    output: type[Out],
    version: int = 1,
    in_place: bool = False,
    all_ranks: bool = False,
) -> Callable[[Callable[P, Out]], FunctionStage[P, Out]]:
    """Turn a function into a ``Stage``.

    The function must take keyword-only parameters, including ``run``. To call
    the stage several times in one run, add ``name: tslab.InstanceName | None = None``;
    without it the stage is single-instance.

    Example::

        @tslab.stage(name='transform', output=tslab.Tensors)
        def transform(*, run: tslab.Run, name: tslab.InstanceName | None = None,
                   representations: tslab.Tensors, epsilon: float) -> tslab.Tensors:
            ...

    Args:
        name: Stage name (folder name), matching ``[A-Za-z0-9._-]+``.
        output: The ``Artifact`` subclass the function returns.
        version: Bump to invalidate cached results after a code change.
        in_place: Write directly into the final folder and resume unfinished
            work (see ``Stage.in_place``).
        all_ranks: Run on every rank in a multi-rank job (for DDP training).

    Returns:
        A decorator producing a ``FunctionStage``.
    """

    def decorate(function: Callable[P, Out]) -> FunctionStage[P, Out]:
        return FunctionStage(
            function=function,
            name=name,
            output=output,
            version=version,
            in_place=in_place,
            all_ranks=all_ranks,
        )

    return decorate
