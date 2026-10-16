"""``Run``: one execution of a user script with one config, bound to one folder.

``Run`` is a context manager (Unit of Work pattern). Entering it prepares the
folder, the lock, logging, seeding, metadata and trackers. Leaving it always
records the final state, finishes the trackers and releases the lock, even
when the user's code raised an exception.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from types import TracebackType
from typing import Literal, Self

from tslab.config import config_to_data, config_to_yaml, save_config
from tslab.fingerprint import canonicalize, compute_fingerprint
from tslab.interfaces.tracker import RunStatus, Tracker
from tslab.meta import collect_git_state, collect_versions, hostname, META_FORMAT_VERSION
from tslab.paths import (
    atomic_write_text,
    InstanceName,
    normalize_instance_name,
    require_path,
    validate_relative_path,
    validate_stage_name,
)
from tslab.seeding import seed_everything
from tslab.status import (
    InstanceRecord,
    InstanceState,
    read_status,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)

__all__ = ['Run', 'RunLockedError', 'RunOptions', 'is_run_active', 'is_writer_process']

_LOGGER = logging.getLogger(__name__)
_RANK_VARIABLES = ('RANK', 'SLURM_PROCID', 'LOCAL_RANK')
_LOCK_FILE = '.lock'
_HEARTBEAT_SECONDS = 60.0
_STALE_AFTER_SECONDS = 600.0
_STATUS_FILE = 'status.json'
_META_FILE = 'meta.json'
_CONFIG_FILE = 'config.yaml'
_LOG_FILE = Path('logs') / 'run.log'
_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


@dataclass(frozen=True)
class RunOptions:
    """Command-line controllable behaviour of a run (``--run.*`` flags via tyro).

    Attributes:
        force_all: Treat every cached result as invalid and run everything again.
        rerun_from: A stage name. Every instance of this stage, and of every
            stage called after it, runs again.
        rerun: Instance ids to run again, for example transform[sma] or transform[h24/sma].
        fail_fast: Re-raise the first stage error instead of continuing.
        dry_run: Print the plan (load, run, skip) without executing or writing anything.
        log_level: Level of TSLab's log output.
    """

    force_all: bool = False
    rerun_from: str | None = None
    rerun: tuple[str, ...] = ()
    fail_fast: bool = False
    dry_run: bool = False
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'

    def fingerprint_settings(self) -> None:
        """Run options never change results, so they never enter a fingerprint."""
        return


class RunLockedError(RuntimeError):
    """Raised when another live process owns the run folder."""


def is_writer_process() -> bool:
    """Return ``True`` unless this process is a non-zero rank of a distributed job.

    Every rank variable that is set must be ``0``. Checking all of them covers
    both ``srun`` tasks (``SLURM_PROCID``) and processes started by Lightning
    inside one SLURM task (``LOCAL_RANK``).
    """
    return all(os.environ.get(variable, '0') == '0' for variable in _RANK_VARIABLES)


class Run:
    """One execution of a user script with one config, bound to one folder.

    Example::

        with tslab.Run(config=config, root=Path('runs'), relative_path=Path('a/b')) as run:
            checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=data)
    """

    def __init__(  # noqa: PLR0913 # why: keyword-only public API; each option is documented below
        self,
        *,
        config: object,
        root: Path,
        relative_path: Path,
        trackers: Sequence[Tracker] = (),
        options: RunOptions | None = None,
        seed: int | None = None,
    ) -> None:
        """Validate arguments. Nothing is written until the ``with`` block is entered.

        Args:
            config: The run's configuration, a dataclass instance. Saved as
                ``config.yaml`` and hashed into ``meta.json``.
            root: Folder that holds all runs.
            relative_path: Relative folder of this run under ``root``; several
                parts create hierarchy, for example ``Path('ts2vec/Coffee/seed_42')``.
            trackers: Experiment trackers; started on enter, finished on exit.
            options: Run behaviour; defaults to ``config.run`` if it is a
                ``RunOptions``, otherwise to ``RunOptions()``.
            seed: Seed for every random generator; defaults to ``config.seed``,
                otherwise ``0`` with a warning.

        Raises:
            TypeError: If ``config`` is not a dataclass instance or a path is not a ``Path``.
            ValueError: If ``relative_path`` is absolute, empty or has invalid parts.
        """
        if not dataclasses.is_dataclass(config) or isinstance(config, type):
            message = f'config must be a dataclass instance, got {type(config).__name__}'
            raise TypeError(message)
        require_path(value=root, parameter='root')
        validate_relative_path(path=relative_path, parameter='relative_path')
        self._config = config
        self._relative_path = relative_path
        self._directory = root / relative_path
        self._trackers = tuple(trackers)
        self._options = options if options is not None else _options_from_config(config=config)
        if self._options.rerun_from is not None:
            validate_stage_name(name=self._options.rerun_from)
        self._seed = seed if seed is not None else _seed_from_config(config=config)
        self._writer = is_writer_process()
        self._snapshot = RunStatusSnapshot(run_state=RunState.RUNNING, updated_at=utc_now_iso())
        self._claimed: dict[str, list[tuple[str, ...]]] = {}
        self._rerun_from_reached = False
        self._matched_rerun_ids: set[str] = set()
        self._dry_run_plan: list[tuple[str, str]] = []
        self._log_handlers: list[logging.Handler] = []
        self._previous_root_level: int | None = None
        self._lock_held = False
        self._heartbeat: _Heartbeat | None = None
        self._entered = False
        self._meta: dict[str, object] = {}

    @property
    def directory(self) -> Path:
        """Return the run folder, ``root / relative_path``."""
        return self._directory

    @property
    def trackers(self) -> tuple[Tracker, ...]:
        """Return the run's trackers."""
        return self._trackers

    @property
    def options(self) -> RunOptions:
        """Return the run options."""
        return self._options

    @property
    def seed(self) -> int:
        """Return the run seed. It is part of every stage fingerprint."""
        return self._seed

    @property
    def writes_files(self) -> bool:
        """Return ``True`` if this process may write (rank 0 and not a dry run)."""
        return self._writer and not self._options.dry_run

    def instance_directory(self, *, stage_name: str, instance_name: InstanceName | None) -> Path:
        """Return the folder of a stage instance.

        Args:
            stage_name: The stage name.
            instance_name: The instance name; ``None`` means ``"default"``.

        Returns:
            ``directory / stage_name / <name parts>``.
        """
        parts = normalize_instance_name(name=instance_name)
        return self._directory.joinpath(validate_stage_name(name=stage_name), *parts)

    def status(self) -> RunStatusSnapshot:
        """Return the current run state and instance records."""
        return self._snapshot

    def log_metrics(self, *, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Send metrics to every tracker; tracker errors become warnings.

        Args:
            metrics: Name to value.
            step: Optional step for time series of metrics.
        """
        for tracker in self._trackers:
            _call_tracker(
                tracker=tracker,
                action='log_metrics',
                call=lambda tracker=tracker: tracker.log_metrics(metrics=metrics, step=step),
            )

    # The methods below form the contract between Run and Stage. They are
    # public so Stage subclasses outside this package can use them, but user
    # scripts normally never call them.

    def claim_instance(self, *, stage_name: str, name_parts: tuple[str, ...]) -> None:
        """Reserve an instance name for this process; reject duplicates and nesting conflicts.

        Two names conflict if one is a prefix of the other, because the shorter
        name's folder would contain the longer name's folder, and replacing it
        would delete the other result.

        Args:
            stage_name: The stage name.
            name_parts: Validated instance name parts.

        Raises:
            ValueError: If the name was already used in this process or conflicts.
        """
        claimed = self._claimed.setdefault(stage_name, [])
        if name_parts in claimed:
            if name_parts == ('default',):
                message = (
                    f'stage {stage_name!r} called twice without name=; '
                    f'pass a distinct name= for each instance'
                )
            else:
                message = f'stage {stage_name!r} called twice with name {"/".join(name_parts)!r}'
            raise ValueError(message)
        recorded = [
            tuple(instance_id[len(stage_name) + 1 : -1].split('/'))
            for instance_id in self._snapshot.instances
            if instance_id.startswith(f'{stage_name}[')
        ]
        for other in [*claimed, *recorded]:
            shorter, longer = sorted((name_parts, other), key=len)
            if shorter != longer and longer[: len(shorter)] == shorter:
                message = (
                    f'instance name {"/".join(name_parts)!r} of stage {stage_name!r} conflicts '
                    f'with '
                    f'{"/".join(other)!r}: one would be stored inside the other'
                )
                raise ValueError(message)
        claimed.append(name_parts)

    def instance_record(self, *, instance_id: str) -> InstanceRecord | None:
        """Return the stored record of an instance, or ``None``."""
        return self._snapshot.instances.get(instance_id)

    def record_instance(self, *, instance_id: str, record: InstanceRecord) -> None:
        """Store an instance record and persist ``status.json`` when allowed."""
        self._snapshot = self._snapshot.with_instance(instance_id=instance_id, record=record)
        self._persist_status()

    def invalidation_reason(self, *, stage_name: str, instance_id: str) -> str | None:
        """Return why ``RunOptions`` force this instance to run, or ``None``.

        Calling this for ``rerun_from``'s stage marks it as reached, so every
        later call (of any stage) is invalidated too.

        Args:
            stage_name: The stage name.
            instance_id: The instance id.

        Returns:
            A short reason for the log, or ``None`` if the cache may be used.
        """
        if stage_name == self._options.rerun_from:
            self._rerun_from_reached = True
        if self._options.force_all:
            return '--run.force-all'
        if instance_id in self._options.rerun:
            self._matched_rerun_ids.add(instance_id)
            return '--run.rerun'
        if self._rerun_from_reached:
            return f'--run.rerun-from {self._options.rerun_from}'
        return None

    def record_dry_run_action(self, *, instance_id: str, action: str) -> None:
        """Record and print what a dry run would do for one instance."""
        self._dry_run_plan.append((instance_id, action))
        _LOGGER.info('dry run: %s -> %s', instance_id, action)

    def __enter__(self) -> Self:
        """Prepare the run. See the class docstring for what happens.

        Raises:
            RuntimeError: If the run was already entered once.
            RunLockedError: If another live process owns the folder.
        """
        if self._entered:
            message = 'a Run can be entered only once; create a new Run'
            raise RuntimeError(message)
        self._entered = True
        try:
            self._configure_logging()
            seed_everything(seed=self._seed)
            if self.writes_files:
                self._directory.mkdir(parents=True, exist_ok=True)
                _acquire_lock(path=self._directory / _LOCK_FILE)
                self._lock_held = True
                self._heartbeat = _Heartbeat(path=self._directory / _LOCK_FILE)
                self._attach_file_log()
            existing = read_status(path=self._directory / _STATUS_FILE)
            if existing is not None:
                self._snapshot = existing.with_run_state(run_state=RunState.RUNNING)
            if self._options.dry_run:
                _LOGGER.info('dry run of %s; nothing is executed or written', self._directory)
                _LOGGER.info('config:\n%s', config_to_yaml(config=self._config))
            elif self.writes_files:
                save_config(config=self._config, path=self._directory / _CONFIG_FILE)
                self._write_meta(finished_at=None)
                self._persist_status()
                self._start_trackers()
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Record the final state, finish trackers, print the summary, release the lock.

        Returns:
            ``False``: exceptions always propagate.
        """
        run_state = _final_run_state(exception=exc_value)
        try:
            self._snapshot = self._snapshot.with_run_state(run_state=run_state)
            self._persist_status()
            if self.writes_files:
                self._write_meta(finished_at=utc_now_iso())
                for tracker in self._trackers:
                    _call_tracker(
                        tracker=tracker,
                        action='finish',
                        call=lambda tracker=tracker: tracker.finish(
                            status=RunStatus(run_state.value)
                        ),
                    )
            self._log_summary(run_state=run_state)
        finally:
            self._release()
        return False

    def _persist_status(self) -> None:
        if self.writes_files:
            write_status(path=self._directory / _STATUS_FILE, snapshot=self._snapshot)

    def _configure_logging(self) -> None:
        level = logging.getLevelNamesMapping()[self._options.log_level]
        root_logger = logging.getLogger()
        self._previous_root_level = root_logger.level
        root_logger.setLevel(min(root_logger.level or logging.WARNING, level))
        logging.getLogger('tslab').setLevel(level)
        if not root_logger.handlers:
            console = logging.StreamHandler(stream=sys.stderr)
            console.setFormatter(logging.Formatter(fmt=_LOG_FORMAT))
            self._add_handler(handler=console)

    def _attach_file_log(self) -> None:
        log_path = self._directory / _LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT))
        self._add_handler(handler=file_handler)

    def _add_handler(self, *, handler: logging.Handler) -> None:
        logging.getLogger().addHandler(handler)
        self._log_handlers.append(handler)

    def _write_meta(self, *, finished_at: str | None) -> None:
        if not self._meta:
            canonical_config = canonicalize(value=self._config)
            without_seed = (
                {key: value for key, value in canonical_config.items() if key != 'seed'}
                if isinstance(canonical_config, dict)
                else canonical_config
            )
            self._meta = {
                'format_version': META_FORMAT_VERSION,
                'relative_path': self._relative_path.as_posix(),
                'seed': self._seed,
                'config_hash': compute_fingerprint(document={'config': canonical_config}),
                'config_hash_without_seed': compute_fingerprint(document={'config': without_seed}),
                'versions': collect_versions(),
                'git': collect_git_state(directory=Path.cwd()),
                'hostname': hostname(),
                'started_at': utc_now_iso(),
            }
        self._meta['finished_at'] = finished_at
        atomic_write_text(
            path=self._directory / _META_FILE,
            text=json.dumps(self._meta, indent=2, sort_keys=True) + '\n',
        )

    def _start_trackers(self) -> None:
        # The config.yaml view, not the fingerprint view: numbers must stay numbers.
        params = _flatten(value=config_to_data(config=self._config), prefix='')
        for tracker in self._trackers:
            _call_tracker(
                tracker=tracker,
                action='start',
                call=lambda tracker=tracker: tracker.start(
                    run_directory=self._directory, run_name=self._relative_path.as_posix()
                ),
            )
            _call_tracker(
                tracker=tracker,
                action='log_params',
                call=lambda tracker=tracker: tracker.log_params(params=params),
            )

    def _log_summary(self, *, run_state: RunState) -> None:
        if self._options.dry_run:
            _LOGGER.info('dry run plan:')
            for instance_id, action in self._dry_run_plan:
                _LOGGER.info('  %-40s %s', instance_id, action)
            return
        counts = Counter(record.state for record in self._snapshot.instances.values())
        _LOGGER.info(
            'run %s: %s',
            run_state.value,
            ', '.join(f'{counts[state]} {state.value}' for state in InstanceState),
        )
        for instance_id, record in self._snapshot.instances.items():
            if record.state is InstanceState.FAILED:
                _LOGGER.error('  failed  %s: %s', instance_id, record.error)
            elif record.state is InstanceState.SKIPPED:
                _LOGGER.warning('  skipped %s: %s', instance_id, record.reason)
        for unmatched in sorted(set(self._options.rerun) - self._matched_rerun_ids):
            _LOGGER.warning('--run.rerun %s matched no instance called in this run', unmatched)
        if self._options.rerun_from is not None and not self._rerun_from_reached:
            _LOGGER.warning(
                '--run.rerun-from %s: no stage with this name was called', self._options.rerun_from
            )

    def _release(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        if self._lock_held:
            (self._directory / _LOCK_FILE).unlink(missing_ok=True)
            self._lock_held = False
        root_logger = logging.getLogger()
        for handler in self._log_handlers:
            root_logger.removeHandler(handler)
            handler.close()
        self._log_handlers.clear()
        if self._previous_root_level is not None:
            root_logger.setLevel(self._previous_root_level)


def _options_from_config(*, config: object) -> RunOptions:
    options = getattr(config, 'run', None)
    return options if isinstance(options, RunOptions) else RunOptions()


def _seed_from_config(*, config: object) -> int:
    seed = getattr(config, 'seed', None)
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    _LOGGER.warning('config has no int field "seed" and no seed= was given; using seed 0')
    return 0


def _final_run_state(*, exception: BaseException | None) -> RunState:
    if exception is None:
        return RunState.FINISHED
    if isinstance(exception, SystemExit):
        return RunState.FINISHED if exception.code in {None, 0} else RunState.CRASHED
    if isinstance(exception, KeyboardInterrupt):
        return RunState.CRASHED
    return RunState.FAILED


def _call_tracker(*, tracker: Tracker, action: str, call: Callable[[], None]) -> None:
    try:
        call()
    except Exception:
        _LOGGER.warning(
            'tracker %s failed in %s; continuing without it for this call',
            type(tracker).__name__,
            action,
            exc_info=True,
        )


def _flatten(*, value: object, prefix: str) -> dict[str, object]:
    if isinstance(value, dict):
        flat: dict[str, object] = {}
        for key, item in value.items():
            flat.update(_flatten(value=item, prefix=f'{prefix}/{key}' if prefix else str(key)))
        return flat
    return {prefix: value}


@dataclass(frozen=True)
class _LockOwner:
    """Contents of the ``.lock`` file."""

    pid: int
    host: str


class _Heartbeat:
    """Touch the lock file every ``_HEARTBEAT_SECONDS`` from a daemon thread.

    The lock's modification time then tells any host whether its owner is
    still alive: a job killed by SLURM stops touching it.
    """

    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._beat, name='tslab-heartbeat', daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._stopped.wait(timeout=_HEARTBEAT_SECONDS):
            try:
                os.utime(self._path)
            except OSError as error:
                _LOGGER.warning('could not refresh %s: %s', self._path.as_posix(), error)

    def stop(self) -> None:
        """Stop beating and wait for the thread to end."""
        self._stopped.set()
        self._thread.join()


def is_run_active(*, directory: Path) -> bool:
    """Return whether a live process holds the lock of a run folder.

    A lock is live when its heartbeat is recent and, on this host, its process
    still exists. Another host's process cannot be checked, so there only the
    heartbeat counts.

    Args:
        directory: The run folder.
    """
    path = directory / _LOCK_FILE
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return not _lock_is_stale(owner=_read_lock_owner(path=path), age=age)


def _lock_is_stale(*, owner: _LockOwner | None, age: float) -> bool:
    if age > _STALE_AFTER_SECONDS:
        return True
    return owner is not None and owner.host == hostname() and not _pid_alive(pid=owner.pid)


def _acquire_lock(*, path: Path) -> None:
    """Create the lock file atomically, replacing it once if it is stale.

    ``O_CREAT | O_EXCL`` makes creation atomic: exactly one process succeeds.
    A lock is stale when its heartbeat stopped more than ``_STALE_AFTER_SECONDS``
    ago (any host), or when its process is dead (this host).

    Raises:
        RunLockedError: If a live process owns the lock.
    """
    payload = json.dumps({'pid': os.getpid(), 'host': hostname()})
    for _attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            owner = _read_lock_owner(path=path)
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if _lock_is_stale(owner=owner, age=age):
                _LOGGER.warning(
                    'removing stale lock %s left by %s (last heartbeat %.0f s ago)',
                    path.as_posix(),
                    owner or 'an unknown process',
                    age,
                )
                path.unlink(missing_ok=True)
                continue
            message = (
                f'{path.parent.as_posix()} is in use by {owner or "an unknown process"}; '
                f'if no such process exists, delete {path.as_posix()}'
            )
            raise RunLockedError(message) from None
        with os.fdopen(descriptor, mode='w', encoding='utf-8') as handle:
            handle.write(payload)
        return
    message = f'could not acquire {path.as_posix()}'
    raise RunLockedError(message)


def _read_lock_owner(*, path: Path) -> _LockOwner | None:
    """Return the lock owner, or ``None`` if the file is unreadable or malformed."""
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    pid, host = document.get('pid'), document.get('host')
    if not isinstance(pid, int) or not isinstance(host, str):
        return None
    return _LockOwner(pid=pid, host=host)


def _pid_alive(*, pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
