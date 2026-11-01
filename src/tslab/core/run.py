"""``Run``: one execution of a user script with one config, bound to one folder.

``Run`` is a context manager (Unit of Work pattern). Entering it prepares the
folder, the lock, logging, seeding, metadata and trackers. Leaving it always
records the final state, finishes the trackers and releases the lock, even
when the code of the user raised an exception.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import logging
from pathlib import Path
import signal
import sys
import threading
from types import FrameType, MappingProxyType, TracebackType
from typing import Literal, Self

from tslab.core.codec import config_to_data
from tslab.core.config import config_to_yaml, save_config
from tslab.core.files import atomic_write_json
from tslab.core.fingerprint import canonicalize, compute_fingerprint
from tslab.core.lock import acquire_lock, Heartbeat, release_lock
from tslab.core.meta import collect_git_state, collect_versions, hostname, META_FORMAT_VERSION
from tslab.core.paths import (
    CONFIG_FILE,
    InstanceName,
    LOCK_FILE,
    LOGS_DIRECTORY,
    META_FILE,
    normalize_instance_name,
    require_path,
    STATUS_FILE,
    validate_relative_path,
    validate_stage_name,
)
from tslab.core.ranks import is_writer_process
from tslab.core.seeding import seed_everything
from tslab.core.status import (
    format_instance_id,
    InstanceRecord,
    InstanceState,
    read_status,
    RunState,
    RunStatusSnapshot,
    utc_now_iso,
    write_status,
)
from tslab.core.tracker import LocalTracker, RunStatus, Tracker

__all__ = ['JobTimeLimitSignal', 'Run', 'RunOptions']

_LOGGER = logging.getLogger(__name__)
_SIGTERM_EXIT_CODE = 128 + signal.SIGTERM
_LOG_FILE = Path(LOGS_DIRECTORY) / 'run.log'
_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


@dataclass(frozen=True, kw_only=True)
class RunOptions:
    """Command-line controllable behavior of a run (``--run.*`` flags via tyro).

    Attributes:
        force_all: Treat every cached result as invalid and run everything again.
            This covers the instances the script calls; folders of instances it
            does not call are left alone.
        rerun_from: A stage name. Every instance of this stage, and of every
            stage called after it, runs again.
        rerun: Instance ids to run again, for example transform[sma] or transform[h24/sma].
        fail_fast: Re-raise the first stage error instead of continuing.
        dry_run: Print the plan (load, run, skip) without executing or writing anything.
        log_level: Level of the log output of TSLab.
    """

    force_all: bool = False
    rerun_from: str | None = None
    rerun: tuple[str, ...] = ()
    fail_fast: bool = False
    dry_run: bool = False
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'

    def fingerprint_settings(self) -> None:
        """Run options never change results, so they never enter a fingerprint.

        ``canonicalize`` tries ``Fingerprintable`` before the dataclass rule, so
        returning ``None`` here removes the options from every fingerprint.
        """
        return


class JobTimeLimitSignal(SystemExit):
    """The SLURM time limit is near; resubmit the run to resume."""


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
            config: The configuration of the run, a dataclass instance. Saved as
                ``config.yaml`` and hashed into ``meta.json``.
            root: Folder that holds all runs.
            relative_path: Relative folder of this run under ``root``; several
                parts create hierarchy, for example ``Path('ts2vec/Coffee/seed_42')``.
            trackers: Experiment trackers; started on enter, finished on exit.
                A ``LocalTracker`` (CSV and YAML files in the run folder) is added in
                front of them unless one of them already is a ``LocalTracker``.
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
        # The local tracker is always on: it is the record when remote trackers fail.
        has_local = any(isinstance(tracker, LocalTracker) for tracker in trackers)
        self._trackers = (*trackers,) if has_local else (LocalTracker(), *trackers)
        self._options = options if options is not None else _options_from_config(config=config)
        if self._options.rerun_from is not None:
            validate_stage_name(name=self._options.rerun_from)
        self._seed = seed if seed is not None else _seed_from_config(config=config)
        self._writer = is_writer_process()
        self._snapshot = RunStatusSnapshot(run_state=RunState.RUNNING, updated_at=utc_now_iso())
        self._claimed: dict[str, list[tuple[str, ...]]] = {}
        self._rerun_from_reached = False
        self._claimed_ids: set[str] = set()
        self._matched_rerun_ids: set[str] = set()
        self._dry_run_plan: list[tuple[str, str]] = []
        self._log_handlers: list[logging.Handler] = []
        self._previous_root_level: int | None = None
        self._previous_tslab_level: int | None = None
        self._previous_sigterm: signal.Handlers | None = None
        self._previous_sigusr2: (
            signal.Handlers | Callable[[int, FrameType | None], object] | int | None
        ) = None
        self._time_limit_signalled = False
        self._executing = False
        self._called_single_rank = False
        self._lock_nonce: str | None = None
        self._heartbeat: Heartbeat | None = None
        self._entered = False
        self._meta: dict[str, object] = {}
        self._environment: dict[str, object] | None = None

    @property
    def directory(self) -> Path:
        """Return the run folder, ``root / relative_path``."""
        return self._directory

    @property
    def trackers(self) -> tuple[Tracker, ...]:
        """Return the trackers of the run."""
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

    @property
    def time_limit_signalled(self) -> bool:
        """Whether this process received the SLURM time-limit signal."""
        return self._time_limit_signalled

    def begin_execution(self) -> None:
        """Mark a stage executing, or stop if the signal arrived between stages.

        Raises:
            JobTimeLimitSignal: If ``SIGUSR2`` arrived before this stage started.
        """
        self._executing = True
        if self._time_limit_signalled:
            message = 'SIGUSR2 before the SLURM time limit; resubmit to resume'
            raise JobTimeLimitSignal(message)

    def end_execution(self) -> None:
        """Mark a stage no longer executing."""
        self._executing = False

    def note_stage_call(self, *, all_ranks: bool) -> bool:
        """Return whether a single-rank stage was previously called in this run.

        Args:
            all_ranks: Whether the current stage runs on every rank.

        Returns:
            Whether an earlier stage runs only on rank zero.
        """
        previous = self._called_single_rank
        if not all_ranks:
            self._called_single_rank = True
        return previous

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

    def environment(self) -> dict[str, object]:
        """Return the package versions and git state of the code that runs now.

        Computed once per ``Run``. Stages record it in the ``meta.json`` of every
        instance they execute, so a result keeps the commit that produced it even
        after a later execution rewrites the ``meta.json`` of the run.

        Returns:
            Mapping with keys ``versions`` and ``git``. The git state is read from
            the folder of the main script, not from the working directory, because
            ``sbatch`` may start the script from elsewhere.
        """
        if self._environment is None:
            self._environment = {
                'versions': collect_versions(),
                'git': collect_git_state(directory=_project_directory()),
            }
        return self._environment

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

    def log_file(self, *, path: Path) -> None:
        """Send a file to every tracker; tracker errors become warnings.

        Args:
            path: The file to upload or copy.
        """
        for tracker in self._trackers:
            _call_tracker(
                tracker=tracker,
                action='log_file',
                call=lambda tracker=tracker: tracker.log_file(path=path),
            )

    # The methods below form the contract between Run and Stage. They are
    # public so Stage subclasses outside this package can use them, but user
    # scripts normally never call them.

    def claim_instance(self, *, stage_name: str, name_parts: tuple[str, ...]) -> None:
        """Reserve an instance name for this process; reject duplicates and nesting conflicts.

        Two names conflict if one is a prefix of the other, because the folder
        of the shorter name would contain the folder of the longer name, and
        replacing it would delete the other result.

        Args:
            stage_name: The stage name.
            name_parts: Validated instance name parts.

        Raises:
            ValueError: If the name was already used in this process or conflicts.
        """
        if stage_name == self._options.rerun_from:
            self._rerun_from_reached = True  # even if this instance is later skipped
        self._claimed_ids.add(format_instance_id(stage_name=stage_name, name_parts=name_parts))
        claimed = self._claimed.setdefault(stage_name, [])
        if name_parts in claimed:
            raise ValueError(_duplicate_message(stage_name=stage_name, name_parts=name_parts))
        recorded = [
            tuple(instance_id[len(stage_name) + 1 : -1].split('/'))
            for instance_id in self._snapshot.instances
            if instance_id.startswith(f'{stage_name}[')
        ]
        for other, is_recorded in [
            *((item, False) for item in claimed),
            *((item, True) for item in recorded),
        ]:
            shorter, longer = sorted((name_parts, other), key=len)
            if shorter != longer and longer[: len(shorter)] == shorter:
                raise ValueError(
                    _conflict_message(
                        stage_name=stage_name,
                        name_parts=name_parts,
                        other=other,
                        recorded_in=(self._directory / STATUS_FILE) if is_recorded else None,
                    )
                )
        claimed.append(name_parts)

    def instance_record(self, *, instance_id: str) -> InstanceRecord | None:
        """Return the stored record of an instance, or ``None``."""
        return self._snapshot.instances.get(instance_id)

    def record_instance(self, *, instance_id: str, record: InstanceRecord) -> None:
        """Store an instance record and persist ``status.json`` when allowed."""
        self._snapshot = self._snapshot.with_instance(instance_id=instance_id, record=record)
        self._persist_status()

    def invalidation_reason(self, *, instance_id: str) -> str | None:
        """Return why ``RunOptions`` force this instance to run, or ``None``.

        ``claim_instance`` marks the ``rerun_from`` stage as reached, so every
        instance claimed after it is invalidated too.

        Args:
            instance_id: The instance id.

        Returns:
            A short reason for the log, or ``None`` if the cache may be used.
        """
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
            self._install_sigterm_handler()
            self._install_sigusr2_handler()
            seed_everything(seed=self._seed)
            if self.writes_files:
                self._directory.mkdir(parents=True, exist_ok=True)
                self._lock_nonce = acquire_lock(path=self._directory / LOCK_FILE)
                self._heartbeat = Heartbeat(
                    path=self._directory / LOCK_FILE, nonce=self._lock_nonce
                )
                self._attach_file_log()
            existing = read_status(path=self._directory / STATUS_FILE)
            if existing is not None:
                self._snapshot = existing.with_run_state(run_state=RunState.RUNNING)
            if self._options.dry_run:
                _LOGGER.info('dry run of %s; nothing is executed or written', self._directory)
                _LOGGER.info('config:\n%s', config_to_yaml(config=self._config))
            elif self.writes_files:
                save_config(config=self._config, path=self._directory / CONFIG_FILE)
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
                self._finish_trackers(run_state=run_state)
            self._log_summary(run_state=run_state)
        finally:
            self._release()
        return False

    def _finish_trackers(self, *, run_state: RunState) -> None:
        for tracker in self._trackers:
            _call_tracker(
                tracker=tracker,
                action='finish',
                call=lambda tracker=tracker: tracker.finish(
                    status=RunStatus.from_run_state(run_state=run_state)
                ),
            )

    def _persist_status(self) -> None:
        if self.writes_files:
            write_status(path=self._directory / STATUS_FILE, snapshot=self._snapshot)

    def _install_sigterm_handler(self) -> None:
        """Turn SIGTERM into ``SystemExit`` so a SLURM kill still records the final state.

        Only the main thread can install a handler, and a handler that someone
        else installed (Lightning does for requeueing) is left alone.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        previous = signal.getsignal(signal.SIGTERM)
        if previous is not signal.SIG_DFL:
            return
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self._previous_sigterm = previous

    def _install_sigusr2_handler(self) -> None:
        sigusr2 = getattr(signal, 'SIGUSR2', None)
        if sigusr2 is None or threading.current_thread() is not threading.main_thread():
            return
        previous = signal.getsignal(sigusr2)
        if previous is not signal.SIG_DFL:
            _LOGGER.warning('replacing an existing SIGUSR2 handler for this run')
        signal.signal(sigusr2, self._on_sigusr2)
        self._previous_sigusr2 = previous

    def _on_sigusr2(self, signal_number: int, frame: FrameType | None) -> None:
        del signal_number, frame
        self._time_limit_signalled = True
        if not self._writer:
            _LOGGER.info('SIGUSR2 received on a non-zero rank; rank 0 stops the run')
            return
        if self._executing:
            message = 'SIGUSR2 before the SLURM time limit; resubmit to resume'
            raise JobTimeLimitSignal(message)

    def _on_sigterm(self, signal_number: int, frame: FrameType | None) -> None:
        del signal_number, frame
        if self._time_limit_signalled:
            _LOGGER.info('SIGTERM ignored while the run saves after SIGUSR2')
            return
        raise SystemExit(_SIGTERM_EXIT_CODE)

    def _configure_logging(self) -> None:
        level = logging.getLevelNamesMapping()[self._options.log_level]
        root_logger = logging.getLogger()
        self._previous_root_level = root_logger.level
        root_logger.setLevel(min(root_logger.level or logging.WARNING, level))
        tslab_logger = logging.getLogger('tslab')
        self._previous_tslab_level = tslab_logger.level
        tslab_logger.setLevel(level)
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
            environment = self.environment()
            self._meta = {
                'format_version': META_FORMAT_VERSION,
                'relative_path': self._relative_path.as_posix(),
                'seed': self._seed,
                'config_hash': compute_fingerprint(document={'config': canonical_config}),
                'config_hash_without_seed': compute_fingerprint(document={'config': without_seed}),
                # The latest execution only; the meta.json of each instance keeps its own.
                'versions': environment['versions'],
                'git': environment['git'],
                'hostname': hostname(),
                'started_at': utc_now_iso(),
            }
        self._meta['finished_at'] = finished_at
        atomic_write_json(path=self._directory / META_FILE, document=self._meta)

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
        if not self._writer:
            return  # one summary per job, not one per rank
        if self._options.dry_run:
            _LOGGER.info('dry run plan:')
            for instance_id, action in self._dry_run_plan:
                _LOGGER.info('  %-40s %s', instance_id, action)
            return
        # Only instances this execution called: records of earlier executions stay out.
        records = {
            instance_id: record
            for instance_id, record in self._snapshot.instances.items()
            if instance_id in self._claimed_ids
        }
        counts = Counter(record.state for record in records.values())
        _LOGGER.info(
            'run %s: %s',
            run_state.value,
            ', '.join(f'{counts[state]} {state.value}' for state in InstanceState),
        )
        for instance_id, record in records.items():
            log_record = _RECORD_LOGGERS.get(record.state)
            if log_record is not None:
                log_record(instance_id=instance_id, record=record)
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
        if self._lock_nonce is not None:
            release_lock(path=self._directory / LOCK_FILE, nonce=self._lock_nonce)
            self._lock_nonce = None
        if self._previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
            self._previous_sigterm = None
        if self._previous_sigusr2 is not None:
            signal.signal(signal.SIGUSR2, self._previous_sigusr2)
            self._previous_sigusr2 = None
        root_logger = logging.getLogger()
        for handler in self._log_handlers:
            root_logger.removeHandler(handler)
            handler.close()
        self._log_handlers.clear()
        if self._previous_root_level is not None:
            root_logger.setLevel(self._previous_root_level)
        if self._previous_tslab_level is not None:
            logging.getLogger('tslab').setLevel(self._previous_tslab_level)


def _log_failed(*, instance_id: str, record: InstanceRecord) -> None:
    _LOGGER.error('  failed  %s: %s', instance_id, record.error)


def _log_skipped(*, instance_id: str, record: InstanceRecord) -> None:
    _LOGGER.warning('  skipped %s: %s', instance_id, record.reason)


#: The summary line of an instance by its state; the other states log nothing.
_RECORD_LOGGERS: Mapping[InstanceState, Callable[..., None]] = MappingProxyType(
    {InstanceState.FAILED: _log_failed, InstanceState.SKIPPED: _log_skipped}
)


def _duplicate_message(*, stage_name: str, name_parts: tuple[str, ...]) -> str:
    if name_parts == ('default',):
        return (
            f'stage {stage_name!r} called twice without name=; '
            f'pass a distinct name= for each instance'
        )
    return f'stage {stage_name!r} called twice with name {"/".join(name_parts)!r}'


def _conflict_message(
    *,
    stage_name: str,
    name_parts: tuple[str, ...],
    other: tuple[str, ...],
    recorded_in: Path | None,
) -> str:
    """The error for two instance names that would be stored one inside the other.

    ``recorded_in`` is the ``status.json`` that holds ``other`` when it comes from an
    earlier execution, and ``None`` when ``other`` was claimed in this process.
    """
    message = (
        f'instance name {"/".join(name_parts)!r} of stage {stage_name!r} conflicts '
        f'with {"/".join(other)!r}: one would be stored inside the other'
    )
    if recorded_in is None:
        return message
    return message + (
        f'. {"/".join(other)!r} comes from an earlier execution in status.json; '
        f'if it is obsolete, delete its folder and its entry in '
        f'{recorded_in.as_posix()}, or choose another name'
    )


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


def _project_directory() -> Path:
    """Return the folder of the main script, or the working directory without one."""
    script = getattr(sys.modules.get('__main__'), '__file__', None)
    if isinstance(script, str) and Path(script).resolve().parent.is_dir():
        return Path(script).resolve().parent
    return Path.cwd()
