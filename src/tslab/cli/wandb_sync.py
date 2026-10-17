"""``tslab wandb-sync``: find local W&B runs the server shows as crashed or failed, and resync them.

Long HPC runs often
finish their work but show "crashed" on W&B, because the final exit record
never reached the server. ``wandb sync`` replays the local transaction log,
which repairs the state **only if** that log recorded a clean exit. This
command finds such runs, resyncs them, and verifies the result. It never
claims a repair it did not observe.

Exit codes are read with ``tslab.tracking.wandb_log``, because wandb 0.30
removed ``wandb.sdk.internal.datastore``; a check built on it silently does
nothing.

Output, one line per run: ``index/total action [server=... local=...] note path``.
Actions are ``skip``, ``WOULD`` (dry run), ``synced`` and ``FAILED``. Local
states are what the files of the run say happened:

- ``finished``: an exit record with code 0 was written, so a resync can turn the run green.
- ``exit-err``: the exit record has a non-zero code. The run really failed and stays failed.
- ``running``: no exit record, but the log was written in the last 30 minutes.
  The run is alive and is never touched.
- ``no-exit``: no exit record and the log went quiet, which usually means it was
  SIGKILLed. A resync uploads the data, but the state stays crashed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Annotated, TYPE_CHECKING

import tyro

from tslab.core.paths import require_path
from tslab.tracking.wandb_log import read_exit_codes, read_run_coordinates

if TYPE_CHECKING:
    from wandb.apis.public import Api, Run

__all__ = [
    'LocalRun',
    'LocalState',
    'WandbSync',
    'discover_runs',
    'parse_run_url',
    'plan_action',
    'resolve_run_path',
]

_LOGGER = logging.getLogger(__name__)
# entity/project/id appear in the logs in two forms, depending on the wandb version.
_FILESTREAM_PATTERN = re.compile(r'wandb\.ai/files/([^/\s]+)/([^/\s]+)/([^/\s]+)/file_stream')
_RUN_URL_PATTERN = re.compile(r'wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([^/\s?]+)')
# The run id is also encoded in the folder name: run-YYYYMMDD_HHMMSS-<id>.
_FOLDER_ID_PATTERN = re.compile(r'run-\d+_\d+-(\w+)$')
# What wandb logs next to its "finishing run" record: "got exitcode: 1".
_EXIT_CODE_PATTERN = re.compile(r'got exitcode: (-?\d+)')
# `wandb sync` prints this and still exits 0 when a filter removed every run.
_NOTHING_SYNCED = 'No runs to sync'
# The server applies a replayed exit record asynchronously; back off before re-checking.
_VERIFY_BACKOFF_SECONDS = (5, 10, 20)
# A run whose .wandb log was written this recently counts as alive.
_RUNNING_WINDOW_SECONDS = 30 * 60


class LocalState(StrEnum):
    """What the files of a run say happened (see the module docstring)."""

    FINISHED = 'finished'
    EXIT_ERROR = 'exit-err'
    RUNNING = 'running'
    NO_EXIT = 'no-exit'


class Action(StrEnum):
    """What the command does with one run."""

    SKIP = 'skip'
    WOULD = 'WOULD'
    SYNC = 'sync'
    SYNCED = 'synced'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class LocalRun:
    """A local W&B run folder and the server coordinates parsed from its logs.

    Attributes:
        run_directory: The ``run-*`` or ``offline-run-*`` folder holding the ``.wandb`` log.
        entity: W&B entity, or ``None`` if the logs do not show it.
        project: W&B project, or ``None`` if the logs do not show it.
        run_id: W&B run id, or ``None`` if neither the logs nor the folder name show it.
    """

    run_directory: Path
    entity: str | None
    project: str | None
    run_id: str | None

    def local_state(self, *, now: float | None = None) -> LocalState:
        """Infer the local state from the logs and file times of the run.

        Args:
            now: Reference time for the liveness window; ``None`` means now.

        Returns:
            The local state.
        """
        for name in ('debug-internal.log', 'debug.log'):
            log = self.run_directory / 'logs' / name
            if not log.is_file():
                continue
            text = log.read_text(encoding='utf-8', errors='ignore')
            if 'finishing run' not in text:
                continue
            code = _EXIT_CODE_PATTERN.search(text)
            clean = code is not None and code.group(1) == '0'
            return LocalState.FINISHED if clean else LocalState.EXIT_ERROR
        reference = time.time() if now is None else now
        modified = [path.stat().st_mtime for path in self.run_directory.glob('*.wandb')]
        if modified and reference - max(modified) < _RUNNING_WINDOW_SECONDS:
            return LocalState.RUNNING
        return LocalState.NO_EXIT

    def transaction_log(self) -> Path | None:
        """Return the ``.wandb`` transaction log of the run, if present."""
        logs = sorted(self.run_directory.glob('*.wandb'))
        return logs[0] if logs else None


def parse_run_url(*, text: str) -> tuple[str, str, str] | None:
    """Extract ``(entity, project, run_id)`` from the first W&B URL in ``text``.

    Args:
        text: Log file contents.

    Returns:
        The coordinates, or ``None`` if ``text`` contains no W&B run URL.
    """
    match = _FILESTREAM_PATTERN.search(text) or _RUN_URL_PATTERN.search(text)
    if match is None:
        return None
    # Every group is mandatory in both patterns, so each one matched a string.
    return str(match[1]), str(match[2]), str(match[3])


def _read_coordinates(*, run_directory: Path) -> tuple[str | None, str | None, str | None]:
    """Read ``(entity, project, run_id)``: debug-log URL, then transaction log, then folder name."""
    for name in ('debug-internal.log', 'debug.log'):
        log = run_directory / 'logs' / name
        if log.is_file():
            coordinates = parse_run_url(text=log.read_text(encoding='utf-8', errors='ignore'))
            if coordinates is not None:
                return coordinates
    entity, project, run_id = None, None, None
    for transaction_log in sorted(run_directory.glob('*.wandb')):
        try:
            entity, project, run_id = read_run_coordinates(path=transaction_log)
        except (OSError, ValueError) as error:
            _LOGGER.warning('cannot read %s: %s', transaction_log, error)
        break
    if run_id is None:
        folder_match = _FOLDER_ID_PATTERN.search(run_directory.name)
        run_id = str(folder_match[1]) if folder_match is not None else None
    return entity, project, run_id


def discover_runs(*, root: Path) -> list[LocalRun]:
    """Find every W&B run folder (a folder containing ``*.wandb``) under ``root``.

    Args:
        root: Folder searched recursively, for example the TSLab runs root.

    Returns:
        Runs sorted by path; the ``latest-run`` pointer folder is skipped.
    """
    runs: dict[Path, LocalRun] = {}
    for wandb_file in root.rglob('*.wandb'):
        run_directory = wandb_file.parent.resolve()
        if run_directory.name == 'latest-run' or run_directory in runs:
            continue
        entity, project, run_id = _read_coordinates(run_directory=run_directory)
        runs[run_directory] = LocalRun(
            run_directory=run_directory, entity=entity, project=project, run_id=run_id
        )
    return sorted(runs.values(), key=lambda run: run.run_directory.as_posix())


class _ApiPool:
    """One reused ``wandb.Api`` client, rebuilt once if a query fails."""

    def __init__(self) -> None:
        self._api: Api | None = None

    def _client(self) -> 'Api':
        if self._api is None:
            import wandb  # noqa: PLC0415 # why: optional extra

            self._api = wandb.Api()
        return self._api

    def run(self, *, path: str, fresh: bool) -> 'Run':
        try:
            client = self._client()
            if fresh:
                # why: no public way to bypass the run cache
                client._runs.pop(path, None)  # noqa: SLF001
            return client.run(path)
        except Exception:  # noqa: BLE001 # why: wandb raises version-unstable types; rebuild once
            self._api = None
            return self._client().run(path)


def resolve_run_path(*, run: LocalRun, entity: str | None, project: str | None) -> str | None:
    """Return the ``entity/project/run_id`` path of the run on the server.

    The coordinates of the run win; ``entity`` and ``project`` fill only the
    gaps (runs that crashed before their logs showed them).

    Args:
        run: The local run.
        entity: Fallback entity (``--wandb-entity``, else ``WANDB_ENTITY``).
        project: Fallback project (``--wandb-project``, else ``WANDB_PROJECT``).

    Returns:
        The path, or ``None`` for offline runs and runs whose coordinates stay unknown.
    """
    resolved_entity = run.entity if run.entity is not None else entity
    resolved_project = run.project if run.project is not None else project
    if (
        run.run_directory.name.startswith('offline-run-')
        or resolved_entity is None
        or resolved_project is None
        or run.run_id is None
    ):
        return None
    return f'{resolved_entity}/{resolved_project}/{run.run_id}'


@dataclass
class _ServerQuery:
    """Looks up server states, with fallback coordinates for runs that crashed early."""

    entity: str | None
    project: str | None
    pool: _ApiPool = field(default_factory=_ApiPool)

    def state(self, *, run: LocalRun, fresh: bool = False) -> str:
        """Return the server state, ``'offline'`` without coordinates, ``'unknown'`` on errors."""
        path = resolve_run_path(run=run, entity=self.entity, project=self.project)
        if path is None:
            return 'offline'
        try:
            return str(self.pool.run(path=path, fresh=fresh).state).lower()
        except Exception as error:  # noqa: BLE001 # why: wandb raises version-unstable types
            _LOGGER.warning('could not query %s: %s', run.run_directory.name, error)
            return 'unknown'


def plan_action(
    *, local: LocalState, state: str, target_states: frozenset[str], dry_run: bool
) -> tuple[Action, str]:
    """Decide what to do with one run.

    Args:
        local: The local state of the run.
        state: The server state of the run.
        target_states: Server states to repair.
        dry_run: Report instead of syncing.

    Returns:
        ``(action, note)``; the note explains a skip or a data-only sync.
    """
    if state not in target_states:
        return Action.SKIP, ''
    if local is LocalState.RUNNING:
        return Action.SKIP, ' live'
    note = ' data-only' if local is LocalState.NO_EXIT else ''
    return (Action.WOULD if dry_run else Action.SYNC), note


def sync_command(*, run: LocalRun) -> list[str]:
    """Return the ``wandb sync`` command for one run.

    ``--no-skip-online`` and ``--no-skip-synced`` are required: since wandb
    0.28, ``wandb sync`` otherwise filters out online runs and already-synced
    runs, prints "No runs to sync" and exits 0 without uploading anything.

    Args:
        run: The run to upload.

    Returns:
        The argument vector, run through the ``wandb`` module of this Python.
    """
    return [
        sys.executable,
        '-m',
        'wandb',
        'sync',
        '--no-skip-online',
        '--no-skip-synced',
        str(run.run_directory),
    ]


def _resync(*, run: LocalRun) -> bool:
    result = subprocess.run(  # noqa: S603 # why: fixed command built by sync_command
        sync_command(run=run), capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or _NOTHING_SYNCED in output:
        _LOGGER.error('wandb sync did not upload %s:\n%s', run.run_directory, output)
        return False
    return True


def _last_exit_code(*, run: LocalRun) -> int | None:
    log = run.transaction_log()
    if log is None:
        return None
    try:
        codes = read_exit_codes(path=log)
    except (OSError, ValueError) as error:
        _LOGGER.warning('cannot read %s: %s', log, error)
        return None
    return codes[-1] if codes else None


def _resync_verified(
    *,
    run: LocalRun,
    local: LocalState,
    note: str,
    query: _ServerQuery,
    sleep: Callable[[float], None],
) -> tuple[bool, str]:
    """Resync, and verify the server state flipped when the logs promise a clean exit."""
    last_code = _last_exit_code(run=run)
    repairable = local is LocalState.FINISHED and last_code == 0
    if local is LocalState.FINISHED and not repairable:
        note = f' data-only exit={"none" if last_code is None else last_code}'
    if not _resync(run=run):
        return False, note
    # why: an offline run has no server state to flip, so the upload is the outcome
    if resolve_run_path(run=run, entity=query.entity, project=query.project) is None:
        return True, note
    if not repairable:
        return True, note
    after = 'unknown'
    for delay in _VERIFY_BACKOFF_SECONDS:
        sleep(delay)
        after = query.state(run=run, fresh=True)
        if after == 'finished':
            return True, note
    return False, f' still {after}'


@dataclass(frozen=True)
class WandbSync:
    """Resync local W&B runs that the server shows as crashed or failed.

    Attributes:
        root: Folder searched for W&B run folders, for example your runs root.
        states: Server states to repair.
        include_offline: Also upload offline runs (never synced).
        wandb_project: Only touch runs of this project. Also the fallback project for
            runs whose logs do not show one; unset falls back to ``WANDB_PROJECT``,
            which never filters.
        wandb_entity: Fallback entity for runs whose logs do not show one; unset
            falls back to ``WANDB_ENTITY``.
        dry_run: List what would be synced without uploading.
    """

    root: Annotated[Path, tyro.conf.Positional]
    states: tuple[str, ...] = ('crashed', 'failed', 'killed')
    include_offline: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    dry_run: bool = False

    def run(self, *, sleep: Callable[[float], None] = time.sleep) -> int:
        """Execute the command.

        Args:
            sleep: Waits between verification queries; tests pass a no-op.

        Returns:
            Process exit code: 0 if every attempted resync succeeded, otherwise 1.
        """
        require_path(value=self.root, parameter='root')
        target_states = frozenset(
            {state.lower() for state in self.states}
            | ({'offline'} if self.include_offline else set())
        )
        runs = discover_runs(root=self.root)
        _LOGGER.info('found %d W&B run folder(s) under %s', len(runs), self.root)
        # Environment variables only fill coordinate gaps; the filter is never implicit.
        query = _ServerQuery(
            entity=self.wandb_entity
            if self.wandb_entity is not None
            else os.environ.get('WANDB_ENTITY'),
            project=self.wandb_project
            if self.wandb_project is not None
            else os.environ.get('WANDB_PROJECT'),
        )
        failures = 0
        for index, run in enumerate(runs, start=1):
            local = run.local_state()
            other_project = (
                self.wandb_project is not None
                and run.project is not None
                and run.project != self.wandb_project
            )
            state = 'other' if other_project else query.state(run=run)
            action, note = plan_action(
                local=local, state=state, target_states=target_states, dry_run=self.dry_run
            )
            if action is Action.SYNC:
                ok, note = _resync_verified(
                    run=run, local=local, note=note, query=query, sleep=sleep
                )
                action = Action.SYNCED if ok else Action.FAILED
                failures += 0 if ok else 1
            _LOGGER.log(
                logging.ERROR if action is Action.FAILED else logging.INFO,
                '%3d/%-3d  %-6s [server=%-8s local=%-8s]%s %s',
                index,
                len(runs),
                action.value,
                state,
                local.value,
                note,
                run.run_directory,
            )
        return 1 if failures else 0
