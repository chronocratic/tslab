"""The ``tslab wandb-sync`` command; the resync logic lives in ``tslab.ml.tracking.wandb.sync``."""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Annotated

import tyro

from tslab.lab.core.paths import require_path
from tslab.ml.tracking.wandb.sync import (
    _resync_verified,
    _ServerQuery,
    Action,
    discover_runs,
    plan_action,
)

__all__ = ['WandbSync']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class WandbSync:
    """Resync local W&B runs that the server shows as crashed or failed.

    For example:
        tslab wandb-sync runs/ --dry-run

    Attributes:
        root: Folder searched for W&B run folders, for example your runs root.
        states: Server states to repair.
        include_offline: Also upload offline runs (never synced).
        wandb_project: Only touch runs of this project. Also the fallback project for
            runs whose logs do not show one; unset falls back to WANDB_PROJECT,
            which never filters.
        wandb_entity: Fallback entity for runs whose logs do not show one; unset
            falls back to WANDB_ENTITY.
        dry_run: List what would be synced without uploading.
    """

    root: Annotated[Path, tyro.conf.arg(metavar='FOLDER'), tyro.conf.Positional]
    states: Annotated[tuple[str, ...], tyro.conf.arg(metavar='STATE')] = (
        'crashed',
        'failed',
        'killed',
    )
    include_offline: bool = False
    wandb_project: Annotated[
        str | None, tyro.conf.arg(metavar='PROJECT', help_behavior_hint='')
    ] = None
    wandb_entity: Annotated[str | None, tyro.conf.arg(metavar='ENTITY', help_behavior_hint='')] = (
        None
    )
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
