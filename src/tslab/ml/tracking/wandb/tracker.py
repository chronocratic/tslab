"""``WandbTracker``: Weights & Biases hardened for long HPC runs.

Long HPC runs kept showing "crashed" on W&B. The root cause: the W&B server marks a run
"crashed" after about 8 minutes without filestream traffic, and a hung HTTP
request blocks the heartbeat, so the request timeout matters more than the
retry count.

The private ``x_*`` settings below were verified against wandb-core 0.30.0.
On every wandb upgrade, re-check for each setting where the wandb-core source
reads it (``wandb/bin/wandb-core`` is built from ``core/``), and keep the
``wandb`` extra pinned to the verified range.
"""

from collections.abc import Generator
import contextlib
import importlib.util
import logging
import os
from pathlib import Path
import shutil
from typing import override, TYPE_CHECKING

from lightning.pytorch.loggers import Logger, WandbLogger

from tslab.lab.core.tracker import RunStatus
from tslab.ml.tracking.lightning_bridge import LightningLoggerTracker
from tslab.ml.tracking.wandb_mode import WandbMode

if TYPE_CHECKING:
    import wandb
    from wandb.sdk.wandb_run import Run

__all__ = ['WandbTracker', 'build_wandb_settings']

_LOGGER = logging.getLogger(__name__)


def build_wandb_settings() -> 'wandb.Settings':
    """Return the hardened ``wandb.Settings`` used by every TSLab W&B run.

    Returns:
        Settings verified against wandb-core 0.30.0 (see the module docstring).
    """
    import wandb  # noqa: PLC0415 # why: optional extra, imported only when W&B is used

    # Deliberately NOT set: x_file_stream_retry_max (the wandb-core default 10085 is a
    # 7-day budget), x_file_stream_retry_wait_min/max (defaults 2s/60s), x_graphql_*,
    # x_file_transfer_*, heartbeat_seconds (dead: wandb-core hardcodes 30s).
    # Settings always win over WANDB_* environment variables.
    return wandb.Settings(
        # Allow a slow wandb-core start-up on loaded nodes.
        x_service_wait=240,
        init_timeout=300,
        # The key fix: bound a hung filestream POST well inside the ~8-minute crash window.
        x_file_stream_timeout_seconds=60,
        # Fewer requests under the per-project rate limit; heartbeats run on their own timer.
        x_file_stream_transmit_interval=120,
        x_stats_sampling_interval=60.0,
        # Keep stdout/stderr off the filestream; the shell log is uploaded once at finish.
        console='off',
        # Wait for the upload on finish and never raise: a non-zero timeout marks the run
        # "crashed" mid-upload.
        finish_timeout=0,
        finish_timeout_raises=False,
        # Room for the final summary flush.
        summary_timeout=300,
    )


class WandbTracker(LightningLoggerTracker):
    """Wraps the Lightning ``WandbLogger`` with hardening for long HPC runs.

    - Initializes eagerly in ``start``, so a wandb-core start-up failure
      happens where TSLab can degrade instead of inside ``Trainer.fit``.
    - Falls back online, then offline, then no W&B, each with a warning. A
      missing ``wandb`` package gives a warning naming the ``tslab[wandb]`` extra.
    - Tags are the given tags plus every part of the relative path of the run, so
      ``ts2vec/Coffee/seed_42`` gives the tags ``ts2vec``, ``Coffee`` and ``seed_42``.
    - ``finish`` uploads the shell log, then calls ``run.finish(exit_code=...)``:
      0 marks the run finished, 1 marks it failed.
    """

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        mode: WandbMode = WandbMode.ONLINE,
        tags: tuple[str, ...] = (),
        shell_log_path: Path | None = None,
    ) -> None:
        """Configure the tracker.

        Args:
            project: W&B project.
            entity: W&B entity (user or team); ``None`` uses the default entity.
            mode: ``WandbMode.ONLINE`` or ``WandbMode.OFFLINE``.
            tags: Extra tags.
            shell_log_path: SLURM stdout/stderr file uploaded at finish, if it exists.
        """
        super().__init__()
        self._project = project
        self._entity = entity
        self._mode = mode
        self._tags = tags
        self._shell_log_path = shell_log_path

    @override
    def _create_logger(self, *, run_directory: Path, run_name: str) -> Logger | None:
        if importlib.util.find_spec('wandb') is None:
            _LOGGER.warning('wandb is not installed; install tslab[wandb]. Continuing without W&B.')
            return None
        tags = [*self._tags, *Path(run_name).parts]
        modes = [self._mode] if self._mode is WandbMode.OFFLINE else [self._mode, WandbMode.OFFLINE]
        for mode in modes:
            built = self._try_build(
                run_directory=run_directory,
                run_name=run_name,
                tags=tags,
                mode=mode,
                last=mode == modes[-1],
            )
            if built is not None:
                return built
        return None

    def _try_build(
        self, *, run_directory: Path, run_name: str, tags: list[str], mode: WandbMode, last: bool
    ) -> WandbLogger | None:
        """Build the logger in one mode; ``None`` with a warning when W&B fails."""
        try:
            return self._build(run_directory=run_directory, run_name=run_name, tags=tags, mode=mode)
        except Exception:  # wandb raises version-unstable private exception types
            if last:
                _LOGGER.warning(
                    'W&B initialization failed, including offline. Continuing without W&B; '
                    'results are unaffected and CSV tracking is still on.',
                    exc_info=True,
                )
            else:
                _LOGGER.warning(
                    'W&B init failed in %s mode; retrying offline. Sync later with '
                    '`tslab wandb-sync`.',
                    mode,
                    exc_info=True,
                )
        return None

    def _build(
        self, *, run_directory: Path, run_name: str, tags: list[str], mode: WandbMode
    ) -> WandbLogger:
        with _environment_variable(name='WANDB_DISABLE_CODE', value='1'):
            wandb_logger = WandbLogger(
                project=self._project,
                entity=self._entity,
                name=run_name,
                mode=mode.value,
                log_model=False,
                # wandb.init(dir=X) creates X/wandb/ itself; passing .../trackers/wandb
                # would give .../trackers/wandb/wandb.
                save_dir=run_directory / 'trackers',
                tags=tags,
                settings=build_wandb_settings(),
            )
            # Eager init: Lightning would otherwise start wandb lazily inside Trainer.fit,
            # where a start-up failure is not caught.
            _ = wandb_logger.experiment
        return wandb_logger

    @override
    def log_file(self, *, path: Path) -> None:
        if isinstance(self._logger, WandbLogger):
            _upload_file(run=self._logger.experiment, path=path)

    @override
    def finish(self, *, status: RunStatus) -> None:
        if not isinstance(self._logger, WandbLogger):
            return
        run = self._logger.experiment
        if self._shell_log_path is not None and self._shell_log_path.is_file():
            try:
                _upload_file(run=run, path=self._shell_log_path)
            except Exception:  # best-effort: the upload must never fail the run
                _LOGGER.warning('could not upload the shell log to W&B', exc_info=True)
        # exit_code alone decides the final state (0 finished, otherwise failed). In wandb
        # 0.30, Run.status is a method; assigning to it has no effect.
        run.finish(exit_code=0 if status is RunStatus.FINISHED else 1)


def _upload_file(*, run: 'Run', path: Path) -> None:
    """Copy ``path`` into the W&B run folder and register it for upload."""
    run_directory = Path(run.dir)
    destination = run_directory / path.name
    shutil.copy2(path, destination)
    run.save(str(destination), base_path=str(run_directory))


@contextlib.contextmanager
def _environment_variable(*, name: str, value: str) -> Generator[None]:
    """Set an environment variable for the duration of the block, then restore it."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
