"""``LightningTrainerConfig``: every setting of ``LightningTrainer``."""

from dataclasses import dataclass, field
from typing import Annotated

from tslab.core.fingerprint import NotFingerprinted

__all__ = ['LightningTrainerConfig']


@dataclass(frozen=True, kw_only=True)
class LightningTrainerConfig:
    """Settings for ``LightningTrainer`` (``--trainer.*`` flags via tyro).

    Fields marked ``NotFingerprinted`` cannot change
    the trained weights, so changing them reuses a cached checkpoint; changing
    any other field trains again from scratch.

    Attributes:
        max_epochs: Upper bound on training epochs.
        min_epochs: Epochs trained before any stop signal is honoured, including
            early stopping and the time limit.
        accelerator: Lightning accelerator (``'auto'``, ``'cpu'``, ``'gpu'``, ...).
        devices: Number of devices, or ``'auto'``.
        check_val_every_n_epoch: Validation interval in epochs.
        log_every_n_steps: Logging interval in optimizer steps; 200 keeps W&B
            traffic under the per-project rate limit.
        enable_progress_bar: Show Lightning's progress bar.
        monitor: Metric used to pick the best checkpoint and for early stopping
            (lower is better).
        checkpoint_save_top_k: Number of best checkpoints kept besides the last one.
        early_stopping_patience: Validation checks without improvement before
            stopping; ``None`` computes ``max(1, max_epochs // (10 * check_val_every_n_epoch))``.
        early_stopping_min_delta: Smallest change of ``monitor`` that counts as improvement.
        max_training_time: Time limit in sbatch format (``"HH:MM"``, ``"HH:MM:SS"`` or
            ``"D-HH:MM:SS"``). Under SLURM it is capped by the remaining job time.
        slurm_time_fraction: Fraction of the remaining SLURM job time used for
            training when ``max_training_time`` is unset or too long.
        extra_trainer_kwargs: Further ``lightning.Trainer`` arguments, for example
            ``{'gradient_clip_val': 1.0}``. Fingerprinted, because their effect is unknown.
    """

    max_epochs: int = 2500
    min_epochs: int = 3
    accelerator: Annotated[str, NotFingerprinted] = 'auto'
    devices: Annotated[int | str, NotFingerprinted] = 'auto'
    check_val_every_n_epoch: int = 1
    log_every_n_steps: Annotated[int, NotFingerprinted] = 200
    enable_progress_bar: Annotated[bool, NotFingerprinted] = True
    monitor: str = 'val_loss_epoch'
    checkpoint_save_top_k: Annotated[int, NotFingerprinted] = 3
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.001
    max_training_time: Annotated[str | None, NotFingerprinted] = None
    slurm_time_fraction: Annotated[float, NotFingerprinted] = 0.5
    extra_trainer_kwargs: dict[str, bool | int | float | str] = field(default_factory=dict)
