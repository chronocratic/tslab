"""``LightningTrainerConfig``: every setting of ``LightningTrainer``."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Annotated, override

from tslab.lab.core.fingerprint import NotFingerprinted
from tslab.lab.core.markers import HardwareSetting, MachineSetting

__all__ = ['LightningTrainerConfig']


class FrozenKwargs(Mapping[str, bool | int | float | str]):
    """A read-only, hashable and picklable mapping of trainer arguments."""

    def __init__(self, items: Mapping[str, bool | int | float | str] | None = None) -> None:
        """Copy ``items``."""
        self._items: dict[str, bool | int | float | str] = dict(items or {})

    @override
    def __getitem__(self, key: str) -> bool | int | float | str:
        return self._items[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    @override
    def __len__(self) -> int:
        return len(self._items)

    @override
    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    @override
    def __repr__(self) -> str:
        return repr(self._items)


@dataclass(frozen=True, kw_only=True)
class LightningTrainerConfig:
    """Settings for ``LightningTrainer`` (``--trainer.*`` flags via tyro).

    Fields marked ``NotFingerprinted`` cannot change the trained weights, so changing them
    reuses a cached checkpoint; changing any other field trains again from scratch.

    Attributes:
        max_epochs: Upper bound on training epochs.
        min_epochs: Epochs trained before early stopping or ``max_training_time`` is honored.
        accelerator: Lightning accelerator (``'auto'``, ``'cpu'``, ``'gpu'``, ...).
        devices: Number of devices. Defaults to one; use ``'auto'`` to use every visible device.
        check_val_every_n_epoch: Validation interval in epochs.
        log_every_n_steps: Logging interval in optimizer steps; 200 keeps W&B
            traffic under the per-project rate limit.
        enable_progress_bar: Show the progress bar of Lightning.
        monitor: Metric used to pick the best checkpoint and for early stopping
            (lower is better).
        checkpoint_save_top_k: Number of best checkpoints kept besides the last one.
        early_stopping_patience: Validation checks without improvement before
            stopping; ``None`` computes ``max(1, max_epochs // (10 * check_val_every_n_epoch))``.
        early_stopping_min_delta: Smallest change of ``monitor`` that counts as improvement.
        max_training_time: Total training limit in sbatch ``--time`` format. It counts across
            resumed jobs and ends training when reached.
        job_time_share: Fraction of the remaining SLURM job time used by each job before
            training pauses.
        extra_trainer_kwargs: Further ``lightning.Trainer`` arguments, for example
            ``{'gradient_clip_val': 1.0}``. Fingerprinted, because their effect is unknown.
    """

    max_epochs: int = 2500
    min_epochs: int = 3
    accelerator: Annotated[str, HardwareSetting, NotFingerprinted] = 'auto'
    devices: Annotated[int | str, HardwareSetting, NotFingerprinted] = 1
    check_val_every_n_epoch: int = 1
    log_every_n_steps: Annotated[int, NotFingerprinted] = 200
    enable_progress_bar: Annotated[bool, MachineSetting, NotFingerprinted] = True
    monitor: str = 'val_loss_epoch'
    checkpoint_save_top_k: Annotated[int, NotFingerprinted] = 3
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.001
    max_training_time: str | None = None
    job_time_share: Annotated[float, MachineSetting, NotFingerprinted] = 0.5
    extra_trainer_kwargs: Mapping[str, bool | int | float | str] = field(
        default_factory=FrozenKwargs
    )

    def __post_init__(self) -> None:
        """Copy ``extra_trainer_kwargs`` into a read-only mapping, so the config stays frozen."""
        object.__setattr__(self, 'extra_trainer_kwargs', FrozenKwargs(self.extra_trainer_kwargs))
