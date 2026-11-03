"""The modes of the Weights & Biases tracker."""

from enum import StrEnum

__all__ = ['WandbMode']


class WandbMode(StrEnum):
    """How a run talks to Weights & Biases; the text is the mode name of ``wandb``.

    Attributes:
        ONLINE: Send metrics to the W&B server while the run goes on.
        OFFLINE: Write the run to disk; sync it later with ``tslab wandb-sync``.
        DISABLED: Do not use W&B.
    """

    ONLINE = 'online'
    OFFLINE = 'offline'
    DISABLED = 'disabled'
