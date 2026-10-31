"""Exceptions raised by the Lightning integration."""

__all__ = ['TrainingPaused']


class TrainingPaused(RuntimeError):  # noqa: N818 # why: public API distinguishes a resumable pause
    """Training paused at the job deadline; resubmit to resume from its checkpoint."""
