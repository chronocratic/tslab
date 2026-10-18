"""The ready ``encode`` stage and the representation checks it shares."""

from collections.abc import Callable

import torch

from tslab.core.artifact import Checkpoint, Tensors
from tslab.core.paths import InstanceName
from tslab.core.run import Run
from tslab.core.stage import stage
from tslab.interfaces import Encoder, LabeledSplits, SPLITS, Trainer

__all__ = ['check_representations', 'encode']

# At or above this fraction of entirely non-finite rows the encoder has diverged.
_DEAD_ENCODER_FRACTION = 0.99
_MATRIX_DIMENSIONS = 2


def check_representations(
    *, representations: torch.Tensor, expected_rows: int, description: str
) -> None:
    """Fail loudly on representations that cannot be right.

    A few non-finite rows pass here, but the probes impute only NaN, so an
    ``inf`` value still fails when the probe is fitted. When 99 % or more of
    the rows are entirely non-finite, the encoder has diverged and evaluating
    would silently report garbage.

    Args:
        representations: Shape ``(rows, D)``.
        expected_rows: The number of encoded samples.
        description: What was encoded, for the error message.

    Raises:
        ValueError: If the shape is wrong or the encoder has diverged.
    """
    if representations.ndim != _MATRIX_DIMENSIONS:
        message = f'{description}: expected 2-D representations, got {tuple(representations.shape)}'
        raise ValueError(message)
    if len(representations) != expected_rows:
        message = f'{description}: {len(representations)} rows for {expected_rows} samples'
        raise ValueError(message)
    if expected_rows == 0:
        return
    finite = torch.isfinite(representations)
    dead_fraction = 1.0 - finite.all(dim=1).float().mean().item()
    if dead_fraction < _DEAD_ENCODER_FRACTION:
        return
    nan_count = int(torch.isnan(representations).sum().item())
    inf_count = int(torch.isinf(representations).sum().item())
    message = (
        f'{description}: {dead_fraction:.2%} of the rows are entirely non-finite '
        f'({nan_count} NaN and {inf_count} inf values in shape {tuple(representations.shape)}); '
        'the encoder has diverged'
    )
    raise ValueError(message)


@stage(name='encode', output=Tensors)
def encode[M](  # noqa: PLR0913 # why: keyword-only public API; each input is fingerprinted
    *,
    run: Run,
    name: InstanceName | None = None,
    trainer: Trainer[M],
    model: M,
    checkpoint: Checkpoint,
    data: LabeledSplits,
    encoder: Callable[[M], Encoder],
) -> Tensors:
    """Encode every split with the best weights of ``checkpoint``.

    Args:
        run: The current run.
        name: Instance name, for encoding several times in one run.
        trainer: Loads the weights into ``model``.
        model: A model of the trained architecture.
        checkpoint: From ``train``, or ``tslab.Checkpoint.external(...)``.
        data: Provides each split as ``(inputs, targets)``.
        encoder: Builds the encoder from the loaded model, for example
            ``tslab.ModelAsEncoder()``; it must be a frozen dataclass or
            otherwise fingerprintable.

    Returns:
        ``<split>_representations`` and ``<split>_targets`` for the train,
        val and test splits.
    """
    del run, name
    trainer.load(model=model, checkpoint=checkpoint)
    model_encoder = encoder(model)
    tensors: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for split in SPLITS:
            inputs, targets = data.labeled_split(split=split)
            representations = model_encoder.encode(inputs=inputs)
            check_representations(
                representations=representations,
                expected_rows=len(inputs),
                description=f'the {split} split',
            )
            tensors[f'{split}_representations'] = representations
            tensors[f'{split}_targets'] = targets
    return Tensors(tensors=tensors)
