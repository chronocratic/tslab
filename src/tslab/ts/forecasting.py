"""The TS2Vec forecasting protocol (Yue et al., AAAI 2022) as stages.

Encoders with per-timestep output (``SequenceEncoder``) encode each full
series once; every horizon is then cut out of that encoding. Other encoders
encode sliding windows, once per horizon. Both paths produce the keys of
``tslab.stages.encode``, so ``tslab.stages.evaluate`` with
``tslab.probes.RidgeProbe`` evaluates the result.
"""

from collections.abc import Callable, Sequence

import torch

from tslab.artifact import Checkpoint, Tensors
from tslab.interfaces import Encoder, SequenceEncoder, SPLITS, Trainer
from tslab.paths import InstanceName, normalize_instance_name
from tslab.run import Run
from tslab.stage import stage
from tslab.stages import check_representations
from tslab.ts.info import checked_split_slices, ForecastingSeries

__all__ = [
    'encode_series',
    'encode_windows',
    'forecasting_representations',
    'horizon_samples',
    'horizon_targets',
    'window_samples',
]

_SEQUENCE_DIMENSIONS = 3


def _require_at_least_one(*, value: int, setting: str) -> None:
    if value < 1:
        message = f'{setting} must be at least 1, got {value}'
        raise ValueError(message)


def horizon_targets(*, series: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return the ``horizon`` steps that follow every position.

    Args:
        series: Shape ``(N, T, C)``.
        horizon: Steps to forecast.

    Returns:
        Shape ``(N, T - horizon, horizon, C)``; entry ``[n, t]`` holds steps
        ``t + 1 .. t + horizon``. An empty series gives an empty result.

    Raises:
        ValueError: If ``horizon`` is below 1 or not shorter than the series.
    """
    _require_at_least_one(value=horizon, setting='horizon')
    total_time = series.shape[1]
    if total_time == 0:
        return series[:, :0, :].unsqueeze(2).expand(-1, -1, horizon, -1)
    if horizon >= total_time:
        message = (
            f'horizon ({horizon}) must be shorter than the series ({total_time} steps); '
            'no forecasting samples can be made'
        )
        raise ValueError(message)
    positions = torch.arange(total_time - horizon, device=series.device)
    offsets = torch.arange(horizon, device=series.device)
    return series[:, positions[:, None] + offsets[None, :] + 1, :]


def window_samples(
    *, series: torch.Tensor, window_length: int, horizon: int, stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut input windows and the target steps that follow each of them.

    Every input window is strictly older than its targets, so even an encoder
    that looks both ways cannot see the future.

    Args:
        series: Shape ``(N, T, C)``.
        window_length: Steps per input window.
        horizon: Target steps after each window.
        stride: Steps between window starts.

    Returns:
        Inputs ``(N, W, window_length, C)`` and targets ``(N, W, horizon, C)``.

    Raises:
        ValueError: If a setting is below 1 or the series is too short.
    """
    _require_at_least_one(value=window_length, setting='window_length')
    _require_at_least_one(value=horizon, setting='horizon')
    _require_at_least_one(value=stride, setting='stride')
    total_time = series.shape[1]
    positions = range(window_length, total_time - horizon + 1, stride)
    if not positions:
        message = (
            f'series too short for windowing: {total_time} steps, but window_length + horizon '
            f'= {window_length + horizon}'
        )
        raise ValueError(message)
    inputs = torch.stack(tensors=[series[:, end - window_length : end] for end in positions], dim=1)
    targets = torch.stack(tensors=[series[:, end : end + horizon] for end in positions], dim=1)
    return inputs, targets


def _input_series(*, data: ForecastingSeries, series: Tensors | None) -> torch.Tensor:
    full = data.full_series()
    if series is None:
        return full.to(dtype=torch.float32)
    replacement = series.tensors['series']
    if replacement.shape != full.shape:
        message = (
            f'the replacement series has shape {tuple(replacement.shape)}, '
            f'but the data series has shape {tuple(full.shape)}'
        )
        raise ValueError(message)
    return replacement.to(dtype=torch.float32)


def _flatten_targets(*, targets: torch.Tensor) -> torch.Tensor:
    # (N, positions, horizon, channels) -> one row per position.
    return targets.reshape(-1, targets.shape[2] * targets.shape[3])


@stage(name='forecast_encode', output=Tensors)
def encode_series[M](  # noqa: PLR0913 # why: keyword-only public API; each input is fingerprinted
    *,
    run: Run,
    name: InstanceName | None = None,
    trainer: Trainer[M],
    model: M,
    checkpoint: Checkpoint,
    data: ForecastingSeries,
    encoder: Callable[[M], Encoder],
    series: Tensors | None = None,
) -> Tensors:
    """Encode every full series once, then cut out each split.

    ``series``, if given, is encoded instead of ``data.full_series()``: a
    ``Tensors`` with the key ``series`` and the same shape.

    Returns:
        ``<split>_sequence`` (representations ``(N, T_split, D)``) and
        ``<split>_series`` (target channels ``(N, T_split, C_target)``).

    Raises:
        TypeError: If the encoder has no ``encode_sequence``.
        ValueError: If the representations do not have one row per timestep, or
            ``series`` does not have the data's shape.
    """
    del run, name
    slices = checked_split_slices(data=data)
    inputs = _input_series(data=data, series=series)
    trainer.load(model=model, checkpoint=checkpoint)
    sequence_encoder = encoder(model)
    if not isinstance(sequence_encoder, SequenceEncoder):
        message = f'{type(sequence_encoder).__qualname__} has no encode_sequence(*, inputs)'
        raise TypeError(message)
    with torch.inference_mode():
        sequences = sequence_encoder.encode_sequence(inputs=inputs)
    if sequences.ndim != _SEQUENCE_DIMENSIONS or sequences.shape[:2] != inputs.shape[:2]:
        message = (
            f'encode_sequence returned shape {tuple(sequences.shape)} for series of shape '
            f'{tuple(inputs.shape)}; expected (N, T, D)'
        )
        raise ValueError(message)
    targets = inputs[..., list(data.info.target_channels)]
    tensors: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        tensors[f'{split}_sequence'] = sequences[:, slices[split]]
        tensors[f'{split}_series'] = targets[:, slices[split]]
    return Tensors(tensors=tensors)


@stage(name='forecast_samples', output=Tensors)
def horizon_samples(
    *, run: Run, name: InstanceName | None = None, sequences: Tensors, horizon: int
) -> Tensors:
    """Cut the forecasting samples of one horizon out of ``encode_series``'s result.

    The features at a position are its representation; the targets are the
    next ``horizon`` steps of the target channels.

    Returns:
        ``<split>_representations`` ``(rows, D)`` and ``<split>_targets``
        ``(rows, horizon * C_target)``.
    """
    del run, name
    tensors: dict[str, torch.Tensor] = {}
    for split in SPLITS:
        targets = horizon_targets(series=sequences.tensors[f'{split}_series'], horizon=horizon)
        features = sequences.tensors[f'{split}_sequence'][:, :-horizon]
        features = features.reshape(-1, features.shape[-1])
        check_representations(
            representations=features,
            expected_rows=targets.shape[0] * targets.shape[1],
            description=f'the {split} split, horizon {horizon}',
        )
        tensors[f'{split}_representations'] = features
        tensors[f'{split}_targets'] = _flatten_targets(targets=targets)
    return Tensors(tensors=tensors)


@stage(name='forecast_windows', output=Tensors)
def encode_windows[M](  # noqa: PLR0913 # why: keyword-only public API; each input is fingerprinted
    *,
    run: Run,
    name: InstanceName | None = None,
    trainer: Trainer[M],
    model: M,
    checkpoint: Checkpoint,
    data: ForecastingSeries,
    encoder: Callable[[M], Encoder],
    horizon: int,
    window_length: int,
    stride: int = 1,
    series: Tensors | None = None,
) -> Tensors:
    """Encode sliding windows of every split and pair them with the steps that follow.

    ``series``, if given, is encoded instead of ``data.full_series()``, as in
    ``encode_series``.

    Returns:
        ``<split>_representations`` ``(rows, D)`` and ``<split>_targets``
        ``(rows, horizon * C_target)``, one row per window.
    """
    del run, name
    slices = checked_split_slices(data=data)
    inputs = _input_series(data=data, series=series)
    windows = {
        split: window_samples(
            series=inputs[:, slices[split]],
            window_length=window_length,
            horizon=horizon,
            stride=stride,
        )
        for split in SPLITS
    }
    trainer.load(model=model, checkpoint=checkpoint)
    window_encoder = encoder(model)
    target_channels = list(data.info.target_channels)
    tensors: dict[str, torch.Tensor] = {}
    for split, (inputs, targets) in windows.items():
        flat_inputs = inputs.reshape(-1, *inputs.shape[-2:])
        with torch.inference_mode():
            features = window_encoder.encode(inputs=flat_inputs)
        check_representations(
            representations=features,
            expected_rows=len(flat_inputs),
            description=f'the {split} split, horizon {horizon}',
        )
        tensors[f'{split}_representations'] = features
        tensors[f'{split}_targets'] = _flatten_targets(targets=targets[..., target_channels])
    return Tensors(tensors=tensors)


def forecasting_representations[M](  # noqa: PLR0913 # why: keyword-only public API
    *,
    run: Run,
    trainer: Trainer[M],
    model: M,
    checkpoint: Checkpoint,
    data: ForecastingSeries,
    encoder: Callable[[M], Encoder],
    horizons: Sequence[int],
    window_length: int | None = None,
    stride: int = 1,
    name: InstanceName | None = None,
    series: Tensors | None = None,
) -> dict[int, Tensors]:
    """Return evaluation-ready representations for every horizon.

    Encoders with ``encode_sequence`` encode each full series once
    (``forecast_encode``) and every horizon is sliced from it
    (``forecast_samples``). Other encoders encode windows of
    ``window_length`` steps per horizon (``forecast_windows``).

    Args:
        run: The current run.
        trainer: Loads the weights into ``model``.
        model: A model of the trained architecture.
        checkpoint: From ``train``, or ``tslab.Checkpoint.external(...)``.
        data: The full series and its split boundaries.
        encoder: Builds the encoder from the loaded model.
        horizons: Steps to forecast, one result each.
        window_length: Input window length; required for encoders without
            ``encode_sequence``.
        stride: Steps between windows (window path only).
        name: Prefix of the instance names, for several forecasting setups
            in one run.
        series: Encoded instead of ``data.full_series()``; a ``Tensors`` with
            the key ``series`` and the same shape, for example from an attack
            stage. If the stage that made it failed, every result is skipped.

    Returns:
        Horizon to representations and targets.

    Raises:
        ValueError: If ``horizons`` is empty or repeats a value, or a window
            length is needed but missing.
    """
    if not horizons or len(set(horizons)) != len(horizons):
        message = f'horizons must be distinct and non-empty, got {list(horizons)}'
        raise ValueError(message)
    prefix = () if name is None else normalize_instance_name(name=name)
    if isinstance(encoder(model), SequenceEncoder):
        sequences = encode_series(
            run=run,
            name=name,
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=encoder,
            series=series,
        )
        return {
            horizon: horizon_samples(
                run=run, name=(*prefix, f'h{horizon}'), sequences=sequences, horizon=horizon
            )
            for horizon in horizons
        }
    if window_length is None:
        message = (
            'this encoder has no encode_sequence, so forecasting uses input windows; '
            'pass window_length='
        )
        raise ValueError(message)
    return {
        horizon: encode_windows(
            run=run,
            name=(*prefix, f'h{horizon}'),
            trainer=trainer,
            model=model,
            checkpoint=checkpoint,
            data=data,
            encoder=encoder,
            horizon=horizon,
            window_length=window_length,
            stride=stride,
            series=series,
        )
        for horizon in horizons
    }
