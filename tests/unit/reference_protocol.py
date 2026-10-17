"""A second, independent implementation of the forecasting samples.

The tests compare ``tslab.ts.forecasting`` with this straightforward version,
which builds the samples step by step, and require identical tensors.
"""

from collections.abc import Callable

import torch
from torch import Tensor


def generate_forecasting_samples(
    *, partition_data: Tensor, forecast_horizon: int
) -> tuple[Tensor, Tensor]:
    total_time = partition_data.shape[1]
    if total_time == 0:
        empty_inputs = partition_data
        empty_labels = partition_data[:, :0, :].unsqueeze(2).expand(-1, -1, forecast_horizon, -1)
        return empty_inputs, empty_labels
    if forecast_horizon >= total_time:
        msg = (
            f'forecast_horizon ({forecast_horizon}) >= partition time steps '
            f'({total_time}). No valid samples can be generated.'
        )
        raise ValueError(msg)
    inputs = partition_data[:, :-forecast_horizon]
    n_valid = total_time - forecast_horizon
    pos = torch.arange(n_valid, device=partition_data.device)
    offset = torch.arange(forecast_horizon, device=partition_data.device)
    index = pos[:, None] + offset[None, :]
    labels = partition_data[:, index + 1, :]
    return inputs, labels


def build_windowed_forecasting_samples(
    *, series: Tensor, forecast_window_length: int, horizon: int, stride: int = 1
) -> tuple[Tensor, Tensor]:
    if forecast_window_length < 1 or horizon < 1 or stride < 1:
        msg = (
            f'forecast_window_length/horizon/stride must be >= 1, '
            f'got {forecast_window_length=}, {horizon=}, {stride=}'
        )
        raise ValueError(msg)
    total_time = series.shape[1]
    positions = list(range(forecast_window_length, total_time - horizon + 1, stride))
    if not positions:
        msg = (
            f'Series too short for windowing: T={total_time}, need at least '
            f'forecast_window_length+horizon={forecast_window_length + horizon}. No valid windows.'
        )
        raise ValueError(msg)
    inputs = torch.stack([series[:, t - forecast_window_length : t] for t in positions], dim=1)
    targets = torch.stack([series[:, t : t + horizon] for t in positions], dim=1)
    return inputs, targets


def dilated_reference(
    *,
    full_data: Tensor,
    slices: dict[str, slice],
    num_ts_features: int,
    forecast_horizon: int,
    encode_sequence: Callable[[Tensor], Tensor],
) -> dict[str, tuple[Tensor, Tensor]]:
    combined_representations = encode_sequence(full_data)
    result = {}
    for partition_name, part_slice in slices.items():
        partition_data = full_data[:, part_slice, num_ts_features:]
        partition_reps = combined_representations[:, part_slice]
        encoded_features = partition_reps[:, :-forecast_horizon]
        _, labels = generate_forecasting_samples(
            partition_data=partition_data, forecast_horizon=forecast_horizon
        )
        features = encoded_features.reshape(-1, encoded_features.shape[-1])
        flat_labels = labels.reshape(-1, labels.shape[2] * labels.shape[3])
        result[partition_name] = (features, flat_labels)
    return result


def windowed_reference(  # noqa: PLR0913
    *,
    full_data: Tensor,
    slices: dict[str, slice],
    num_ts_features: int,
    forecast_horizon: int,
    forecast_window_length: int,
    stride: int,
    encode: Callable[[Tensor], Tensor],
) -> dict[str, tuple[Tensor, Tensor]]:
    result = {}
    for partition_name, part_slice in slices.items():
        inputs, targets_full = build_windowed_forecasting_samples(
            series=full_data[:, part_slice],
            forecast_window_length=forecast_window_length,
            horizon=forecast_horizon,
            stride=stride,
        )
        targets = targets_full[..., num_ts_features:]
        n, n_win = inputs.shape[0], inputs.shape[1]
        reps = encode(inputs.reshape(n * n_win, *inputs.shape[-2:])).reshape(n, n_win, -1)
        features = reps.reshape(-1, reps.shape[-1])
        flat_labels = targets.reshape(-1, targets.shape[2] * targets.shape[3])
        result[partition_name] = (features, flat_labels)
    return result
