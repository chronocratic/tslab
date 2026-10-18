from pathlib import Path

import pytest
from reference_protocol import (
    build_windowed_forecasting_samples,
    dilated_reference,
    generate_forecasting_samples,
    windowed_reference,
)
from stage_fakes import CausalEncoder, FakeTrainer, run_in, ToySeries, VectorEncoder
import torch

import tslab
from tslab.tasks import forecasting_representations, horizon_targets, window_samples

SLICES = {'train': slice(0, 24), 'val': slice(24, 32), 'test': slice(32, 40)}


def representations(  # noqa: PLR0913
    *,
    run: tslab.Run,
    model: VectorEncoder,
    horizons: tuple[int, ...],
    window_length: int | None = None,
    name: str | None = None,
    series: tslab.Tensors | None = None,
) -> dict[int, tslab.Tensors]:
    trainer = FakeTrainer()
    checkpoint = tslab.stages.train(run=run, trainer=trainer, model=model, data=ToySeries())
    return forecasting_representations(
        run=run,
        trainer=trainer,
        model=model,
        checkpoint=checkpoint,
        data=ToySeries(),
        encoder=tslab.ModelAsEncoder(device='cpu'),
        horizons=horizons,
        window_length=window_length,
        stride=2,
        name=name,
        series=series,
    )


@tslab.stage(name='broken_series', output=tslab.Tensors)
def broken_series(*, run: tslab.Run) -> tslab.Tensors:
    del run
    message = 'broken on purpose'
    raise RuntimeError(message)


def test_horizon_targets_match_reference() -> None:
    series = torch.randn(2, 12, 3, generator=torch.Generator().manual_seed(0))
    for horizon in range(1, 6):
        _, expected = generate_forecasting_samples(partition_data=series, forecast_horizon=horizon)
        assert torch.equal(horizon_targets(series=series, horizon=horizon), expected)


def test_horizon_longer_than_the_series_raises() -> None:
    series = torch.zeros(1, 12, 1)
    with pytest.raises(ValueError, match=r'horizon \(12\)'):
        horizon_targets(series=series, horizon=12)
    with pytest.raises(ValueError, match='at least 1'):
        horizon_targets(series=series, horizon=0)


def test_window_samples_match_reference() -> None:
    series = torch.randn(2, 20, 3, generator=torch.Generator().manual_seed(0))
    inputs, targets = window_samples(series=series, window_length=4, horizon=3, stride=2)
    expected_inputs, expected_targets = build_windowed_forecasting_samples(
        series=series, forecast_window_length=4, horizon=3, stride=2
    )
    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(targets, expected_targets)


def test_window_longer_than_the_series_raises() -> None:
    with pytest.raises(ValueError, match='too short'):
        window_samples(series=torch.zeros(1, 10, 1), window_length=8, horizon=3)


def test_dilated_encoders_encode_once_for_all_horizons(tmp_path: Path) -> None:
    model = CausalEncoder()
    with run_in(root=tmp_path) as run:
        results = representations(run=run, model=model, horizons=(2, 3, 5))
        instances = set(run.status().instances)
    assert model.sequence_calls == 1
    assert sorted(results) == [2, 3, 5]
    assert {'forecast_encode[default]', 'forecast_samples[h2]', 'forecast_samples[h5]'} <= instances


def test_dilated_path_matches_reference(tmp_path: Path) -> None:
    model = CausalEncoder()
    with run_in(root=tmp_path) as run:
        results = representations(run=run, model=model, horizons=(2, 5))
    for horizon, result in results.items():
        expected = dilated_reference(
            full_data=ToySeries().full_series(),
            slices=SLICES,
            num_ts_features=1,
            forecast_horizon=horizon,
            encode_sequence=lambda data: model.encode_sequence(inputs=data),
        )
        for split, (features, labels) in expected.items():
            assert torch.equal(result.tensors[f'{split}_representations'], features)
            assert torch.equal(result.tensors[f'{split}_targets'], labels)


def test_windowed_path_matches_reference(tmp_path: Path) -> None:
    model = VectorEncoder()
    with run_in(root=tmp_path) as run:
        results = representations(run=run, model=model, horizons=(2, 3), window_length=4)
        instances = set(run.status().instances)
    assert {'forecast_windows[h2]', 'forecast_windows[h3]'} <= instances
    for horizon, result in results.items():
        expected = windowed_reference(
            full_data=ToySeries().full_series(),
            slices=SLICES,
            num_ts_features=1,
            forecast_horizon=horizon,
            forecast_window_length=4,
            stride=2,
            encode=lambda data: model.encode(inputs=data),
        )
        for split, (features, labels) in expected.items():
            assert torch.equal(result.tensors[f'{split}_representations'], features)
            assert torch.equal(result.tensors[f'{split}_targets'], labels)


def test_vector_encoders_need_a_window_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='window_length'), run_in(root=tmp_path) as run:
        representations(run=run, model=VectorEncoder(), horizons=(2,))


def test_targets_hold_the_target_channels_only(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        result = representations(run=run, model=CausalEncoder(), horizons=(3,))[3]
    assert result.tensors['train_targets'].shape == (2 * (24 - 3), 3 * 2)


def test_names_nest_under_the_given_name(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        representations(run=run, model=CausalEncoder(), horizons=(2,), name='w')
        instances = set(run.status().instances)
    assert {'forecast_encode[w]', 'forecast_samples[w/h2]'} <= instances


def test_a_new_horizon_reuses_the_cached_encoding(tmp_path: Path) -> None:
    with run_in(root=tmp_path) as run:
        representations(run=run, model=CausalEncoder(), horizons=(2,))
    model = CausalEncoder()
    with run_in(root=tmp_path) as run:
        representations(run=run, model=model, horizons=(2, 3))
    assert model.sequence_calls == 0


def test_a_replacement_series_is_encoded_but_targets_stay_clean(tmp_path: Path) -> None:
    clean = ToySeries().full_series()
    shifted = tslab.Tensors(tensors={'series': clean + 1.0})
    model = CausalEncoder()
    with run_in(root=tmp_path) as run:
        result = representations(
            run=run, model=model, horizons=(2,), name='shifted', series=shifted
        )
    encoded = dilated_reference(
        full_data=shifted.tensors['series'],
        slices=SLICES,
        num_ts_features=1,
        forecast_horizon=2,
        encode_sequence=lambda data: model.encode_sequence(inputs=data),
    )
    truth = dilated_reference(
        full_data=clean,
        slices=SLICES,
        num_ts_features=1,
        forecast_horizon=2,
        encode_sequence=lambda data: model.encode_sequence(inputs=data),
    )
    for split in SLICES:
        assert torch.equal(result[2].tensors[f'{split}_representations'], encoded[split][0])
        assert torch.equal(result[2].tensors[f'{split}_targets'], truth[split][1])


def test_a_replacement_series_keeps_clean_window_targets(tmp_path: Path) -> None:
    clean = ToySeries().full_series()
    shifted = tslab.Tensors(tensors={'series': clean + 1.0})
    model = VectorEncoder()
    with run_in(root=tmp_path) as run:
        result = representations(
            run=run, model=model, horizons=(2,), window_length=4, series=shifted
        )

    def reference(full_data: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return windowed_reference(
            full_data=full_data,
            slices=SLICES,
            num_ts_features=1,
            forecast_horizon=2,
            forecast_window_length=4,
            stride=2,
            encode=lambda data: model.encode(inputs=data),
        )

    encoded, truth = reference(shifted.tensors['series']), reference(clean)
    for split in SLICES:
        assert torch.equal(result[2].tensors[f'{split}_representations'], encoded[split][0])
        assert torch.equal(result[2].tensors[f'{split}_targets'], truth[split][1])


def test_a_replacement_series_must_match_the_data_shape(tmp_path: Path) -> None:
    short = tslab.Tensors(tensors={'series': ToySeries().full_series()[:, :30]})
    with pytest.raises(ValueError, match=r'shape \(2, 30, 3\)'), run_in(root=tmp_path) as run:
        representations(
            run=run, model=VectorEncoder(), horizons=(2,), window_length=4, series=short
        )


def test_a_failed_replacement_series_skips_the_encoding(tmp_path: Path) -> None:
    with run_in(root=tmp_path, fail_fast=False) as run:
        series = broken_series(run=run)
        results = representations(run=run, model=CausalEncoder(), horizons=(2,), series=series)
        states = {key: record.state for key, record in run.status().instances.items()}
    assert not tslab.is_available(artifact=results[2])
    assert states['forecast_encode[default]'] == tslab.InstanceState.SKIPPED
    assert states['forecast_samples[h2]'] == tslab.InstanceState.SKIPPED
