# Extending ts

For stages, artifacts and trackers, see [Extending the lab](../lab/extending.md).
For trainers and data sources, see [Extending ml](../ml/extending.md).

## A stage that changes the series

Return a `tslab.ml.Tensors` with the key `series` and pass it to
`tslab.ts.tasks.forecasting_representations(..., series=...)`. `examples/advanced`
does this.

## A forecasting data source

Forecasting data offers `full_series()` and `split_slices()`
(`tslab.ts.tasks.ForecastingSeries`).

## An encoder

`encode` takes an `encoder` argument: a callable that turns the trained model
into an object with `encode(*, inputs)` (`tslab.ts.Encoder`), and optionally
`encode_sequence(*, inputs)` (`tslab.ts.SequenceEncoder`) for the
encode-once forecasting path. `tslab.ts.stages.ModelAsEncoder()` uses the methods
of the model under those names.

## A probe or downstream model

Subclass `tslab.ts.DownstreamModel` (`fit`, `predict`, `default_metrics`) or
`tslab.ts.Probe`. `tslab.ts.probes.SVMProbe` and `tslab.ts.probes.RidgeProbe` follow
the TS2Vec evaluation protocols.
