# Extending TSLab

Everything TSLab offers ready-made has a small interface you can implement.

## A stage

```python
@tslab.stage(name='attack', output=tslab.Tensors, version=1)
def attack(
    *,
    run: tslab.Run,
    name: tslab.InstanceName | None = None,
    representations: tslab.Tensors,
    epsilon: float,
) -> tslab.Tensors: ...
```

Arguments are keyword-only. Artifacts among them are the inputs of the stage; the
rest are settings and must be fingerprintable. Omit `name` for a stage that
runs once per run. `in_place=True` lets a long stage write into its final
folder while running and resume from what it wrote.

An artifact you build yourself (not returned by a stage) has no fingerprint
and cannot be passed to a stage: compute it in a stage, or use
`Artifact.external(path=...)` for a file.

Only artifacts passed **directly** as arguments count as inputs. An artifact
wrapped inside another object is invisible to TSLab: a failure upstream would
not skip the stage. Pass it as its own argument instead. For a stage that
changes the series (an attack, a corruption), return a `Tensors` with the key
`series` and pass it to `tslab.ts.forecasting_representations(..., series=...)`;
`examples/advanced` does this.

## An artifact

Subclass `tslab.Artifact`, set `FILE_NAME`, and implement `save(*, directory)`
and the class method `_from_file(cls, *, path)`.

## A trainer

`tslab.interfaces.Trainer[M]` has two methods: `fit(*, run, model, data, checkpoint_directory) -> Checkpoint`
and `load(*, model, checkpoint)`. `tslab.lightning.LightningTrainer` is the
ready one.

## A data source

`tslab.interfaces.DataSource[I]` offers `info` and the train, validation and test
loaders. `tslab.lightning.LightningDataSource` wraps a `LightningDataModule`.
For the ready stages, classification data also implements
`labeled_split(*, split)` (`tslab.interfaces.LabeledSplits`), and forecasting data
`full_series()` and `split_slices()` (`tslab.ts.ForecastingSeries`).

## An encoder

`encode` takes an `encoder` argument: a callable that turns the trained model
into an object with `encode(*, inputs)` (`tslab.interfaces.Encoder`), and optionally
`encode_sequence(*, inputs)` (`tslab.interfaces.SequenceEncoder`) for the
encode-once forecasting path. `tslab.ModelAsEncoder()` uses the methods
of the model under those names.

## A probe or downstream model

Subclass `tslab.interfaces.DownstreamModel` (`fit`, `predict`, `default_metrics`) or
`tslab.interfaces.Probe`. `tslab.probes.SVMProbe` and `tslab.probes.RidgeProbe` follow
the TS2Vec evaluation protocols.

## A tracker

Subclass `tslab.Tracker`: `start`, `log_params`, `log_metrics`, `log_file`,
`finish`. Pass instances to `Run(trackers=...)`.
