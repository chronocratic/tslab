# Extending ml

## A trainer

`tslab.ml.Trainer[M]` has two methods:
`fit(*, run, model, data, checkpoint_directory) -> Checkpoint` and `load(*, model, checkpoint)`. `tslab.ml.lightning.LightningTrainer` is the
ready one.

## A data source

`tslab.ml.DataSource[I]` offers `info` and the train, validation and test
loaders. `tslab.ml.lightning.LightningDataSource` wraps a `LightningDataModule`.
For the ready stages, labeled data also implements
`labeled_split(*, split)` (`tslab.ml.LabeledSplits`).

## A tensor stage

A stage can return `tslab.ml.Tensors`. Importing `tslab.ml` is enough to get
torch seeding and torch metadata for the run; see [Seeding](index.md#seeding).
