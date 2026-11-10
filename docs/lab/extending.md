# Extending the lab

Everything the lab offers ready-made has a small interface you can implement.
Trainers and data sources are in [Extending ml](../ml/extending.md). The other
layers are in [Extending ts](../ts/extending.md).

## A stage

```python
@tslab.lab.stage(name='measure', output=tslab.lab.Metrics, version=1)
def measure(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,
    values: tslab.lab.Json,
    scale: float,
) -> tslab.lab.Metrics: ...
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
not skip the stage. Pass it as its own argument instead.

A stage that uses torch must import `tslab.ml`, so the run seeds torch and
records the torch metadata. See [Seeding](run-options.md#seeding).

## An artifact

Subclass `tslab.lab.Artifact`, set `FILE_NAME`, and implement `save(*, directory)`
and the class method `_from_file(cls, *, path)`.

## A tracker

Subclass `tslab.lab.Tracker`: `start`, `log_params`, `log_metrics`, `log_file`,
`finish`. Pass instances to `Run(trackers=...)`.
