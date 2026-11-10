# Concepts

## Run, stage, instance

- A **run** is one execution context:
  `with tslab.lab.Run(config=..., root=..., relative_path=...) as run:`.
  It owns a folder `root / relative_path`, takes a lock on it, seeds Python,
  NumPy and torch, writes `config.yaml`, `meta.json` and `status.json`, and
  starts the trackers. Leaving the `with` block always records the final state.
- A **stage** is a cached step: a function decorated with `@tslab.lab.stage`, or a
  ready one such as `tslab.ml.stages.train`. Stages are ordinary callables; your
  script decides their order, loops and conditions.
- An **instance** is one call of a stage in a run. A stage called several times
  gives each call a name: `transform(run=run, name=('forecasting', 'h24', 'smooth'), ...)`
  is the instance `transform[forecasting/h24/smooth]`, stored in
  `transform/forecasting/h24/smooth/`. Names may nest in parts, but within one
  stage no name may be a prefix of another.

## Fingerprints

Every instance has a fingerprint: 16 hexadecimal characters of a SHA-256 over

- the stage name and its `version`,
- the seed of the run,
- the settings it was called with (numbers, strings, paths, enums,
  dataclasses, and objects that offer `fingerprint_settings()`),
- the fingerprints of the artifacts it received.

When the stored fingerprint matches, the result is loaded. Anything else runs
the instance again, and everything downstream follows because the
fingerprints of its inputs changed.

Fields annotated `Annotated[..., tslab.lab.NotFingerprinted]` (device, number of
workers, logging intervals) never change a fingerprint: they cannot change the
result. Paths are fingerprinted as paths, not by content; use
`Artifact.external(path=...)` for a file whose content must count.

Bump the `version` of a stage after changing its code, so results of the old
code are not reused.

## Artifacts

A stage returns an **artifact**, saved atomically in the instance folder:

| Artifact | Stores |
|---|---|
| `tslab.lab.Json` | a JSON document |
| `tslab.lab.Metrics` | named numbers (`metrics.values`), also sent to the trackers |
| `tslab.ml.Tensors` | named tensors, with safetensors (no pickle) |
| `tslab.ml.Checkpoint` | the best and last Lightning checkpoints of `train` |

`SomeArtifact.external(path=...)` brings an outside file into the pipeline,
fingerprinted by its content, for example a pretrained checkpoint.

## The run folder

```
runs/ts2vec/Coffee/seed_0/
  config.yaml        the parsed config; reload it with tslab.lab.load_config
  meta.json          versions, git commit, hardware, host, config hashes
  status.json        the state of every instance
  logs/run.log       everything logged during the run
  trackers/csv/      always-on CSV metrics
  train/default/     checkpoints
  encode/default/    representations
  evaluate/default/  metrics
```
