# TSLab

**A batteries-included lab for research experiments, with torch and time series on top.**
TSLab turns a research script into a pipeline of **cached, resumable stages**.
Keep writing a normal Python `main()`. TSLab saves each stage result, so a
second run reuses finished work and reruns only steps affected by a change.

TSLab is one distribution with three layers. Each layer builds on the one before, and each has an install extra (the base install is the lab):

| Layer | Package | What it is | Needs torch |
|---|---|---|---|
| Lab | `tslab.lab` | runs, stages, artifacts, caching, studies, SLURM jobs, command line | no |
| ML | `tslab.ml` | the Lightning trainer, `train` stage, `Tensors`, `Checkpoint`, trackers | yes |
| TS | `tslab.ts` | encoders, probes, forecasting and classification stages, chronocratic adapters | yes |

Install with `uv add tslab` (lab), `uv add "tslab[ml]"`, `uv add "tslab[ts]"` or
`uv add "tslab[full]"`. Extras combine, for example `"tslab[full,wandb]"`; the
README lists every extra.

## Which section to read

- **Lab:** [Getting started with the lab](lab/getting-started.md) and the
  [lab guide](lab/guide.md). Read this for any experiment pipeline, with or
  without torch.
- **ML:** [ml](ml/index.md) for Lightning training, trackers and checkpoints.
- **TS:** [ts](ts/index.md) and [Getting started with time series](ts/getting-started.md)
  for the time series project template.

## How a run works

Your script chooses the order of stages. Each stage call saves an artifact and a
fingerprint of its settings and inputs. Run the script again to load matching
results. Change a setting and TSLab recomputes the affected stages and their
downstream work, not the whole run. If a stage fails, inspect the run and
continue from the work already saved. See [Concepts](lab/concepts.md) and
[Resume and failures](lab/resume-and-failures.md).

## Seeding

The lab seeds Python `random` and NumPy. Importing `tslab.ml` also seeds torch and
records torch, Lightning, CUDA and GPU metadata. See [Seeding](lab/run-options.md#seeding).

```{toctree}
:maxdepth: 2
:hidden:

lab/guide
ml/index
ts/index
reference/index
```
