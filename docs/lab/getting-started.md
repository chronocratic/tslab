# Getting started with the lab

The lab needs Python 3.12 or newer and no torch: `uv add tslab` installs only the lab.
With [uv](https://docs.astral.sh/uv/):

```bash
uvx tslab new my-experiments --template lab
cd my-experiments
uv sync
uv run python -m my_experiments.experiment --scale 2.0
```

`tslab new --template lab` copies a small project you own from then on. The
default template is `ts` when torch is installed and `lab` otherwise (see [the ts getting started page](../ts/getting-started.md)).

| File | What it is |
|---|---|
| `src/my_experiments/stages.py` | an example stage, `measure`, that returns `Metrics` |
| `src/my_experiments/experiment.py` | the config and one run that calls the stage |
| `src/my_experiments/studies/` | one package per study (see [Studies and HPC](studies.md)) |
| `machines/` | a local and an example SLURM machine profile |
| `tests/test_smoke.py` | runs the script once and dry-runs the `example` study |

## A first stage and run

A stage is a function with keyword-only arguments. The decorator caches its result
by a fingerprint of its settings and inputs:

```python
import numpy as np

import tslab.lab


@tslab.lab.stage(name='measure', output=tslab.lab.Metrics, version=1)
def measure(*, run: tslab.lab.Run, scale: float) -> tslab.lab.Metrics:
    rng = np.random.default_rng(run.seed)
    return tslab.lab.Metrics(data={'value': scale * float(rng.random())})
```

A run owns a folder, seeds the random number generators and calls the stages:

```python
with tslab.lab.Run(config=config, root=Path('runs'), relative_path=Path('experiment')) as run:
    measure(run=run, scale=config.scale)
```

Run the script again: the stage loads its result. Change `--scale` and only the
affected stage runs again. The lab seeds Python `random` and NumPy; for torch, see
[Seeding](run-options.md#seeding).

## Run a study

```bash
uv run tslab study submit example --machine local --dry-run
uv run tslab study submit example --machine local
uv run tslab status runs/
```

See [Concepts](concepts.md) for what is cached and why, and
[Studies and HPC](studies.md) for grids on a laptop or a cluster.
