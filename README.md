# TSLab

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/tslab.svg)](https://pypi.org/project/tslab/)
[![Python versions](https://img.shields.io/pypi/pyversions/tslab.svg)](https://pypi.org/project/tslab/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tslab?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/tslab)
[![Build Status](https://github.com/skafai/tslab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/skafai/tslab/actions)
[![Documentation Status](https://readthedocs.org/projects/tslab/badge/?version=latest)](https://tslab.readthedocs.io/en/latest/?badge=latest)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub stars](https://img.shields.io/github/stars/skafai/tslab)](https://github.com/skafai/tslab/stargazers)

TSLab turns a research script into a pipeline of **cached, resumable stages**.
You keep writing a normal Python `main()`. TSLab fingerprints the inputs and
settings of each step, stores its result, and on the next run loads finished
results, resumes unfinished training, and reruns only what changed or failed.

```python
import tslab


@tslab.stage(name='smooth', output=tslab.Tensors)
def smooth(*, run: tslab.Run, data: tslab.Tensors, window: int) -> tslab.Tensors: ...


with tslab.Run(config=config, root=Path('runs'), relative_path=Path('minimal')) as run:
    data = make_series(run=run, series=8, length=256)
    smoothed = smooth(run=run, data=data, window=9)  # loaded, not computed, on the next run
```

It also ships:

- ready stages for time series representation learning: train, encode, and
  evaluate with linear probes;
- Lightning training with safe resume and SLURM time limits;
- experiment trackers (CSV always; Weights & Biases, MLflow and TensorBoard as extras);
- typed command-line configs with presets;
- adapters for the chronocratic model and dataset libraries;
- `tslab new`, a project template; `tslab status`, an overview of runs;
  `tslab study`, experiment grids on SLURM or locally; and `tslab wandb-sync`,
  which uploads offline W&B runs.

## Install

Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv add tslab                                                 # core only
uv add "tslab[chronocratic-models,chronocratic-datasets]"    # with the adapters
```

Tracker extras: `tslab[wandb]`, `tslab[mlflow]`, `tslab[tensorboard]`.

## Start a project

```bash
uvx tslab new my-paper
cd my-paper
uv sync
uv run python -m my_paper.toy_data data
uv run python -m my_paper.classify ts2vec-archive-split
uv run tslab status runs
```

The generated `README.md` explains the rest.

## Examples

- [`examples/minimal`](examples/minimal/main.py): three plain stages, no deep
  learning; shows caching and instances.
- [`examples/advanced`](examples/advanced/main.py): TS2Vec with your own
  attack stage, a failing instance and selective reruns.

## Documentation

[tslab.readthedocs.io](https://tslab.readthedocs.io): concepts, resuming and
failures, configs and presets, studies, HPC, tracking, extending TSLab, and
the API.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
