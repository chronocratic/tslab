<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/header/surface.svg?title=TSLab&subtitle=A+batteries-included+lab+for+time+series+research&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIiByb2xlPSJpbWciIGFyaWEtbGFiZWw9IlRTTGFiIj4KICA8dGl0bGU%2BVFNMYWI8L3RpdGxlPgogIDxwb2x5bGluZSBwb2ludHM9IjIsNjAgMjAsNjAgMzIsMjIgNTAsODYgNjYsMTAgNzgsNTAgOTgsNTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBmNzY2ZSIgc3Ryb2tlLXdpZHRoPSI5IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc%2BCg%3D%3D&align=center&font=jetbrains-mono&logoColor=5eead4&bg=0c1015&titleColor=5eead4&subtitleColor=b0b5bd&mode=dark" />
    <img alt="TSLab: a batteries-included lab for time series research" src="https://shieldcn.dev/header/surface.svg?title=TSLab&subtitle=A+batteries-included+lab+for+time+series+research&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIiByb2xlPSJpbWciIGFyaWEtbGFiZWw9IlRTTGFiIj4KICA8dGl0bGU%2BVFNMYWI8L3RpdGxlPgogIDxwb2x5bGluZSBwb2ludHM9IjIsNjAgMjAsNjAgMzIsMjIgNTAsODYgNjYsMTAgNzgsNTAgOTgsNTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBmNzY2ZSIgc3Ryb2tlLXdpZHRoPSI5IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc%2BCg%3D%3D&align=center&font=jetbrains-mono&logoColor=0f766e&bg=fcfdff&titleColor=0f766e&subtitleColor=61666e&mode=light" />
  </picture>
</p>

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
import tslab.lab


@tslab.lab.stage(name='smooth', output=tslab.ml.Tensors)
def smooth(*, run: tslab.lab.Run, data: tslab.ml.Tensors, window: int) -> tslab.ml.Tensors: ...


with tslab.lab.Run(config=config, root=Path('runs'), relative_path=Path('minimal')) as run:
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

One distribution, three layers: `tslab.lab` (runs, stages, studies, jobs; no
torch), `tslab.ml` (torch and Lightning) and `tslab.ts` (time series). The lab
seeds Python `random` and NumPy; importing `tslab.ml` adds torch seeding and
torch metadata.

## Install

Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv add tslab                  # lab only, no torch
uv add "tslab[ml]"            # plus torch, Lightning, safetensors
uv add "tslab[ts]"            # plus scikit-learn, for the time series layer
uv add "tslab[full]"          # ts, both chronocratic adapters and analysis
uv add "tslab[full,wandb]"    # extras combine
```

| Extra | Adds |
|---|---|
| `ml` | torch, Lightning, safetensors |
| `ts` | `ml` and scikit-learn |
| `chronocratic-models`, `chronocratic-datasets` | `ts` and the chronocratic adapter packages |
| `wandb`, `mlflow`, `tensorboard` | `ml` and the tracker |
| `wandb-sync` | W&B alone, to run `tslab wandb-sync` on a login node without torch |
| `analysis` | pandas, SciPy, Matplotlib |
| `full` | `ts`, both chronocratic adapters and `analysis` |

## Start a project

```bash
uvx tslab new my-paper                  # the ts template; add --template lab for a bare project
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
  transform stage, a failing instance and selective reruns.

## Documentation

[tslab.readthedocs.io](https://tslab.readthedocs.io): three sections, one per
layer (`tslab.lab` without torch, `tslab.ml`, `tslab.ts`), plus the API and a
migration page for older code.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
