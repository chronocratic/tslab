<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/header/surface.svg?title=TSLab&subtitle=A+batteries-included+lab+for+time+series+research.&align=center&font=jetbrains-mono&width=1200&height=600&bg=0E2A44&titleColor=F3F1EC&subtitleColor=A8D5D1&mode=dark&overlay=0&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fchronocratic%2Ftslab%2F9ae637acdb2d074a84bafe629987f45c77cf24b4%2F.github%2Fassets%2Flogo-dark.png&image=https%3A%2F%2Fraw.githubusercontent.com%2Fchronocratic%2Fchronocratic-models%2Fmain%2F.github%2Fassets%2Fheader-bg-dark.png&v=2" />
    <img alt="TSLab: a batteries-included lab for time series research" src="https://shieldcn.dev/header/surface.svg?title=TSLab&subtitle=A+batteries-included+lab+for+time+series+research.&align=center&font=jetbrains-mono&width=1200&height=600&bg=F3F1EC&titleColor=0E2A44&subtitleColor=2A6A7C&mode=light&overlay=0&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fchronocratic%2Ftslab%2F9ae637acdb2d074a84bafe629987f45c77cf24b4%2F.github%2Fassets%2Flogo-light.png&image=https%3A%2F%2Fraw.githubusercontent.com%2Fchronocratic%2Fchronocratic-models%2Fmain%2F.github%2Fassets%2Fheader-bg-light.png&v=2" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/tslab/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/pypi/tslab.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="PyPI version" src="https://shieldcn.dev/pypi/tslab.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://pepy.tech/projects/tslab"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fimg.shields.io%2Fpepy%2Fdt%2Ftslab.json&query=%24.message&label=downloads&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="Total downloads" src="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fimg.shields.io%2Fpepy%2Fdt%2Ftslab.json&query=%24.message&label=downloads&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/tslab/actions/workflows/ci.yml"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/ci/chronocratic/tslab.svg?workflow=ci.yml&branch=main&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="CI status" src="https://shieldcn.dev/github/ci/chronocratic/tslab.svg?workflow=ci.yml&branch=main&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/tslab/blob/main/LICENSE"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/license/chronocratic/tslab.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="License" src="https://shieldcn.dev/github/license/chronocratic/tslab.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/tslab/stargazers"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/stars/chronocratic/tslab.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="GitHub stars" src="https://shieldcn.dev/github/stars/chronocratic/tslab.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/astral-sh/ruff"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/code_style-ruff-261230.svg?logo=ruff&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="Code style: ruff" src="https://shieldcn.dev/badge/code_style-ruff-261230.svg?logo=ruff&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
</p>

<div align="center">

| Python | Docs |
| :---: | :---: |
| [![Python versions](https://img.shields.io/pypi/pyversions/tslab.svg)](https://pypi.org/project/tslab/) | [![Documentation status](https://readthedocs.org/projects/tslab/badge/?version=latest)](https://tslab.readthedocs.io/en/latest/) |

</div>

> [!WARNING]
> TSLab is under active development. Expect breaking changes between releases until the API is declared stable in v1.0.0.

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
- a **CLI** (`tslab`): project templates, run overviews, and experiment
  grids on SLURM or locally (see [CLI](#cli)).

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

## Quick start

Create a project from the template, then train and evaluate on toy data:

```bash
uvx tslab new my-paper                            # add --template lab for a bare project
cd my-paper
uv sync
uv run python -m my_paper.toy_data data           # write small synthetic datasets
uv run python -m my_paper.classify ts2vec-archive-split
uv run tslab status runs                          # see what ran and what was cached
```

Run the `classify` line again: every stage loads instead of computing. The
generated `README.md` explains the rest.

> ⭐ **Found this useful? Give it a star.** TSLab is free and open source. A star takes one click, tells us the work is worth continuing, and helps other time series researchers find the project.

## CLI

TSLab ships with a CLI. Installing it puts a `tslab` command on your path
(or run it without installing via `uvx tslab`). Your experiments stay plain
Python scripts; the CLI creates projects, launches studies and shows what runs
are doing.

| Command | Does |
|---|---|
| `tslab new FOLDER` | Create a project from a template (`ts` by default, or `lab`). |
| `tslab status PATH` | Show the instances of a run, or one line per run below a folder. |
| `tslab study ...` | Run an experiment grid on SLURM or locally: `submit`, `status`, `wait`, `failures`, `logs`, `resubmit`, `results`. |
| `tslab machine ...` | Register machine profiles (local, SLURM clusters): `add`, `list`, `remove`. |
| `tslab doctor` | Check whether a machine, and optionally a study, is ready. |
| `tslab wandb-sync` | Resync offline W&B runs that the server shows as crashed or failed. |

A study, step by step:

```bash
tslab study submit STUDY --machine local --dry-run   # check the plan
tslab study submit STUDY --machine local             # launch the runs
tslab study status STUDY --machine local             # follow them
tslab study results STUDY --machine local --out results.csv
```

`tslab --help` and `tslab COMMAND --help` list every option.

## Examples

- [`examples/minimal`](examples/minimal/main.py): three plain stages, no deep
  learning; shows caching and instances.
- [`examples/advanced`](examples/advanced/main.py): TS2Vec with your own
  transform stage, a failing instance and selective reruns.

## Documentation

[tslab.readthedocs.io](https://tslab.readthedocs.io): three sections, one per
layer (`tslab.lab` without torch, `tslab.ml`, `tslab.ts`), plus the API.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
