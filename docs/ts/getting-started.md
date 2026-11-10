# Getting started with time series

This page uses the `ts` project template. For a general project without time
series code, see [Getting started with the lab](../lab/getting-started.md).

## Install

TSLab needs Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv add "tslab[chronocratic-models,chronocratic-datasets]"
```

The extras add the chronocratic adapters. Trackers are extras too:
`tslab[wandb]`, `tslab[mlflow]`, `tslab[tensorboard]`. `tslab` alone needs
none of them.

## Start a project

```bash
uvx tslab new my-paper --template ts
cd my-paper
uv sync
uv run python -m my_paper.toy_data data
```

`--template ts` is the default, so `tslab new my-paper` gives the same project.
It copies a small project you own from then on:

| File | What it is |
|---|---|
| `src/my_paper/classify.py` | train, encode, evaluate an SVM probe; plus a noisy-test robustness check |
| `src/my_paper/forecast.py` | train, encode the series, evaluate a ridge probe per horizon |
| `src/my_paper/stages.py` | an example custom stage, `add_noise` |
| `src/my_paper/augmentations.py` | an example augmentation recipe |
| `src/my_paper/models.py` | the model choice shared by the scripts |
| `src/my_paper/studies/` | one package per study; `common/` holds shared grids (see [Studies and HPC](../lab/studies.md)) |
| `machines/` | a local and an example SLURM machine profile |
| `src/my_paper/toy_data.py` | writes small synthetic datasets, so everything runs before you have real data |
| `tests/test_smoke.py` | runs both scripts on toy data and dry-runs the `example` and `quick` studies |

## Run

```bash
uv run python -m my_paper.classify ts2vec-archive-split
uv run python -m my_paper.forecast ts2vec-full-series
uv run tslab status runs/
```

Add `--watch SECONDS` to refresh run status until Ctrl-C. Status hints suggest
the next command; open a failed run to see its `error.txt` path.

Run the same command again: every stage loads its result. Change a setting
and only the stages it affects run again. See [Configs and presets](../lab/configs-and-presets.md) for
choosing models and datasets, and [Concepts](../lab/concepts.md) for what is cached and why.
