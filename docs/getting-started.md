# Getting started

## Install

TSLab needs Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv add "tslab[chronocratic-models,chronocratic-datasets]"
```

The extras add the chronocratic adapters. Trackers are extras too:
`tslab[wandb]`, `tslab[mlflow]`, `tslab[tensorboard]`. The core
(`tslab` alone) needs none of them.

## Start a project

```bash
uv run tslab new my-paper
cd my-paper
uv sync
uv run python -m my_paper.toy_data data
```

`tslab new` copies a small project you own from then on:

| File | What it is |
|---|---|
| `src/my_paper/classify.py` | train, encode, evaluate an SVM probe; plus a noisy-test robustness check |
| `src/my_paper/forecast.py` | train, encode the series, evaluate a ridge probe per horizon |
| `src/my_paper/stages.py` | an example custom stage, `add_noise` |
| `src/my_paper/augmentations.py` | an example augmentation recipe |
| `src/my_paper/toy_data.py` | writes small synthetic datasets, so everything runs before you have real data |
| `tests/test_smoke.py` | runs both scripts on the toy data with one training batch |

## Run

```bash
uv run python -m my_paper.classify ts2vec-archive-split
uv run python -m my_paper.forecast ts2vec-full-series
uv run tslab status runs/
```

Run the same command again: every stage loads its result. Change a setting
and only the stages it affects run again. See {doc}`configs-and-presets` for
choosing models and datasets, and {doc}`concepts` for what is cached and why.
