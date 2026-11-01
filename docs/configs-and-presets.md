# Configs and presets

## A config is a frozen dataclass

Every field becomes a command-line flag through
[tyro](https://brentyi.github.io/tyro/): `--trainer.max-epochs 50`,
`--run-path ts2vec/Coffee/seed_0`. Union fields become subcommands:
`model:ts2vec`, `data:ucr`. `tslab.parse_config` reads the command line, and
`Run` writes the result to `config.yaml`.

```python
config = tslab.parse_config(config_class=ClassifyConfig, presets=PRESETS)
```

Flags may come before or after a subcommand: `model:ts2vec --seed 3` and
`--seed 3 model:ts2vec` are the same.

## Presets

With `presets=`, the first word picks where to start:

- `no-preset`: the config class itself; choose every model and dataset with
  subcommands and flags;
- a preset name: a finished config. A preset **fixes its model and dataset**
  (`my-preset model:cost` is an error, so the model settings of a preset are never
  silently dropped), while flags still change any value
  (`my-preset --trainer.max-epochs 10`).

### What a preset is

A preset is a **named, complete config for one run by hand**: every field
has a value, so the single command `classify ts2vec-paper` runs as is.
Completeness is deliberate:

- one command reproduces the run — the preset *is* the record of it;
- its model and dataset choices are fixed on the command line, so a
  mistaken extra subcommand is an error instead of a silently dropped
  setting;
- it names a dataset, so it runs without any other input. That is also why
  a study [drops the data part](#a-preset-as-a-study-recipe): a round chooses
  its datasets on its own axis.

### Building one

`model_settings` and `data_settings` build the generated choice members
(the settings classes behind `model:ts2vec`, `data:ucr` are generated and
cannot be imported). A real recipe sets several model and trainer values:

```python
from tslab.adapters.chronocratic.models import model_settings
from tslab.adapters.chronocratic.datasets import data_settings

PRESETS = {
    'ts2vec-paper': (
        'TS2Vec with the settings from its paper',
        ClassifyConfig(
            model=model_settings(choice=Models, name='ts2vec', depth=10, dropout_rate=0.2),
            data=data_settings(
                name='ucr', dataset_folder_path=Path('data/UCR/Coffee'), target_column_name='target'
            ),
            trainer=LightningTrainerConfig(max_epochs=600),
        ),
    )
}
```

Keyword values go straight into the chosen member; fields not given keep
their defaults.

### `tslab.REQUIRED`

A preset may put `tslab.REQUIRED` on a field its author cannot or does not
want to know — a dataset path that lives in a different folder on each
machine. The marker is **optional**; presets without it are unaffected.

- **By hand:** `classify ts2vec-any` without the value exits with the
  tyro error box — `Missing value for --data.dataset-folder-path` (exit code 2);
  with `--data.dataset-folder-path …` it runs.
- **In a study:** `from_presets` skips marked fields, so the study must
  supply that field on one of its axes (usually the dataset axis); otherwise
  the per-run parse at submit fails with the same message.
- A parsed config never holds the marker, so saved configs never contain it.

### A preset as a study recipe

`from_presets` turns presets into the settings of the recipe axis of a study. It
keeps each choice field as the **name** of its member and every other field
**only where it differs from the default**, plus every field without a
default. `ts2vec-paper` above gives:

```python
{'model': 'ts2vec', 'model.depth': 10, 'model.dropout_rate': 0.2, 'trainer.max_epochs': 600}
```

(with the dataset paths removed by `without=('data',)`). Each setting mapping
becomes flags of the `no-preset` command of the grid:

```bash
uv run python -m my_paper.classify no-preset model:ts2vec \
    --model.depth 10 --model.dropout-rate 0.2 --trainer.max-epochs 600 …
```

```python
'recipe': from_presets(presets=PRESETS, names=('ts2vec-paper', 'cost-paper'), without=('data',))
```

### Settings: how the values become flags

Settings are **dotted paths** of the config class (`model.depth`,
`trainer.max_epochs`) with values: `str`, `int`, `float`, `bool`, `Path`,
`Enum`, tuples and lists, or `None`.

- A path that ends at a **union field** with a string value becomes the
  subcommand word `<path>:<value>` — `model:ts2vec`, `data:ucr`. The names
  are the tyro subcommand names of the union; members of your own choice
  unions must be named with `tyro.conf.subcommand(name=...)` for that.
- Every other path becomes `--<flag>` (underscores become dashes) plus its
  value: `--model.depth 10`, a bool as `--model.use-mask` or
  `--model.no-use-mask`, a `Path` as POSIX text, an `Enum` by member name,
  a tuple as one argument per item.
- Subcommand words come first, then flags; `parse_config` reads them in any
  order.

### Which do I use?

- **A preset** — one run by hand: `classify ts2vec-paper`.
- **`from_presets` plus axes** — a study: the preset becomes the recipe axis,
  and the data and seed axes of the study choose what it runs over.
- **A hand-written settings axis** — what no preset covers, e.g. a learning
  rate sweep: `setting_axis(path='model.learning_rate', values=(1e-3, 5e-4))`.

A full guide to studies is on the [Studies and HPC](studies.md) page.

## Reproducing a run

```python
config = tslab.load_config(config_class=ClassifyConfig, path=Path('runs/.../config.yaml'))
config = tslab.parse_config(config_class=ClassifyConfig, default=config)  # flags change it further
```

A run reproduced from its `config.yaml` has the same fingerprints, so it loads
every result.
