# Configs and presets

## A config is a frozen dataclass

Every field becomes a command-line flag through
[tyro](https://brentyi.github.io/tyro/): `--scale 2.0`,
`--run-path experiment/small/seed_0`. Union fields become subcommands:
`method:mean`, `method:median`. `tslab.lab.parse_config` reads the command line, and
`Run` writes the result to `config.yaml`.

```python
config = tslab.lab.parse_config(config_class=ExperimentConfig, presets=PRESETS)
```

Flags may come before or after a subcommand: `method:mean --seed 3` and
`--seed 3 method:mean` are the same.

## Presets

With `presets=`, the first word picks where to start:

- `no-preset`: the config class itself; choose every method and dataset with
  subcommands and flags;
- a preset name: a finished config. A preset **fixes its method and dataset**
  (`my-preset method:median` is an error, so the method settings of a preset are never
  silently dropped), while flags still change any value
  (`my-preset --scale 10`).

### What a preset is

A preset is a **named, complete config for one run by hand**: every field
has a value, so the single command `experiment mean-small` runs as is.
Completeness is deliberate:

- one command reproduces the run: the preset *is* the record of it;
- its method and dataset choices are fixed on the command line, so a
  mistaken extra subcommand is an error instead of a silently dropped
  setting;
- it names a dataset, so it runs without any other input. That is also why
  a study [drops the data part](#a-preset-as-a-study-recipe): a round chooses
  its datasets on its own axis.

### Building one

A preset is an ordinary instance of the config class. A real recipe sets
several values of the chosen method:

```python
PRESETS = {
    'mean-small': (
        'Mean with a small trim',
        ExperimentConfig(
            method=MeanMethod(trim=0.1, weights='uniform'),
            data=FileData(path=Path('data/small.csv')),
            scale=2.0,
        ),
    )
}
```

Fields not given keep their defaults. For models and datasets that come from
a package with generated settings classes, see the helper functions of that
package, such as the [ts layer](../ts/index.md).

### `tslab.lab.REQUIRED`

A preset may put `tslab.lab.REQUIRED` on a field its author cannot or does not
want to know, such as a dataset path that lives in a different folder on each
machine. The marker is **optional**; presets without it are unaffected.

- **By hand:** `experiment mean-any` without the value exits with the
  tyro error box (`Missing value for --data.path`, exit code 2);
  with `--data.path ...` it runs.
- **In a study:** `from_presets` skips marked fields, so the study must
  supply that field on one of its axes (usually the dataset axis); otherwise
  the per-run parse at submit fails with the same message.
- A parsed config never holds the marker, so saved configs never contain it.

### A preset as a study recipe

`from_presets` turns presets into the settings of the recipe axis of a study. It
keeps each choice field as its enum member and every other field
**only where it differs from the default**, plus every field without a
default. `mean-small` above gives:

```python
{'method': 'mean', 'method.trim': 0.1, 'method.weights': 'uniform', 'scale': 2.0}
```

(with the dataset removed by `without=('data',)`). Each setting mapping
becomes flags of the `no-preset` command of the grid:

```bash
uv run python -m my_project.experiment no-preset method:mean \
    --method.trim 0.1 --method.weights uniform --scale 2.0 ...
```

```python
'recipe': from_presets(presets=PRESETS, names=('mean-small', 'median-small'), without=('data',))
```

### Settings: how the values become flags

Settings are **dotted paths** of the config class (`method.trim`,
`scale`) with values: `str`, `int`, `float`, `bool`, `Path`,
`StrEnum`, tuples and lists, or `None`.

- A path that ends at a **union field** with a string value becomes the
  subcommand word `<path>:<value>`: `method:mean`, `data:file`. The names
  are the tyro subcommand names of the union; members of your own choice
  unions must be named with `tyro.conf.subcommand(name=...)` for that.
- Every other path becomes `--<flag>` (underscores become dashes) plus its
  value: `--method.trim 0.1`, a bool as `--method.use-cache` or
  `--method.no-use-cache`, a `Path` as POSIX text, a `StrEnum` by its value (`--method.mode fast`),
  a tuple as one argument per item.
- Subcommand words come first, then flags; `parse_config` reads them in any
  order.

### Which do I use?

- **A preset**: one run by hand, `experiment mean-small`.
- **`from_presets` plus axes**: a study. The preset becomes the recipe axis,
  and the data and seed axes of the study choose what it runs over.
- **A hand-written settings axis**: what no preset covers, such as a
  sweep over a number: `setting_axis(path='scale', values=(1.0, 2.0))`.

A full guide to studies is on the [Studies and HPC](studies.md) page.

## Reproducing a run

```python
config = tslab.lab.load_config(config_class=ExperimentConfig, path=Path('runs/.../config.yaml'))
config = tslab.lab.parse_config(
    config_class=ExperimentConfig, default=config
)  # flags change it further
```

A run reproduced from its `config.yaml` has the same fingerprints, so it loads
every result.
