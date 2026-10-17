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

```python
PRESETS = {
    'ts2vec-archive-split': (
        'TS2Vec, archive split',
        ClassifyConfig(
            model=model_settings(choice=Models, name='ts2vec'),
            data=data_settings(
                name='ucr', dataset_folder_path=Path('data/Coffee'), target_column_name='target'
            ),
        ),
    )
}
```

## Reproducing a run

```python
config = tslab.load_config(config_class=ClassifyConfig, path=Path('runs/.../config.yaml'))
config = tslab.parse_config(config_class=ClassifyConfig, default=config)  # flags change it further
```

A run reproduced from its `config.yaml` has the same fingerprints, so it loads
every result.
