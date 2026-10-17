# {{project_name}}

Experiments built with [TSLab](https://tslab.readthedocs.io/). Every step
(training, encoding, evaluation, your own stages) is cached and resumable:
running the same command again skips finished work and resumes unfinished
training.

## Set up and try it

```bash
uv sync
uv run python -m {{package_name}}.toy_data data      # small synthetic datasets in data/
uv run python -m {{package_name}}.classify ts2vec-archive-split
uv run python -m {{package_name}}.forecast ts2vec-full-series
```

Results land in `runs/<run path>/`: `config.yaml`, `status.json`, `logs/run.log`,
and one folder per stage instance.

## Choose a preset, or set everything yourself

The first word after the script is a preset from `PRESETS` in the script, or
`no-preset`:

```bash
# a preset fixes its model and dataset; flags change any value
uv run python -m {{package_name}}.classify ts2vec-archive-split --trainer.max-epochs 50

# no preset: pick the model and dataset yourself
uv run python -m {{package_name}}.classify no-preset model:tstcc data:ucr \
    --data.dataset-folder-path /path/to/UCR/Coffee --data.target-column-name target \
    --run-path tstcc/Coffee/seed_0 --seed 0

# every setting, with its default
uv run python -m {{package_name}}.classify no-preset model:tstcc data:ucr --help
```

Add your own presets to `PRESETS`; `model_settings(...)` and `data_settings(...)`
build the settings by name.

## Rerun and check

```bash
uv run python -m {{package_name}}.classify ... --run.rerun-from encode   # redo encode and everything after it
uv run python -m {{package_name}}.classify ... --run.dry-run              # show what would run
uv run tslab status runs/                                                # one line per run
uv run tslab status runs/ts2vec/ToySpectra/archive-split                 # one line per instance
```

A run killed by a time limit shows as `interrupted`; the same command resumes it.

## Add a stage

`src/{{package_name}}/stages.py` has an example (`add_noise`) and explains
`@tslab.stage`. Call your stage from a script like any function.

## Check the code

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```
