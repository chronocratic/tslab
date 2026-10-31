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

## Run a study

A study runs one command of a script over a product of axes (see
`src/{{package_name}}/studies.py` for the example), on a machine profile in
`machines/`:

```bash
uv run tslab study show {{package_name}}.studies:example
uv run tslab study submit {{package_name}}.studies:example --machine local --dry-run
uv run tslab study submit {{package_name}}.studies:example --machine local
uv run tslab study status {{package_name}}.studies:example --machine local  # add --watch 60 to refresh
uv run tslab study wait {{package_name}}.studies:example --machine local
uv run tslab study failures {{package_name}}.studies:example --machine local
uv run tslab study run {{package_name}}.studies:example --machine local --grid classification --only seed=0  # one run in the terminal, nothing recorded
uv run tslab study usage {{package_name}}.studies:example --machine local  # per-run and per-group accounting, SLURM only
uv run tslab study resubmit {{package_name}}.studies:example --machine local
uv run tslab study results {{package_name}}.studies:example \
    --machine local --out results.csv
```

`--grid classification --only seed=0` keeps a slice of the runs; arguments
after `--` go to every run. Slurm: copy `machines/slurm.example.toml` to
`machines/cluster.toml` and fill in the values of your cluster. `--machine cluster`
finds it from this folder or any folder below. To use the name from any folder, run
`uv run tslab machine add cluster machines/cluster.toml`. The full guide:
the [Studies page](https://tslab.readthedocs.io/en/latest/studies.html) of the
TSLab docs.

## Rerun and check

```bash
# redo encode and everything after it
uv run python -m {{package_name}}.classify ... --run.rerun-from encode
# show what would run
uv run python -m {{package_name}}.classify ... --run.dry-run
# one line per run
uv run tslab status runs/
# one line per instance
uv run tslab status runs/ts2vec/ToySpectra/archive-split
```

A run killed by a time limit shows as `interrupted`; the same command resumes it.

## Add a stage

`src/{{package_name}}/stages.py` has an example (`add_noise`) and explains
`@tslab.stage`. Call your stage from a script like any function.

## Check the code

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```
