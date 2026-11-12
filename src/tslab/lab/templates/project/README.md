# {{project_name}}

Experiments built with [TSLab](https://tslab.readthedocs.io/). Every step is
cached and resumable: running the same command again skips finished work.

## Set up and try it

```bash
uv sync
uv run pytest
uv run python -m {{package_name}}.experiment --scale 2.0
```

Results land in `runs/<run path>/`: `config.yaml`, `status.json`, `logs/run.log`,
and one folder per stage instance.

## Run a study

A study runs one command over a product of axes. The example lives in
`src/{{package_name}}/studies/example/study.py`:

```bash
uv run tslab study list
uv run tslab study show example
uv run tslab study submit example --machine local --dry-run
uv run tslab study submit example --machine local
uv run tslab study status example --machine local  # add --watch 60 to refresh
uv run tslab study wait example --machine local
uv run tslab study failures example --machine local
uv run tslab study results example --machine local --out results.csv
```

`machines/local.toml` runs the tasks on this machine. Copy
`machines/slurm.example.toml` to `machines/hpc.toml` and fill in the
`CHANGE_ME` values to run on a SLURM cluster.

## Where to change things

- `src/{{package_name}}/stages.py`: your own cached steps.
- `src/{{package_name}}/experiment.py`: the script that calls them in one run.
- `src/{{package_name}}/studies/`: studies over the script.
