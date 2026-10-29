# AGENTS.md: working in {{project_name}}

Instructions for coding agents (and humans) who run and extend the experiments
in this project. The project was created by `tslab new` from the TSLab
template and belongs to you: change any file here. TSLab itself is a library
installed from PyPI; never edit it inside `.venv/`. The full TSLab guide is at
<https://tslab.readthedocs.io>; `README.md` has the short version.
For agent-driven study runs, follow the [TSLab skill](.agents/skills/tslab/SKILL.md).

## 1. What TSLab gives you

- **Run:** `with tslab.Run(config=..., root=..., relative_path=...) as run:`
  owns one folder, locks it, seeds Python, NumPy and torch, writes
  `config.yaml`, `meta.json` and `status.json`, and starts the trackers.
- **Stage:** a cached, resumable step. Ready ones are `tslab.stages.train`,
  `tslab.stages.encode` and `tslab.stages.evaluate`; your own are functions
  decorated with `@tslab.stage`. A script calls stages like ordinary
  functions, in any order, in loops and conditions.
- **Instance:** one call of a stage in a run. A stage called several times in
  one run gets a `name` per call (`evaluate[h24]`, `add_noise[std-0.5]`).
- **Fingerprint:** a hash of the stage name and `version`, the seed, the
  settings passed in and the fingerprints of the input artifacts. A matching
  fingerprint loads the saved result instead of computing it. Anything else
  recomputes the instance and everything after it.
- **Artifact:** what a stage returns, saved atomically: `tslab.Json`,
  `tslab.Metrics` (also sent to the trackers), `tslab.Tensors` (safetensors),
  `tslab.Checkpoint`.

Running the same command again is always safe: finished work is loaded,
interrupted training resumes from its last checkpoint, failed and skipped
instances run again.

## 2. How the project is split

```
pyproject.toml                  dependencies (tslab with the chronocratic extras) and tool settings
README.md                       quick start for humans
machines/
  local.toml                    machine profile: run study tasks one after another, here
  slurm.example.toml            machine profile template for a SLURM cluster (copy, fill CHANGE_ME)
src/{{package_name}}/
  classify.py                   script: train, encode, SVM probe; noisy-test robustness check
  forecast.py                   script: train, encode, ridge probe per horizon
  models.py                     `Models`: the model choice shared by both scripts and the study
  augmentations.py              your augmentation recipes, offered on matching models
  stages.py                     your own stages (example: add_noise)
  studies.py                    studies: grids of runs over the scripts
  toy_data.py                   writes small synthetic datasets into data/
tests/
  test_smoke.py                 runs every preset and the example study for one batch
data/                           datasets (git-ignored)
runs/                           run folders (git-ignored)
```

### What each file is for

- **A script** (`classify.py`, `forecast.py`) is one experiment type. It has
  three parts:
  1. a config: one frozen dataclass whose fields are the command-line flags
     (`--trainer.max-epochs`, `model:ts2vec`, `data:ucr`);
  2. `PRESETS`: named, complete configs for one run by hand;
  3. `main(*, config)`: opens a `tslab.Run` and calls the stages.

  `tslab.parse_config(config_class=..., presets=PRESETS)` turns the command
  line into a config.
- **`models.py`** builds `Models` with `chronocratic_model_choice(...)`: every
  model of chronocratic-models becomes a subcommand (`model:ts2vec`,
  `model:tstcc`, `model:cost`, ...), with all its settings as flags. Both
  scripts and the study import it, so a recipe added there shows up
  everywhere.
- **`augmentations.py`** holds recipes: frozen dataclasses whose `__call__`
  builds a chronocratic augmentation producer. The return annotation decides
  which models offer the recipe (`ViewPair`: TSTCC, MHCCL, SimCLR, CoST;
  `AlignedPair`: TS2Vec; `SingleView`: AutoTCL).
- **`stages.py`** holds your own `@tslab.stage` functions.
- **`studies.py`** holds `tslab.study.Study` objects: data only, no logic.
- **`machines/*.toml`** says where and how study runs execute (local or
  SLURM, output root, environment, SLURM limits, per-grid overrides).

## 3. Setup and everyday commands

```bash
uv sync
uv run python -m {{package_name}}.toy_data data       # toy datasets in data/
uv run python -m {{package_name}}.classify --help     # list the presets
uv run python -m {{package_name}}.classify ts2vec-archive-split
uv run python -m {{package_name}}.forecast ts2vec-full-series
```

The first word after the script is a preset name or `no-preset`:

```bash
# a preset fixes its model and dataset; flags change any other value
uv run python -m {{package_name}}.classify ts2vec-archive-split --trainer.max-epochs 50

# no preset: choose model and dataset with subcommands, then flags
uv run python -m {{package_name}}.classify no-preset model:tstcc data:ucr \
    --data.dataset-folder-path data/ToySpectra --data.target-column-name target \
    --run-path tstcc/ToySpectra/seed_0 --seed 0

# every setting of a model and dataset, with defaults
uv run python -m {{package_name}}.classify no-preset model:tstcc data:ucr --help
```

`my-preset model:cost` is an error on purpose: a preset never silently loses
its model settings. Use `no-preset` to pick another model.

### Run options (never change a fingerprint)

| Flag | Effect |
|---|---|
| `--run.dry-run` | print what would be loaded or run; write nothing |
| `--run.rerun-from STAGE` | run this stage and every stage after it again |
| `--run.rerun ID ...` | run these instances again, e.g. `'evaluate[noise-std-0.5]'` |
| `--run.force-all` | run everything again |
| `--run.fail-fast` | raise the first error instead of recording it and going on |
| `--run.log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Inspect results

```bash
uv run tslab status runs/                                      # one line per run
uv run tslab status runs/ts2vec/ToySpectra/archive-split       # one line per instance
```

A run folder looks like this:

```
runs/<run path>/
  config.yaml        the parsed config; tslab.load_config reloads it
  meta.json          versions, git commit, host, config hashes
  status.json        state of every instance (done, failed, skipped, ...)
  logs/run.log       everything logged during the run
  trackers/csv/      always-on CSV metrics
  train/default/     checkpoints
  encode/default/    representations
  evaluate/<name>/   metrics; a failed instance has error.txt with the traceback
```

A run killed by a time limit shows as `interrupted`; the same command resumes it.

### Tracking

CSV metrics are always written. W&B, MLflow and TensorBoard are opt-in with
`--tracking.*` flags (for example `--tracking.wandb-mode offline`); they need
the matching TSLab extra (`tslab[wandb]`, `tslab[mlflow]`, `tslab[tensorboard]`)
added to `pyproject.toml`. A tracker that fails never fails the run.

## 4. How to change things

### Add or change a preset

Add an entry to `PRESETS` in the script: `'name': ('one-line description', Config(...))`.
Build the model and dataset parts with `model_settings(choice=Models, name='ts2vec', depth=10)`
and `data_settings(name='ucr', dataset_folder_path=Path(...), target_column_name='target')`;
the settings classes behind `model:*` and `data:*` are generated and cannot be
imported. A preset is complete: every field has a value. Put
`tslab.REQUIRED` on a field that differs per machine (a dataset path); the
command line or the study must then supply it.

### Add a setting to an experiment

Add a field to the config dataclass with a default and an `Attributes:` line
in the docstring; use it in `main`. Settings that cannot change a result
(device, workers, logging intervals) are typed
`Annotated[..., tslab.NotFingerprinted]` so they do not invalidate cached results.

### Add a stage

In `stages.py`:

```python
@tslab.stage(name='my_step', output=tslab.Tensors, version=1)
def my_step(
    *,
    run: tslab.Run,
    name: tslab.InstanceName | None = None,  # only if one run calls it several times
    representations: tslab.Tensors,  # artifacts are inputs
    strength: float,  # everything else is a fingerprinted setting
) -> tslab.Tensors: ...
```

Then call it from `main` like a function:
`my_step(run=run, name=f'strength-{s}', representations=representations, strength=s)`.

- Parameters are keyword-only, and so are the calls.
- **Bump `version` after changing what the code computes**, or cached results
  of the old code are reused.
- Pass every artifact the stage depends on as its own argument. An artifact
  hidden inside another object is not seen as an input.
- An artifact you build by hand has no fingerprint and cannot be passed to a
  stage: compute it in a stage, or wrap a file with
  `tslab.Tensors.external(path=...)` (or another artifact class).
- `in_place=True` lets a long stage write into its folder while running and
  resume from what it wrote.
- Within one stage no instance name may be a prefix of another.

### Add an experiment type

Copy the closer of `classify.py` and `forecast.py` to a new module, rename the
config class, keep the three parts (config, `PRESETS`, `main`) and the
`if __name__ == '__main__':` line, add a test to `tests/test_smoke.py`, and a
grid to `studies.py` if it runs in a study.

### Add an augmentation recipe

Add a `@dataclass(frozen=True, kw_only=True)` class with `__call__` returning
`AugmentationProducer[<view type>]` to `augmentations.py`, and list it in
`chronocratic_model_choice(augmentations=(...))` in `models.py`. Select it with
`model.augmentation:<recipe name>`; `model.augmentation:default` keeps the
built-in one.

### Use your own model, data, probe or trainer

Implement the TSLab interface and pass the object where the chronocratic one
is passed now: `tslab.interfaces.DataSource`, `Encoder`, `DownstreamModel` or
`Probe`, `Trainer`; see the "Extending TSLab" page of the docs. Ready ones:
`tslab.probes.SVMProbe`, `tslab.probes.RidgeProbe`, `tslab.lightning.LightningTrainer`.

### Use real data

Put datasets under `data/` (git-ignored) or anywhere else, and point the
dataset flags at them (`--data.dataset-folder-path`, `--data.dataset-file-path`).
Never commit datasets.

## 5. Studies (many runs, local or SLURM)

A study runs one script command over the product of its axes. See
`studies.py` for the example.

```bash
uv run tslab study show {{package_name}}.studies:example
uv run tslab study submit {{package_name}}.studies:example --machine local --dry-run
uv run tslab study submit {{package_name}}.studies:example --machine local
uv run tslab study status {{package_name}}.studies:example --machine local  # add --watch 60 to refresh
uv run tslab study wait {{package_name}}.studies:example --machine local
uv run tslab study failures {{package_name}}.studies:example --machine local
uv run tslab study resubmit {{package_name}}.studies:example --machine local
uv run tslab study results {{package_name}}.studies:example --machine local --out results.csv
```

`--grid NAME` and `--only axis=value` select a slice; arguments after `--` go
to every run (`-- --trainer.max-epochs 1`). Always `--dry-run` first.

Writing a study:

- A `Grid` has a `short_name` (SLURM job name, at most 8 characters), a
  `command` (the script with `no-preset`), the `config` class and `axes`.
- Axes are mappings of value name to settings (dotted config paths:
  `'model.depth': 10`, `'data': 'ett'`). Helpers: `from_presets(PRESETS, names=..., without=('data', 'run_path'))`
  for recipes, `ucr_datasets` / `uea_datasets` for datasets,
  `setting_axis(path='seed', values=(0, 1))` for sweeps; combine axes with `|`.
- **Axis order is folder order:** runs land in
  `<output_root>/<study>/<grid>/<axis 1 value>/<axis 2 value>/...`.
- Settings merge in this order, later wins: axes, grid `run_settings`, grid
  `overrides`, machine `[settings]`, machine `[[overrides]]`.
- `exclude=` drops combinations; `Override(match=..., settings=...)` changes
  settings of matching runs.
- `${VAR}` in a string expands from the machine `[env]`, then the shell
  environment. `{output_root}`, `{study}`, `{run_path}`, `{log_file}`, `{cpus}` expand per run;
  `{cpus}` is `cpus_per_task` on SLURM and the usable CPUs on a local machine.
- A study holds data, not logic: no computation or branching on results.
  Every path is checked against the config class at import, and every run is
  parsed at submit, so typos fail before anything runs.

For a SLURM cluster, copy `machines/slurm.example.toml` to
`machines/cluster.toml` and fill every `CHANGE_ME`. `--machine cluster` finds
`machines/cluster.toml` from the working directory or any folder above it; to use
the name from any folder, register it once with
`uv run tslab machine add cluster machines/cluster.toml` (`tslab machine list`
and `tslab machine remove cluster` manage the register). Details:
the "Studies and HPC" page of the TSLab docs.

## 6. Code rules for this project

The project uses the same tooling as TSLab: ruff with every rule on, ty, pytest.

- Run before you say a change is done, and report failures as they are:

  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
  ```

- Full type hints, Google-style docstrings on public functions and classes.
- Keyword-only parameters (`def main(*, config: ...)`) and keyword arguments in calls.
- Configs and recipes are `@dataclass(frozen=True, kw_only=True)`.
- Paths are `pathlib.Path`, never `str`; TSLab raises `TypeError` for `str`.
- Raise with a message variable: `message = '...'; raise ValueError(message)`.
- Fix lint instead of silencing it; a needed `# noqa: CODE` gets a reason.
- Keep `tests/test_smoke.py` passing: every preset and the example study run
  for one batch on toy data. Add a case when you add a preset or a script.
- Tests write only to `tmp_path`, never into `runs/` or `data/`.

## 7. Things not to do

- Do not edit TSLab inside `.venv/`; upgrade it with `uv lock --upgrade-package tslab`.
- Do not delete or hand-edit files in a run folder to force a rerun; use the
  `--run.*` flags.
- Do not change what a stage computes without bumping its `version`.
- Do not run two processes on the same run folder; the second one fails with
  `tslab.RunLockedError`.
- Do not commit `data/`, `runs/` or cluster secrets.
