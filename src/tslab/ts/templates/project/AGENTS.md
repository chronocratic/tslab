# AGENTS.md: working in {{project_name}}

Instructions for coding agents (and humans) who run and extend the experiments
in this project. The project was created by `tslab new` from the TSLab
template and belongs to you: change any file here. TSLab itself is a library
installed from PyPI; never edit it inside `.venv/`. The full TSLab guide is at
<https://tslab.readthedocs.io>; `README.md` has the short version.
For agent-driven study runs, follow the [TSLab skill](.agents/skills/tslab/SKILL.md).

## 1. What TSLab gives you

- **Run:** `with tslab.lab.Run(config=..., root=..., relative_path=...) as run:`
  owns one folder, locks it, seeds Python `random` and NumPy, and torch once `tslab.ml` is imported, writes
  `config.yaml`, `meta.json` and `status.json`, and starts the trackers.
- **Stage:** a cached, resumable step. Ready ones are `tslab.ml.stages.train`,
  `tslab.ts.stages.encode` and `tslab.ts.stages.evaluate`; your own are functions
  decorated with `@tslab.lab.stage`. A script calls stages like ordinary
  functions, in any order, in loops and conditions.
- **Instance:** one call of a stage in a run. A stage called several times in
  one run gets a `name` per call (`evaluate[h24]`, `add_noise[std-0.5]`).
- **Fingerprint:** a hash of the stage name and `version`, the seed, the
  settings passed in and the fingerprints of the input artifacts. A matching
  fingerprint loads the saved result instead of computing it. Anything else
  recomputes the instance and everything after it.
- **Artifact:** what a stage returns, saved atomically: `tslab.lab.Json`,
  `tslab.lab.Metrics` (also sent to the trackers), `tslab.ml.Tensors` (safetensors),
  `tslab.ml.Checkpoint`.

Running the same command again is always safe: finished work is loaded,
interrupted training resumes from its last checkpoint, failed and skipped
instances run again.

## 2. How the project is split

```
pyproject.toml                  dependencies (tslab with the chronocratic extras) and tool settings
README.md                       quick start for humans
machines/
  local.toml                    machine profile: run study tasks on this machine
  slurm.example.toml            machine profile template for a SLURM cluster (copy, fill CHANGE_ME)
src/{{package_name}}/
  classify.py                   script: train, encode, SVM probe; noisy-test robustness check
  forecast.py                   script: train, encode, ridge probe per horizon
  probe.py                      script: load a trained checkpoint, encode, SVM or Ridge probe
  models.py                     `Models`: the model choice shared by both scripts and the study
  augmentations.py              your augmentation recipes, offered on matching models
  stages.py                     your own stages (example: add_noise)
  studies/
    common/                      datasets, recipes and grids shared by studies
    example/study.py             classification and forecasting study
    quick/study.py               small classification canary study
    chain/study.py, grids.py    probe grid that runs after the pretrain grid (`After`)
  toy_data.py                   writes small synthetic datasets into data/
tests/
  test_smoke.py                 runs presets and one example run; dry-runs the studies
data/                           datasets (git-ignored)
runs/                           run folders (git-ignored)
```

The `chain` study trains one checkpoint, then runs separate SVM and Ridge
probes against it. The Ridge head rounds regression outputs to the nearest
training class before reporting accuracy.

### What each file is for

- **A script** (`classify.py`, `forecast.py`) is one experiment type. It has
  three parts:
  1. a config: one frozen dataclass whose fields are the command-line flags
     (`--trainer.max-epochs`, `model:ts2vec`, `data:ucr`);
  2. `PRESETS`: named, complete configs for one run by hand;
  3. `main(*, config)`: opens a `tslab.lab.Run` and calls the stages.

  `tslab.lab.parse_config(config_class=..., presets=PRESETS)` turns the command
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
- **`stages.py`** holds your own `@tslab.lab.stage` functions.
- **`studies/<name>/study.py`** defines one `tslab.lab.study.Study` as `study` per package;
  `studies/common/` holds shared datasets, recipes and grid builders.
- **`machines/*.toml`** says where and how study runs execute (local or
  SLURM, output root, environment, SLURM limits, per-grid overrides). Its
  settings may contain only machine settings; put experiment settings on the study.
  `retry` covers grids that are not chained; `chained_retry` covers chained
  grids, and `requeue` is SLURM only.

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
  config.yaml        the parsed config; tslab.lab.load_config reloads it
  meta.json          versions, git commit, host, config hashes
  status.json        state of every instance (done, failed, skipped, ...)
  logs/run.log       everything logged during the run
  trackers/csv/      always-on CSV metrics
  train/default/     checkpoints
  encode/default/    representations
  evaluate/<name>/   metrics; a failed instance has error.txt with the traceback
```

A run killed by a time limit shows as `interrupted`; the same command resumes it.

`notebooks/analysis.ipynb` analyzes a study with `tslab.lab.analysis`: results, missing runs,
a critical-difference diagram, paired tests and a numbers file. Its comparison runs on
data marked "SYNTHETIC DATA" until the swap line in the notebook is applied. Output lands
in `notebooks/analysis/` (`stats/`, `tables/`, `coverage/`, `plots/<category>/<format>/`,
`results_numbers.tex`); synthetic files go to its `synthetic/` folder.

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
`tslab.lab.REQUIRED` on a field that differs per machine (a dataset path); the
command line or the study must then supply it.

### Add a setting to an experiment

Add a field to the config dataclass with a default and an `Attributes:` line
in the docstring; use it in `main`. Settings that cannot change a result
(device, workers, logging intervals) are typed
`Annotated[..., tslab.lab.NotFingerprinted]` so they do not invalidate cached results.

### Add a stage

In `stages.py`:

```python
@tslab.lab.stage(name='my_step', output=tslab.ml.Tensors, version=1)
def my_step(
    *,
    run: tslab.lab.Run,
    name: tslab.lab.InstanceName | None = None,  # only if one run calls it several times
    representations: tslab.ml.Tensors,  # artifacts are inputs
    strength: float,  # everything else is a fingerprinted setting
) -> tslab.ml.Tensors: ...
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
  `tslab.ml.Tensors.external(path=...)` (or another artifact class).
- `in_place=True` lets a long stage write into its folder while running and
  resume from what it wrote.
- Within one stage no instance name may be a prefix of another.

### Add an experiment type

Copy the closer of `classify.py` and `forecast.py` to a new module, rename the
config class, keep the three parts (config, `PRESETS`, `main`) and the
`if __name__ == '__main__':` line, add a test to `tests/test_smoke.py`, and a
grid to `studies/<name>/study.py` if it runs in a study; keep shared grid
builders in `studies/common/`.

### Add an augmentation recipe

Add a `@dataclass(frozen=True, kw_only=True)` class with `__call__` returning
`AugmentationProducer[<view type>]` to `augmentations.py`, and list it in
`chronocratic_model_choice(augmentations=(...))` in `models.py`. Select it with
`model.augmentation:<recipe name>`; `model.augmentation:default` keeps the
built-in one.

### Use your own model, data, probe or trainer

Implement the TSLab interface and pass the object where the chronocratic one
is passed now: `tslab.ml.interfaces.DataSource`, `Encoder`, `DownstreamModel` or
`Probe`, `Trainer`; see the "Extending TSLab" page of the docs. Ready ones:
`tslab.ts.probes.SVMProbe`, `tslab.ts.probes.RidgeProbe`, `tslab.ml.lightning.LightningTrainer`.

### Use real data

Put datasets under `data/` (git-ignored) or anywhere else, and point the
dataset flags at them (`--data.dataset-folder-path`, `--data.dataset-file-path`).
Never commit datasets.

## 5. Studies (many runs, local or SLURM)

A study runs one script command over the product of its axes. See
`studies/example/` for the example and `studies/common/` for shared grids.

```bash
uv run tslab study list
uv run tslab study show example
uv run tslab study submit example --machine local --dry-run
uv run tslab study submit quick --machine local --dry-run
uv run tslab study submit example --machine local --detach  # background launch; never submit in the foreground
uv run tslab study status example --machine local  # add --watch 60 to refresh
uv run tslab study wait example --machine local
uv run tslab study failures example --machine local
uv run tslab study logs example --machine local --grid classification --only <axis>=<value> -n 100  # last lines of one run log; never -f, it waits until Ctrl-C
uv run tslab study run example --machine local --grid classification --only seed=0  # one run in the terminal, nothing recorded
uv run tslab study usage example --machine local  # per-run and per-group accounting
uv run tslab study resubmit example --machine local
uv run tslab study cancel example --machine local  # stops the live local launch
uv run tslab study results example --machine local --out results.csv
```

`tslab.toml` at the project root names the studies package, so `example` stands
for `{{package_name}}.studies.example:study`. Experiment settings belong in
`Study.settings`; machine profiles may set only machine settings. Its
`[machine_by_host]` table, commented out by default, picks the machine by host:
uncomment it and drop `--machine` from every command.

`--grid NAME` and `--only axis=value` select a slice; arguments after `--` go
to every run (`-- --trainer.max-epochs 1`). Always `--dry-run` first.

`--code-snapshot` on `submit` (or `code_snapshot = true` in `tslab.toml`)
copies `src/` into a code snapshot the jobs import, so queued and resubmitted
runs keep the code of submit time; `resubmit --new-code-snapshot` and
`--no-code-snapshot` change the code a resubmit runs. If a submit stops with
untracked files under `src/` (exit 2 with their list), ask the researcher how
to proceed — `git add` them, `--code-snapshot-include-untracked`, or drop the
snapshot — and never pick the flag on your own.

Writing a study:

- A `Grid` has a `short_name` (the default SLURM job name), a
  `command` (the script with `no-preset`), the `config` class and `axes`.
  Optional `job_name='{recipe}_{short_name}'` names jobs from axis values and
  splits the grid into one array per name.
- SLURM commands for a study: `tslab study slurm STUDY` lists them,
  `tslab study slurm STUDY NAME` runs one (also `.study/slurm.sh`). Agents use
  `tslab study status` and the `sacct-*` commands, not `squeue-*`.
- Axes are mappings of value name to settings (dotted config paths:
  `'model.depth': 10`, `'data': 'ett'`). Helpers: `from_presets(presets=PRESETS, names=..., without=('data', 'run_path'))`
  for recipes, `ucr_datasets` / `uea_datasets` for datasets,
  `setting_axis(path='seed', values=(0, 1))` for sweeps; combine axes with `|`.
- **Run paths:** runs land in `<output_root>/<study>/<grid>/<axis 1 value>/...`;
  `run_path='{grid}/{dataset}/seed_{seed}'` on the grid changes it.
- Settings merge in this order, later wins: axes, `Study.settings`, default run
  settings, grid `run_settings`, grid `overrides`, machine `[settings]`, machine
  `[[overrides]]`. Machine profiles may set only marked machine settings.
- `exclude=` drops combinations; `Override(match=..., settings=...)` changes
  settings of matching runs.
- `${VAR}` in a string expands from the machine `[env]`, then the shell
  environment. `{output_root}`, `{study}`, `{run_path}`, `{log_file}`, `{cpus}`,
  `{cpus_per_run}` expand per run; `{cpus}` is `cpus_per_task` on SLURM and the usable CPUs on
  a local machine; `{cpus_per_run}` is the CPUs of one run when several run at once.
- SLURM jobs run through `srun`; `gpus` means one task per GPU. `cpus_per_task` applies to each
  task, so 4 GPUs with 8 CPUs per task request 32 CPUs.
- `[slurm] setup` lists shell lines that run before tasks. `gpu_type` selects the GPU type for the
  GRES request. `signal_seconds` sends `SIGUSR2` before the time limit; `0` disables it.
- Training saves `last.ckpt` at the next batch boundary after `SIGUSR2`; the run is marked
  interrupted. Other stages stop without a checkpoint. Resubmit the study to resume training.
- `[slurm.sbatch_extra]` maps sbatch long names to values. Entries replace matching defaults;
  `false` removes an option. The old list form is rejected. `clusters` is rejected: TSLab tracks
  jobs on the default cluster only.
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
- Do not use torch in a stage without importing `tslab.ml`. A run seeds Python `random` and NumPy; importing `tslab.ml` adds torch seeding and the torch, CUDA and GPU metadata. The ready ts stages import it for you, your own stages must do it.
- Do not run two processes on the same run folder; the second one fails with
  `tslab.lab.RunLockedError`.
- Do not commit `data/`, `runs/` or cluster secrets.
