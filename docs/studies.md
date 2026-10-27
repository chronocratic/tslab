# Studies and HPC

A **study** is a saved definition of a round of experiments: a named set of
grids, each grid a product of axes that runs one script command. The same
study is submitted to a laptop or to a cluster, checked with `tslab study
status`, resubmitted after crashes, and collected with `tslab study results`
into one table.

```bash
uv run tslab study submit my_paper.studies:example --machine cluster
uv run tslab study status   my_paper.studies:example --machine cluster
uv run tslab study resubmit my_paper.studies:example --machine cluster
uv run tslab study results  my_paper.studies:example --machine cluster \
    --out results.csv
```

The study layer turns every run into the flags of the user's own `no-preset`
command and hands them to a launcher. TSLab never runs experiment code.

## Define a study

The `tslab new` template ships `src/<package>/studies.py`, a two-grid example
over the presets and toy data of the template:

```python
# my_paper/studies.py
from tslab.adapters.chronocratic.datasets import ucr_datasets
from tslab.study import from_presets, Grid, Override, setting_axis, Study

from my_paper import classify, forecast

example = Study(
    name='example',
    grids={
        'classification': Grid(
            short_name='cls',
            command=('uv', 'run', 'python', '-m', 'my_paper.classify', 'no-preset'),
            config=classify.ClassifyConfig,
            axes={
                'recipe': from_presets(
                    classify.PRESETS, names=('ts2vec-archive-split',), without=('data', 'run_path')
                ),
                'dataset': ucr_datasets(root='data', names=('ToySpectra',)),
                'seed': setting_axis(path='seed', values=(0, 1)),
            },
        )
        # ...
    },
)
```

A larger study: classification over UCR and UEA in one grid, forecasting with
a `variant` axis, an `exclude`, and an `Override`:

```python
from tslab.adapters.chronocratic.datasets import ucr_datasets, uea_datasets

UCR = ('Coffee', 'Wallet', 'EigenWorms')  # user code may compute these
UEA = ('BasicMotions', 'Epilepsy')

classification = Grid(
    short_name='cls',  # SLURM job name, at most 8 characters
    command=('uv', 'run', 'python', '-m', 'my_paper.classify', 'no-preset'),
    config=classify.ClassifyConfig,
    axes={
        'recipe': from_presets(
            classify.PRESETS, names=('ts2vec-paper', 'cost-paper'), without=('data',)
        ),
        'dataset': ucr_datasets(root='${UCR_ROOT}', names=UCR)
        | uea_datasets(root='${UEA_ROOT}', names=UEA),
        'seed': setting_axis(path='seed', values=(42, 7)),
    },
    exclude=({'recipe': 'cost-paper', 'dataset': 'EigenWorms'},),
    overrides=(
        Override(
            match={'recipe': 'ts2vec-paper', 'dataset': 'EigenWorms'},
            settings={'data.batch_size': 2},
        ),
    ),
)

forecasting = Grid(
    short_name='fc',
    command=('uv', 'run', 'python', '-m', 'my_paper.forecast', 'no-preset'),
    config=forecast.ForecastConfig,
    axes={
        'recipe': from_presets(forecast.PRESETS, names=('ts2vec-paper',), without=('data',)),
        'variant': {
            v: {'data': 'ett', 'data.dataset_file_path': '${ETT_ROOT}/ETT.csv', 'data.variant': v}
            for v in ('ETTh1', 'ETTh2')
        },
        'seed': setting_axis(path='seed', values=(42,)),
    },
)

example = Study(
    name='example', grids={'classification': classification, 'forecasting': forecasting}
)
```

From the shell a study is referenced as `module:attribute`:
`my_paper.studies:example`. Importing it imports the user's scripts, so a few
seconds on a login node is normal.

Rules that matter when writing one:

- **Axis order is folder order.** The path of a run is
  `<grid>/<value of axis 1>/<value of axis 2>/…`, and its folder is
  `<output_root>/<study>/<run path>`. Reorder the axes and the folders move
  with them.
- **Merge order, later paths replace earlier ones:** the value of each axis in
  axis order, then `run_settings` of the grid, then matching `Grid.overrides`,
  then `[settings]` of the machine profile, then matching machine overrides.
  After the merge each path appears once.
- **Names.** Study, grid, axis and value names match `[A-Za-z0-9._-]+` and
  are not `.` or `..`; `short_name` matches `[A-Za-z0-9_-]{1,8}` and is unique
  among the grids. Violations raise `ValueError` when the study is built.
- **Validation at import:** every settings path must exist on the
  `config` of the grid (a typo fails before anything is submitted). **Validation at
  `submit`, `resubmit` and `--dry-run`:** each selected run is parsed with
  `tslab.parse_config` before any job is written, so missing required fields,
  wrong types and impossible union choices are caught there.
- **`setting_axis(*, path, values)`** is `{str(v): {path: v} for v in values}`:
  seeds, learning rates, epochs. Any other hand-written mapping of value name
  to settings is also an axis, and one can combine axes with `|`.
- **`${VAR}`** in any string setting expands at submit time from the machine
  [env] of the machine profile and then the environment of the submit process; an unset
  variable raises. **`{output_root}`, `{study}`, `{run_path}`, `{log_file}`, `{cpus}`**
  are TSLab placeholders, expanded per run; nothing else in `{...}` is
  substituted. `{cpus}` is the `cpus_per_task` of the run on a SLURM machine
  and the CPUs the submit process may use on a local one, so a profile can set
  `"data.num_workers" = "{cpus}"` once and keep it in step with its overrides.
- **A study holds data, not logic.** A study file may list data — including
  one dict comprehension over a constant tuple, as the `variant` axis of
  `forecasting` does — but it must not compute, branch or loop over results. The presets of the
  script stay complete and unchanged (see
  [Configs and presets](configs-and-presets.md)); the study only re-reads
  them through `from_presets` and drops the parts it chooses on its own axes.

## Machine profiles

One TOML file per machine, committed by the user (`machines/cluster.toml`,
`machines/laptop.toml`). `${VAR}` in string values expands **on that
machine**, so no personal path has to be hard-coded. The template ships a
runnable `machines/local.toml` and a fully commented
`machines/slurm.example.toml`.

`--machine` takes a name or a path. A value with a `/`, a `.toml` suffix, or
that is an existing file, is a path. Any other value is a name, resolved in
this order:

1. `machines/<name>.toml` in the working directory, then in each folder above
   it. The first hit wins, so the nearest `machines/` folder is used when
   projects nest. The jobs run in the folder that holds that `machines/`
   folder.
2. The register of names, a TOML file mapping each name to the absolute path
   of a profile file: `$TSLAB_MACHINES_FILE` if set, else
   `$XDG_CONFIG_HOME/tslab/machines.toml`, else `~/.config/tslab/machines.toml`.
   The jobs run in the working directory.

A name found in neither place fails with the folders searched and the
registered names. A registered name whose file is gone fails with its own
message.

```bash
uv run tslab machine add cluster /home/me/lab/experiments/machines/cluster.toml
uv run tslab machine list
uv run tslab machine remove cluster
```

`add` stores the absolute path after checking that the file loads as a machine
profile. `list` marks files that are missing. Follow-up jobs on a SLURM cluster
get the absolute path of the profile, so the compute nodes need no register.

| Key | Default | Meaning |
|---|---|---|
| `launcher` | required | `"local"` or `"slurm"` |
| `output_root` | required | Root of the run folders, e.g. `"${SCRATCH}/exps"` |
| `tslab_command` | `["uv", "run", "tslab"]` | How follow-up jobs call TSLab |
| `[settings]` | none | Settings of the kind of the study, applied to every run on this machine |
| `[env]` | none | Variables exported in every job and read by `${...}` at submit |
| `[slurm]` | — | Job settings (below) |
| `[followup]` | — | Automatic helper jobs (below) |
| `[[overrides]]` | none | Per-composition settings and SLURM values |

`[slurm]` keys and defaults: `partition` (required), `qos = null`,
`time = "2-00:00:00"`, `cpus_per_task = 1`, `mem_per_cpu = null`, `gpus = 0`,
`max_concurrent = 100` (the `%N` of `--array`), `max_array_size = 1000`
(array indices must stay below the MaxArraySize of the cluster),
`sbatch_extra = []` (raw `#SBATCH` lines), `node_local_tmp = true` (points
`TMPDIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR` at node-local scratch),
`submit_pause_seconds = 0.5`.

`[followup]` keys and defaults: `auto_retry = false`, `max_attempts = 3`,
`wandb_sync = false`, `time = "00:30:00"`.

Rules:

- Parsed with the standard library into frozen dataclasses. **Unknown keys
  raise** (a typo like `cpu_per_task` must not be ignored); wrong types
  raise.
- **Settings keys are dotted paths and must be quoted in TOML**
  (`"tracking.wandb_mode" = "offline"`), because an unquoted dotted key would
  make nested tables. Nested tables are accepted too and flattened, so
  `[settings.tracking]` + `wandb_mode = "offline"` is the same.
- **`[env]` values may use `${VAR}`**; they expand at submit, are exported in
  the job script, and are visible to `${...}` in study settings during that
  submit.
- **Overrides** (`[[overrides]]`) carry a `match` (one value or an explicit
  list per axis, plus the optional `grid` key), and may set `slurm` values
  and `settings`. Every matching override applies; more matched keys wins
  over fewer; ties go to the later entry in the file. `slurm` values replace
  field by field; `settings` merge path by path in the same order.
- **Unknown names in machine overrides only warn.** One profile serves many
  studies, so an axis, grid or value name the current study lacks is a
  warning printed once per command, not an error (an override written for an older
  study must not block a newer one). Matches written in the study
  itself (`exclude`, `Override`) still raise.
- **Secrets never go in the profile.** W&B keys come from `wandb login`.
- **Resources resolve when a run is submitted** (and again at resubmit), so
  editing the profile and resubmitting moves runs to another partition or
  time limit.

## Submit

```bash
uv run tslab study submit my_paper.studies:example --machine cluster \
    [--grid NAME] [--only AXIS=V1,V2] [--include-done] [--dry-run] [-- extra args]
```

- **Filters** (`--grid`, `--only`, both repeatable) choose a subset; they
  never change the path of a run. `--only` applies to every grid that has that
  axis; an unknown axis or value name raises.
- **Skips** runs that are already done or have a live job (queued or
  running). `--include-done` resubmits done runs too (useful with
  `-- --run.rerun-from evaluate`).
- **`--dry-run`** validates every selected run, writes the job files and
  prints the plan (runs per array, resources, the full command of the first
  run of each array) without submitting or recording.
- **Pass-through:** everything after `--` is appended, raw, to every
  command of the run — flags of the user's script that TSLab does not know about.
- The study folder `<output_root>/<study>/.study/` holds everything the
  layer writes: `jobs/` (one `.sbatch` and one `.tasks` file per submission),
  `logs/` (one file per run, every attempt) and `submissions.jsonl` — the
  public record, `format_version: 1`, one JSON object per line: submission
  id, time, launcher, job id, and per task the array index, grid, run path,
  full command, resolved resources, attempt number and progress snapshot.

**Run names on the cluster.** The first time a run is submitted it gets the
next free array index of its grid and keeps it through every resubmit, so
adding, removing or reordering axis values never moves an existing run. The
index plus `short_name` of the grid names the run everywhere: `<short_name>_<index>`
in `squeue`, in `scancel <job>_<index>`, and in its log
`.study/logs/cls_17.out`, which collects every attempt under a header line.
`short_name` is at most 8 characters because the default name column of
`squeue` shows 8. An index at or above `max_array_size` stops the submission before
anything is written, naming the grid — raise `max_array_size` (or the
MaxArraySize of the cluster) for grids beyond about 1000 runs.

Runs of one grid with identical resolved SLURM settings share one array job
(`--array=3,7,12%max_concurrent`, consecutive indices written as ranges). A
resubmit is a sparse array of the same indices.

## Status

```bash
uv run tslab study status my_paper.studies:example --machine cluster \
    [--grid NAME] [--only AXIS=V1,V2] [--states STATE]
```

Add `--watch SECONDS` to refresh this report until Ctrl-C. Hints suggest
following active runs, resubmitting failed, interrupted, or missing runs, or collecting results
when all runs finish.

Every expected run gets exactly one of six states, decided in this order:

| State | When |
|---|---|
| `queued` | its latest job is pending in SLURM |
| `running` | its latest job is running, or its lock is live |
| `done` | `status.json` says `finished` and no instance is `failed` or `skipped` |
| `interrupted` | its run state is `running` or `crashed` with a stale lock; or its latest job ended `TIMEOUT`, `NODE_FAIL` or `PREEMPTED`; or an instance failed with the time limit |
| `failed` | any other `failed` run, or `finished` with failed or skipped instances; or its latest job ended `OUT_OF_MEMORY`, `FAILED` or `CANCELLED` with no run folder |
| `missing` | no run folder and no live job |

Job states come from `squeue` and `sacct`. When those commands are not
available (a laptop looking at a copied run folder) the job state is unknown
and the output says so once; the folder-based states still work. The output
lists per-grid counts by state, totals, then one line per non-`done` run:
run path, state, the first line of the error, job id. The exit code is 0
always: a status report is not a failure.

## Resubmit

```bash
uv run tslab study resubmit my_paper.studies:example --machine cluster \
    [--states failed,interrupted,missing] [--cancel-live] [--dry-run] [-- extra args]
```

- **Default states: `interrupted,missing`.** `failed` must be asked for with
  `--states failed`, because a failure usually needs a change first.
- A run with a live job is never resubmitted; `--cancel-live` runs `scancel`
  on those jobs first. Replacements start about 11 minutes later, when the
  heartbeat of the old lock (refreshed every minute) has gone stale and the new
  run takes the folder over.
- Stale locks are removed when SLURM says the job has ended, so a run does
  not wait out the full ten minutes if the job is provably gone.
- Each resubmitted task records `attempt = previous + 1`, and resources are
  resolved again from the current profile. Pass-through arguments after `--`
  are kept from the last submission of the run when new ones are not given.

## Automatic retry

Off by default (`[followup] auto_retry = true` turns it on).

After each submission, one follow-up job is queued with
`--dependency=afterany:<all job ids of that submission>` and minimal
resources (`[followup] time`, 1 CPU, the same partition and qos). It runs
`<tslab_command> study resubmit <study> --machine <name> --auto`:

- `--auto` resubmits **only `interrupted` runs**, never `failed` or
  `missing` (startup failures and code errors, including out-of-memory, stay
  manual);
- a run is not retried once its attempt number reaches `max_attempts`
  (default 3);
- **progress rule:** a run is retried only if it made progress during its
  last attempt — more `done` instances than the snapshot taken when that
  attempt was submitted, or a newer `last.ckpt` in the run folder.
  Otherwise it is reported as stuck and left for the user;
- if anything was resubmitted, the follow-up queues the next follow-up the
  same way.

**Escalating resources (more time, another qos, a GPU) is never
automatic.** For runs stopped by time limits, `status` suggests it; the
change goes in `[[overrides]]` of the machine profile.

With `[followup] wandb_sync = true`, one more follow-up job runs
`<tslab_command> wandb-sync <output_root>/<study>` after the jobs of the
submission end, so offline W&B runs reach the server even though the cluster has
no network (see below).

## Results

```bash
uv run tslab study results my_paper.studies:example --machine cluster \
    --out results.csv
```

One row per (run, instance), plus one row per run that has no instances, so
missing and failed-at-start runs appear too. Columns: `grid`, one column per
axis name (holding the value names), `run_path`, `study_state`, `instance`,
`instance_state`, `seconds` (the wall-clock time of the instance), `error`, then one column per metric name found in
`Metrics` instances (`metric/<name>`), empty where absent. Unreadable or
malformed files do not stop the table: the row gets
`instance_state = "unreadable"` and the error text.

TSLab writes the CSV with the standard library and adds no analysis;
`pandas.DataFrame(rows)` is one line in user code (also available in
Python: `Study.results(*, machine)` returns the rows).

## HPC environment notes

- **Node-local temporary folders.** With `node_local_tmp = true` (the
  default), each job points `TMPDIR`, `WANDB_CACHE_DIR` and
  `WANDB_CONFIG_DIR` at node-local scratch, so shared filesystems are not
  the bottleneck for cache and scratch files.
- **W&B on clusters.** W&B can mark a run `crashed` when the network drops
  or a job is killed. TSLab starts W&B with generous timeouts, falls back to
  offline mode and then to no W&B when the server is unreachable, and
  records the real exit status. On a cluster without network, run with
  `--tracking.wandb-mode offline` (via `[settings]` of the profile) and sync
  afterwards — from the login node, or from the automatic follow-up job:

  ```bash
  uv run tslab wandb-sync runs/ --dry-run     # what would be resynced
  uv run tslab wandb-sync runs/               # resync crashed, failed and killed runs
  uv run tslab wandb-sync runs/ --include-offline
  ```

- **Time limits** — training stops in time to save a checkpoint, and a
  run stopped by the limit resumes on resubmit: see
  [HPC and SLURM](hpc.md#time-limits).

## Known limitations

- **Union members must be named.** Every member of a choice union needs a
  `tyro.conf.subcommand(name=...)` name; the study layer addresses members
  by those names.
- **The text `'None'` is ambiguous.** The string `None` cannot be a setting
  value for a field that accepts `None`; pass the Python value `None`.
- **No concurrent submits of one study.** Two `submit` commands for the same
  study at the same time are not supported; the record files are written per
  process.
- **Dataset caches must be concurrent-safe.** When many array tasks read the
  same dataset, the dataset library must make its cache writing safe for
  concurrent jobs.
