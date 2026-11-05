# Studies and HPC

A **study** is a saved definition of a round of experiments: a named set of
grids, each grid a product of axes that runs one script command. The same
study is submitted to a laptop or to a cluster, checked with `tslab study
status`, resubmitted after crashes, and collected with `tslab study results`
into one table.

For an agent workflow with JSON documents and exit codes, see the
[Agent guide](agents.md).

```bash
uv run tslab study submit example --machine cluster
uv run tslab study status   example --machine cluster
uv run tslab study resubmit example --machine cluster
uv run tslab study results  example --machine cluster \
    --out results.csv
```

The study layer turns every run into the flags of the user's own `no-preset`
command and hands them to a launcher. TSLab never runs experiment code.

## Define a study

The `tslab new` template ships one package per study and keeps shared datasets,
recipes and grid builders under `studies/common/`:

```
my_paper/studies/
  common/
    __init__.py
    datasets.py
    recipes.py
    grids.py
  example/
    __init__.py
    study.py
  quick/
    __init__.py
    study.py
```

The common package can build a grid from shared axes and recipes:

```python
# my_paper/studies/common/grids.py
from tslab.study import Grid, setting_axis

from my_paper import classify
from my_paper.studies.common.datasets import TOY_UCR
from my_paper.studies.common.recipes import classification_recipes


def classification_grid(*, seeds: tuple[int, ...]) -> Grid:
    return Grid(
        short_name='cls',
        command=('uv', 'run', 'python', '-m', 'my_paper.classify', 'no-preset'),
        config=classify.ClassifyConfig,
        axes={
            'recipe': classification_recipes(),
            'dataset': TOY_UCR,
            'seed': setting_axis(path='seed', values=seeds),
        },
    )
```

Each study package exposes its `Study` as `study`. Experiment settings stay on
that study, while machine profiles set only machine settings:

```python
# my_paper/studies/example/study.py
from tslab.study import Study

from my_paper.studies.common import classification_grid

study = Study(
    name='example',
    settings={'trainer.max_epochs': 500, 'trainer.max_training_time': '6:00:00'},
    grids={'classification': classification_grid(seeds=(0, 1))},
)
```

A sibling `quick/study.py` can reuse the grid with `seeds=(0,)` and shorter
training limits. Package `__init__.py` files expose `study`; the template also
shares a forecasting grid in `common`.

### Study settings

Put experiment-wide settings on `Study`. Each path applies only to grids whose
config has that path. A study setting cannot overlap a path from any axis or
`Grid.run_settings`; equal paths and parent-child paths both count. Remove a
recipe value with `from_presets(..., without=(...))` when the study owns it.
`Grid.overrides` may still set a study path for matching runs.

With `studies = "my_paper.studies"` in `tslab.toml`, the short name `example`
resolves to `my_paper.studies.example:study`. Put each study in its own module
or package; shared definitions belong in `common`. Run `tslab study list` to
discover the available names; `--json` emits the `tslab.study-list/1` schema.
Importing a study imports the user's scripts, so a few seconds on a login node
is normal.

Rules that matter when writing one:

- **Default run path.** By default the path of a run is
  `<grid>/<value of axis 1>/<value of axis 2>/…`, and its folder is
  `<output_root>/<study>/<run path>`; reorder the axes and the folders move
  with them. `Grid(run_path=...)` changes the layout.
- **Merge order, later paths replace earlier ones:** values of each axis in
  axis order, `Study.settings`, default run settings, `Grid.run_settings`,
  matching `Grid.overrides`, `[settings]` of the machine profile, then matching
  machine overrides. Study settings apply only to grids whose config has each
  path. Machine profiles may name only fields marked `MachineSetting` or
  `HardwareSetting`; move experiment settings to the study. After the merge
  each path appears once.
- **Names.** Study, grid, axis and value names match `[A-Za-z0-9._-]+` and
  are not `.` or `..`; `short_name` matches `[A-Za-z0-9_-]+` and is unique
  among the grids. No axis may be named `short_name`, `study`, `output_root`,
  `run_path`, `log_file`, `cpus` or `cpus_per_run`. Violations raise `ValueError` when the study is built.
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
  variable raises. **`{output_root}`, `{study}`, `{run_path}`, `{log_file}`, `{cpus}`,
  `{cpus_per_run}`**
  are TSLab placeholders, expanded per run; nothing else in `{...}` is
  substituted. `{cpus}` is the `cpus_per_task` of the run on a SLURM machine
  and the CPUs the submit process may use on a local one, so a profile can set
  `"data.num_workers" = "{cpus}"` once and keep it in step with its overrides.
  `{cpus_per_run}` is the CPUs of one run when several run at once: the
  `cpus_per_task` of the run on a SLURM machine and the usable CPUs split over
  the parallel slots on a local one; use it for settings that do not change
  results, such as `data.num_workers`.
- **A study holds data, not logic.** A study file may list data — including
  one dict comprehension over a constant tuple, as the `variant` axis of
  `forecasting` does — but it must not compute, branch or loop over results. The presets of the
  script stay complete and unchanged (see
  [Configs and presets](configs-and-presets.md)); the study only re-reads
  them through `from_presets` and drops the parts it chooses on its own axes.

**Run paths.** `run_path` is a template of `{grid}` (the grid key),
`{short_name}` and the axis names of the grid; between placeholders only
letters, digits, dot, dash, underscore and slash are allowed. Examples:

```python
Grid(..., run_path='{grid}/{recipe}/{dataset}/seed_{seed}')  # seed_0 folder
Grid(..., run_path='{dataset}/{recipe}/{seed}')  # no grid folder
Grid(..., run_path='{short_name}-{recipe}-{dataset}-s{seed}')  # one flat folder
```

Value names do not change, so `--only seed=0` and the results table work as
before. The study checks every filled path: each part between slashes is a
valid name, and no two runs, in one grid or across grids, share a path. A
template that leaves out an axis with two or more values fails this check.
Changing the template of a study that already ran moves its folders, so resume
and results no longer find the old runs.

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

A project can skip `--machine` altogether: the `[machine_by_host]` table of
[tslab.toml](project-settings.md) picks the profile by host, and the same file
expands short study names.

| Key | Default | Meaning |
|---|---|---|
| `launcher` | required | `"local"` or `"slurm"` |
| `output_root` | required | Root of the run folders, e.g. `"${SCRATCH}/exps"` |
| `tslab_command` | `["uv", "run", "tslab"]` | How follow-up jobs call TSLab |
| `[settings]` | none | Machine settings marked `MachineSetting` or `HardwareSetting`, applied to every run on this machine |
| `[env]` | none | Variables exported in every job and read by `${...}` at submit |
| `[slurm]` | — | Job settings (below) |
| `[followup]` | — | Automatic helper jobs (below) |
| `[[overrides]]` | none | Per-composition settings and SLURM values |

`[slurm]` keys and defaults: `partition` (required), `qos = null`,
`time = "2-00:00:00"`, `cpus_per_task = 1`, `mem_per_cpu = null`, `gpus = 0`,
`gpu_type = null`, `setup = []`, `signal_seconds = 120` (send `SIGUSR2` this
many seconds before the time limit; `0` disables it),
`max_concurrent = 100` (the `%N` of `--array`), `max_array_size = 1000`
(array indices must stay below the MaxArraySize of the cluster),
`[slurm.sbatch_extra]` (optional sbatch long names; replaces matching default
lines, with `false` removing a line), `node_local_tmp = true` (points
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
- **Settings paths.** Every `[settings]` and `[[overrides]].settings` path
  must end at a `MachineSetting` or `HardwareSetting` field of each applicable
  grid config. A choice-union field itself is not a machine setting. Put
  experiment settings in the study instead.
- **Pinned machine settings.** A study or grid may set a machine setting to
  require a specific device or worker count. If the machine profile resolves
  that path to a different value after placeholder expansion, planning fails
  with both values and the profile path. Matching values pass silently.
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

## Local machines

A profile with `launcher = "local"` runs the tasks of a submit on the machine
that calls `tslab`. The optional `[local]` table sets how:

| Key | Default | Meaning |
|---|---|---|
| `timeout` | none | Time limit of one task, in sbatch duration format; a task that passes it is stopped |
| `signal_seconds` | `120` | Seconds before the limit to send `SIGUSR2`, so training saves a checkpoint; `0` disables it |
| `gpus` | `[]` | GPU ids the launch may use: a list of NVIDIA ids, `["mps"]`, or `"auto"` to find them at launch |
| `skip_busy_gpus` | `false` | Leave out NVIDIA GPUs that already run a compute process |
| `busy_gpu_memory_fraction` | `0.1` | With `skip_busy_gpus`, also skip a GPU with more of its memory in use than this fraction |
| `max_parallel` | `1` | Tasks at once when the resolved GPU list is empty; an error with an explicit `gpus` list |
| `tasks_per_gpu` | `1` | Tasks that share one GPU id at once |
| `keep_awake` | `true` | Keep the machine awake while a launch runs |

- **Tasks.** Without GPUs the launch runs `max_parallel` tasks at once. With
  GPUs it runs `tasks_per_gpu` tasks on each GPU, and each task sees one GPU
  through `CUDA_VISIBLE_DEVICES`. A task that fails does not stop the others.
  With one slot the console shows the output of each task; with more slots it
  shows one line when a task starts and one when it ends. The full output of
  every task goes to its log file in `.study/logs/`.
- **Apple GPU.** On Apple Silicon the GPU is one device with the id `mps`. Set
  `gpus = ["mps"]` or let `"auto"` find it. `skip_busy_gpus` does not apply to
  it, and `mps` cannot be listed next to NVIDIA ids.
- **GPU locks.** A task takes a lock on its GPU before it starts, so two
  studies that run on the same machine never share a GPU by accident. The lock
  files live in a folder of the current user, so other users do not see them.
  A task waits for a free GPU, and the console says so.
- **Time limit.** With `timeout`, a task gets `SIGUSR2` at
  `timeout - signal_seconds`, then `SIGTERM`, and `SIGKILL` if it is still
  running a little later.
- **Stopping.** `Ctrl-C`, `SIGTERM` and `SIGHUP` to `tslab` stop the launch:
  running tasks get `SIGTERM`, then `SIGKILL`, and no new task starts. The
  command exits with `128 + N` for signal N. Resubmit to resume. A hard kill
  (`SIGKILL`) of `tslab` cannot clean up and may leave tasks running; stop
  them by hand.
- **`--detach`.** `tslab study submit --detach` (and `resubmit --detach`)
  starts the launch in the background and returns once it runs. Each launch writes
  its own output to `.study/local_logs/<pid>.log` (launcher lines only; run
  output stays in `.study/logs/`), and `.study/local-launch.json` names the
  process while it runs. Follow it with `tslab study wait` or `status`. It is
  an error on a SLURM machine, and a second launch of the same study fails
  while one is live.
- **Cancel.** `tslab study cancel` sends `SIGTERM` to the live local launch.
  A launch is one unit, so the selection must cover all of its live tasks, and
  `--signal` is for SLURM only. The stopped runs read as `interrupted`.
- **Usage.** `tslab study usage` reads the usage file each task writes next to
  its log: the elapsed time, the peak memory, the CPU time and the GPU id. A
  run without a limit shows `no limit`. A local machine has no time limit or
  memory request, so it prints no suggestion line.
- **Automatic retry.** With `[followup] auto_retry = true`, the launch submits
  the interrupted runs that made progress again when it ends, up to
  `max_attempts` attempts each, in the same process. `wandb_sync` is for SLURM
  machines.

## Training stop rules

Training ends by one of three rules: `max_epochs`, `early_stopping`, or
`max_training_time`. The ready train stage writes `training.json` next to the
checkpoints with the rule that stopped it (`stop_reason`), the epochs and the
steps. A run stopped by `max_training_time` logs a warning.

- `max_epochs` is reproducible from the seed.
- `--trainer.max-training-time 23:30:00` caps training (any sbatch `--time`
  format: `M`, `M:S`, `H:M:S`, `D-H`, `D-H:M`, `D-H:M:S`). It ends training
  normally when the total training time reaches the limit, counted across all
  resumed jobs. It changes trained weights and is fingerprinted, so changing it
  trains again.
- `max_training_time` keeps a run in a time budget, but the number of epochs
  depends on the speed of the hardware, so the result is not reproducible from
  the seed. Use `max_epochs` for results that must be.

## Submit

```bash
uv run tslab study submit example --machine cluster \
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
next free array index of its job name and keeps it through every resubmit, so
adding, removing or reordering axis values never moves an existing run. A run
is named `<job_name>_<index>`; the job name is `<study>_<short_name>` unless
a `job_name` template (below) sets it. The log is `<job_name>_<index>.out`,
such as `.study/logs/my_paper_cls_17.out`, and collects every
attempt under a header line. Cancel a run with its SLURM job id and array
index: `scancel <job>_<index>`.
An index at or above `max_array_size` stops the submission before
anything is written, naming the grid. Indices restart at 0 for each job name,
so the limit applies to one array, not to the whole grid: split a grid beyond
about 1000 runs with `job_name`, or raise `max_array_size` (and the
MaxArraySize of the cluster).

Runs of one grid with identical resolved SLURM settings and the same job name
share one array job (`--array=3,7,12%max_concurrent`, consecutive indices
written as ranges). A resubmit is a sparse array of the same indices. Each
array has its own `%max_concurrent` cap, because SLURM cannot share one cap
across arrays.

**Job names.** By default every job of a grid is named `<study>_<short_name>`,
so jobs of two studies are told apart in `squeue`. A grid can
name its jobs from axis values with `job_name`, a template of `{short_name}`,
`{study}` and axis names, written with single braces:

```python
Grid(short_name='ucr', job_name='{recipe}_{short_name}', ...)  # s11_ucr, b1_ucr, ...
```

The axes the template names split the grid into one array per value
combination; the other axes stay tasks inside the array. Between placeholders
only letters, digits, dot, dash and underscore are allowed, so `${VAR}`, `{{`
and spaces are rejected. Two grids must not give the same job name, and no
name may be `<study>_retry` or `<study>_wbsync`, the names of the follow-up
jobs. Each job name numbers its runs from 0 and names its log files
`<job_name>_<index>.out`, so arrays never share a log file. A run keeps its
index when the template changes, but its log file follows the new name, so
the logs of earlier attempts stay under the old one. Each array is one
`sbatch` call, so a finer split costs more submit pauses.

## Status

```bash
uv run tslab study status example --machine cluster \
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
run path, state, the first line of the error, job id. A queued run ends its
line with `starts <time>`, the estimate of the start of the job from the
scheduler, and, when known, the pending reason: `waiting: <reason>` while
SLURM may still start the job, and `will not start: <reason>` for reasons
that never clear on their own (for example `PartitionTimeLimit`).
`--json` adds `expected_start` (null when the estimate is unknown),
`pending_reason` and `pending_stuck` to the runs. The exit code is 0 always: a
status report is not a failure.

## Resubmit

```bash
uv run tslab study resubmit example --machine cluster \
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
- A resubmitted run keeps the code snapshot of its latest job;
  `--new-code-snapshot` and `--no-code-snapshot` change it: see
  [Code snapshots](#code-snapshots).

## Code snapshots

A job that waits in the queue for hours runs the code of the project folder
when it starts. A code snapshot copies the code at submit time and makes the
jobs import that copy, so later edits never change queued or resubmitted
runs. It is opt-in and off by default.

```bash
uv run tslab study submit example --machine cluster --code-snapshot
```

`code_snapshot = true` in [tslab.toml](project-settings.md) turns it on for
the project; `--no-code-snapshot` turns it off for one submit.

- **What is copied:** only the `src/` folder (src layout). Tracked files are
  copied with their working-tree bytes, so uncommitted edits are included; a
  tracked file deleted from the working tree is skipped. Ignored files are
  never copied.
- **Untracked files stop the submit.** With untracked files under `src/`, the
  submit stops before writing anything (exit 2) and lists them. The ways out
  are `git add` on them, `--code-snapshot-include-untracked` (copy them too,
  recorded as `untracked_copied`), or `--no-code-snapshot` (run the live
  code). `code_snapshot_check_untracked = false` in `tslab.toml` turns the
  stop off.
- **Where it lives:** `<output_root>/.tslab-code-snapshots/<hash>/`, the hash
  a SHA-256 over the copied files shortened to 16 hex characters. Same code,
  same folder: reused, never overwritten, shared across studies. Nothing
  deletes a snapshot folder; remove one by hand once no queued or
  resubmittable job records it.
- **How the jobs use it:** the jobs keep the project root as working folder;
  the snapshot's `src/` is prepended to `PYTHONPATH`, so the snapshot package
  wins over an editable install. The job checks that the folder still holds a
  `src/` before the task and fails the task with a clear line; there is no
  fallback to the live code.
- **Resubmit reuses the record.** Every resubmitted run runs the snapshot
  recorded on its latest job, so a retry runs the same code as the first
  attempt; the `--auto` follow-ups do the same. `--new-code-snapshot` takes a
  fresh snapshot with the checks above, `--no-code-snapshot` drops the
  recorded one and runs the live code. The `code_snapshot` of `tslab.toml`
  does not apply to resubmits. Resubmitting `failed` runs warns once per
  snapshot: "reusing the code snapshot from \<time\>; pass
  `--new-code-snapshot` if you fixed the code". A recorded folder that is
  gone is an error naming it and suggesting `--new-code-snapshot`.
- **What a run used:** the run `meta.json` holds a `code_snapshot` block
  (`folder`, `hash`, `commit`, `dirty`, `untracked_copied`) — read it for the
  code a run used; its `commit` is the `HEAD` at submit time and `dirty` says
  tracked files under `src/` differed from it. The existing `git` block is
  the repository at job start.
- **What is not frozen:** the dependencies, TSLab itself, and code outside
  `src/` (scripts called by path, notebooks). `tslab study run` (one run in
  the terminal) always runs the live code: it is the debugging path.
- **`--dry-run`** runs every check and computes the hash, copies nothing, and
  the job files name the folder the real submit would use.

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
  (default 3); attempts whose job ended `PREEMPTED` or `NODE_FAIL` do not
  count and skip the progress rule below, but after 20 of them in a row the
  run is left alone;
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
uv run tslab study results example --machine cluster \
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

Use `--out -` to write CSV to standard output. Add `--json` to write the
versioned `tslab.study-results/1` document to a file or standard output:

```bash
uv run tslab study results example --machine cluster --out -
uv run tslab study results example --machine cluster --out - --json
```

## Show a study

`tslab study show` lists each grid, its command, axis value names, and run
counts without needing a machine profile. Use those names to form `--grid` and
`--only axis=value` selections. Add `--json` for the
`tslab.study-show/1` document. See the [Agent guide](agents.md).

```bash
uv run tslab study show example --json
```

## Wait for a study

`tslab study wait` ends when work finishes or needs attention. The first check
runs at once; later checks start 30 seconds apart, the gap doubles while
nothing changes and goes back to 30 seconds when a run changes state.
`--interval SECONDS` sets the longest gap (default 300, at least 10).
`--timeout` returns exit code 124. Use `--keep-going` to wait until every run
stops, even after failures. It works in a background shell and supports
`--json`. See the [Agent guide](agents.md) for exit-code actions.

```bash
uv run tslab study wait example --machine cluster --json
```

## Inspect failures

`tslab study failures` shows failed and interrupted runs by default, with full
instance errors and the last 30 lines of each job log and run log. Each run
gets a `hint:` line with what the data shows: the peak and the requested
memory for an out-of-memory run, a pointer to `time` or `job_time_share` for a
run stopped by a limit, or a plain resubmit for a preemption or node failure.
A Python error in the code gets no hint; the traceback is the answer. When
SLURM recorded an exit code for the job, the line of the run ends with
`exit <code>`. `--json` adds the `hint` and `exit_code` fields to the runs.
Missing log files are named, not treated as command failures. Use `--states`
and `--lines` to change the selection and tail size. The command only reads
data. See the [Agent guide](agents.md).

```bash
uv run tslab study failures example --machine cluster --json
```

## Read the log of a run

`tslab study logs` prints the last 50 lines of `logs/run.log` of one run, found
from the same `--grid` and `--only` flags as the other commands. The selection
must match exactly one run; otherwise the command exits 2 and lists the
matches. `-f` keeps printing new lines until Ctrl-C, like `tail -f`, and waits
for a run that has not started yet. `-n N` changes the number of lines.
`--job` shows the scheduler output of the job instead, and `--launch` the log
of the latest local launch started with `--detach`.

```bash
uv run tslab study logs example --machine local --only seed=1 -f
```

## Run one task in the foreground

`tslab study run` runs exactly one run of the study in the current terminal:
the command of the run with the `[env]` exports and the `[slurm] setup` lines
of the machine, standard output and error to the terminal, and the exit code
of the command. It is a debugging step, not a submission: nothing is recorded
in the study folder, and SLURM is never called. It runs the live code, even
when the study used a code snapshot. A done run runs again with the
cached stages loaded at once; a queued or running run is refused (exit 1),
because two writers on one run folder clash.

The selection must name exactly one run; zero or several matches are a usage
error (exit 2) that lists the paths. Arguments after `--` replace the
pass-through arguments of the latest submission of the run; without them,
those are reused. The command starts one process; for a multi-GPU job it says
how many tasks the job would start.

```bash
uv run tslab study run example --machine cluster \
    --grid classification --only dataset=Coffee --only seed=42
```

Inside an interactive `salloc` allocation the machine still applies
`job_time_share` to the training; pass `-- --trainer.job-time-share 1` to
train for the whole allocation.

## Usage

`tslab study usage` reads the accounting of the selected runs from `sacct`
and shows what each run used: the state, the elapsed time against the time
limit, the peak memory against the requested memory, the CPU time, and the
GPUs. Runs that ended paused or signalled are marked `paused: needs more than
one job`; they use their share of the time limit by design and are left out
of the time suggestion, their memory still counting.

One line per resource group (the runs that share the resolved `[slurm]`
settings) suggests a `time` and a `mem_per_cpu` that fit the largest run of
the group with a 20 percent margin; that is the value to put in the machine
profile. `--json` prints the `tslab.study-usage/1` document. On a local machine
it reads the usage files of the tasks instead of `sacct` (see
[Local machines](#local-machines)).

```bash
uv run tslab study usage example --machine cluster
```

## Cancel live jobs

`tslab study cancel` cancels queued and running SLURM jobs in the selected
slice. Use `--dry-run` to inspect job ids first. `--signal` sends `USR2` to
the running jobs instead of cancelling them; the runs save a checkpoint and
end on their own, and read as `interrupted`. It is a way to test the timeout
path without waiting for the time limit; queued runs are skipped, because a
pending job has no step to signal. See [HPC and SLURM](hpc.md#time-limits)
for the test. On a local machine it stops the live local launch (see
[Local machines](#local-machines)); `--signal` is for SLURM only. See the
[Agent guide](agents.md).

```bash
uv run tslab study cancel example --machine cluster --dry-run --json
```

## Check a machine

`tslab doctor --machine NAME [STUDY]` checks the profile, required commands,
output storage, project, and optional study compatibility before submission.
For a SLURM machine it also asks the cluster: the array size against the
cluster `MaxArraySize` limit, and whether a node of the partition can hold the
time, CPUs, memory and GPUs of one job. A missing or failing `scontrol` or
`sinfo` warns, never fails. Warnings do not fail the command; inspect each
check. It does not verify that a real cluster job succeeds. See the [Agent
guide](agents.md).

```bash
uv run tslab doctor --machine cluster example --json
```

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
