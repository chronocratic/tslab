# HPC and SLURM

A cluster is a machine profile in `machines/`. The `[machine_by_host]` table
of [tslab.toml](project-settings.md) can pick it by host, so the study
commands of the project need no `--machine`:

```bash
uv run tslab study submit example
```

## Time limits

`LightningTrainer` uses separate limits to end training or pause it.

### Ends training

- `--trainer.max-training-time 23:30:00` caps training (any sbatch `--time`
  format: `M`, `M:S`, `H:M:S`, `D-H`, `D-H:M`, `D-H:M:S`). It ends training
  normally when the total training time reaches the limit. It counts across all
  resumed jobs.
- `max_training_time` changes trained weights and is fingerprinted. Changing it
  trains again.

### Pauses until the next job

- Under SLURM, `--trainer.job-time-share` (default `0.5`) gives training that
  share of the remaining job time. The deadline applies to the current job only.
  When it passes, TSLab saves `last.ckpt` and marks the run interrupted. Resubmit
  the same command to resume.
- SLURM also sends `SIGUSR2` before the job time limit. Training saves
  `last.ckpt` at the next batch boundary, then the run is marked `interrupted`.
  Other stages stop without a checkpoint. Resubmit the study to resume training;
  set `[slurm] signal_seconds = 0` to disable the signal.

Machine profiles set `trainer.job_time_share` (and `[slurm] time`) for the
time budget, not `max_training_time`: that setting is fingerprinted, so a
profile that sets it would make the same study train differently on every
machine.

### Testing the timeout path

`tslab study cancel STUDY --machine cluster --signal` sends `USR2` to the
running jobs of the selected runs instead of cancelling them, so a run
reaches the save-and-end path of a real time limit without waiting for it.
The runs save `last.ckpt` and end on their own as `interrupted`; queued jobs
are skipped. See [Studies and HPC](studies.md#cancel-live-jobs).

To test the pause path, submit one run with a tiny share; the training then
pauses with `TrainingPaused` after one batch:

```bash
uv run tslab study submit my_paper.studies:example --machine cluster \
    --grid classification --only dataset=Coffee --only seed=42 \
    -- --trainer.job-time-share 0.01
```

Machine profiles can run site setup commands before each task with
`[slurm] setup = ["module load cuda"]`. Set `gpu_type = "a100"` to request
`--gres=gpu:a100:<count>` when `gpus` is greater than zero.

Every study job runs through `srun` on one node. `gpus = 4` runs four tasks,
one per GPU; `cpus_per_task = 8` allocates eight CPUs per task, 32 total.
The default `signal_seconds = 120` sends `SIGUSR2` before the time limit.
Set `signal_seconds = 0` to turn off this warning.

Add site-specific options under `[slurm.sbatch_extra]`, using sbatch long
names. For example, `account = "project"` or `exclusive = true`. An entry
such as `time = "04:00:00"` replaces the normal time line; `no_requeue = false`
removes that line. The old list form is not accepted. `clusters` is rejected:
TSLab tracks jobs on the default cluster only.

## Resubmitting

A job killed at its time limit leaves its lock behind. The lock carries a
heartbeat refreshed every minute, so after ten minutes without one the next
job takes the run over, on any node. `tslab status runs/` lists such runs as
`interrupted`. `tslab study resubmit` resubmits such runs of a study
automatically; see [Studies and HPC](studies.md#resubmit). A follow-up job
submits with the SLURM variables of its own job removed, so the new job does
not inherit its allocation. Attempts that end `PREEMPTED` or `NODE_FAIL` do not
use up the retries of a run; see [Studies and HPC](studies.md#resubmit).
A queued job runs the code of the project folder when it starts; a
[code snapshot](studies.md#code-snapshots) copies the code at submit time and
makes the jobs import that copy.

## SLURM commands

After every submit on a SLURM machine, tslab writes `.study/slurm.sh` in the
study folder. `source` it and call a command by name; each one is a read-only
SLURM command for the jobs of that study, with the full job name shown. The
same commands run without sourcing as `tslab study slurm STUDY NAME` (no name
lists them) and in a notebook as `tslab.study.slurm_output`.

Commands for queued and running jobs take the ids from `sacct`, so a job that
has ended is not asked of `squeue`; on a cluster without accounting they use
every recorded id.

| Name | What it shows |
|---|---|
| `squeue-full-name` | Queued and running jobs with the full job name. |
| `squeue-long` | Queued and running jobs in the long format with the full job name. |
| `squeue-start` | Expected start times of the queued jobs. |
| `squeue-running` | Running tasks only, with the full job name. |
| `squeue-stock` | Queued and running jobs in the stock squeue format. |
| `sacct-full-name` | Every job of the study from the accounting database, full job name. |
| `sacct-failed` | Jobs that failed, timed out, ran out of memory or lost their node. |
| `sacct-usage` | Time, CPU and memory use of every job of the study. |
| `seff-all` | Efficiency summary of every job of the study. |
| `sstat-live` | Live CPU and memory use of the running tasks. |
| `sprio-jobs` | Scheduling priority of the queued jobs. |
| `scontrol-show` | Full controller record of each job still held by the controller. |

The `squeue-*` commands are for people at a terminal; agents use
`tslab study status` and the `sacct-*` commands.

## Many runs

A [study](studies.md) is the supported way to run many runs: one saved
definition, one SLURM array per grid, fixed run names, status and resubmit.

For one-off batches without a study, give each run its own folder
(`--run-path ts2vec/Coffee/seed_0`) and submit one job per run. Check the
folders afterwards:

```bash
uv run tslab status runs/            # one line per run: state and instance counts
uv run tslab status runs/ --stage attack
```

Add `--watch SECONDS` to refresh run status until Ctrl-C. Run-level hints point to failed-instance
error files and explain interrupted runs; folder hints suggest opening one run.

## Distributed training

SLURM jobs use one node and run through `srun`. The `gpus` setting requests one
task per GPU. For example, `gpus = 4` with `cpus_per_task = 8` starts four
training tasks and allocates 32 CPUs. All ranks train together, then ranks
other than rank 0 leave. Only rank 0 writes TSLab files. Stages other than
`train` run on rank 0 alone.
