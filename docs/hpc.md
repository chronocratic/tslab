# HPC and SLURM

## Time limits

`LightningTrainer` stops training in time to save a checkpoint:

- `--trainer.max-training-time 23:30:00` caps training (any sbatch `--time`
  format: `M`, `M:S`, `H:M:S`, `D-H`, `D-H:M`, `D-H:M:S`);
- under SLURM, the remaining job time (`SLURM_JOB_END_TIME`) caps it as well:
  training gets `--trainer.slurm-time-fraction` (default 0.5) of the time left,
  so the stages after training can still run.

Training stopped by the time limit is unfinished: submit the same command
again and it resumes from its last checkpoint.

## Resubmitting

A job killed at its time limit leaves its lock behind. The lock carries a
heartbeat refreshed every minute, so after ten minutes without one the next
job takes the run over, on any node. `tslab status runs/` lists such runs as
`interrupted`. `tslab study resubmit` resubmits such runs of a study
automatically; see [Studies and HPC](studies.md#resubmit). A follow-up job
submits with the SLURM variables of its own job removed, so the new job does
not inherit its allocation. Attempts that end `PREEMPTED` or `NODE_FAIL` do not
use up the retries of a run; see [Studies and HPC](studies.md#resubmit).

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

Only rank 0 writes TSLab files; the other ranks read. Stages other than
`train` run in one process.
