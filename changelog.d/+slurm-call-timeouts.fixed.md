SLURM queries now time out after 3 minutes and retry twice, so a hung `sacct` no longer blocks `status`, `wait` or `resubmit`.
