# Tracking

A CSV tracker is always on: metrics land in `trackers/csv/version_<n>/` of
the run folder, one `version` per execution. Others are opt-in through
`tslab.tracking.TrackingOptions` (`--tracking.*` flags) and
`tslab.tracking.build_trackers(options=...)`:

| Tracker | Install | Flags |
|---|---|---|
| Weights & Biases | `tslab[wandb]` | `--tracking.wandb-mode online\|offline`, `--tracking.wandb-project`, `--tracking.wandb-entity`, `--tracking.wandb-tags` |
| MLflow | `tslab[mlflow]` | `--tracking.mlflow`, `--tracking.mlflow-tracking-uri`, `--tracking.mlflow-experiment` |
| TensorBoard | `tslab[tensorboard]` | `--tracking.tensorboard` |

Each tracker receives the config (flattened with `/`), every `Metrics`
result, and the training metrics of Lightning. The final state of the run (`finished`
or `failed`) is sent at the end. A tracker that fails never fails the run: it
is logged and skipped.

## W&B on clusters

W&B can mark a run `crashed` when the network drops or a job is killed. The
full notes — offline mode on clusters without network, and
`tslab wandb-sync` to repair the server afterwards (also available as an
automatic follow-up job of a study) — are on the
[Studies and HPC](studies.md#hpc-environment-notes) page.

`--tracking.shell-log-path` uploads a SLURM output file to W&B at the end of
the run.
