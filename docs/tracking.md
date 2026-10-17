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
result, and Lightning's training metrics. The run's final state (`finished`
or `failed`) is sent at the end. A tracker that fails never fails the run: it
is logged and skipped.

## W&B on clusters

W&B can mark a run `crashed` when the network drops or a job is killed. TSLab
starts W&B with generous timeouts, falls back to offline mode and then to no
W&B when the server is unreachable, and records the real exit status. Repair
the server's view afterwards:

```bash
uv run tslab wandb-sync runs/ --dry-run     # what would be resynced
uv run tslab wandb-sync runs/               # resync crashed and failed runs
uv run tslab wandb-sync runs/ --include-offline
```

`--tracking.shell-log-path` uploads a SLURM output file to W&B at the end of
the run.
