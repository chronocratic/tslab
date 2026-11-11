# Migration

TSLab is now three packages in one distribution: `tslab.lab` (no torch), `tslab.ml` (torch and Lightning) and `tslab.ts` (time series). `tslab/__init__.py` exports only `__version__`; the old import paths no longer exist and there are no shims. Import every name from its new path.

| Old import | New import |
|---|---|
| `tslab.REQUIRED` | `tslab.lab.REQUIRED` |
| `tslab.Artifact` | `tslab.lab.Artifact` |
| `tslab.Checkpoint` | `tslab.ml.Checkpoint` |
| `tslab.ConfigFileError` | `tslab.lab.ConfigFileError` |
| `tslab.Fingerprintable` | `tslab.lab.Fingerprintable` |
| `tslab.HardwareSetting` | `tslab.lab.HardwareSetting` |
| `tslab.InstanceName` | `tslab.lab.InstanceName` |
| `tslab.InstanceState` | `tslab.lab.InstanceState` |
| `tslab.JobTimeLimitSignal` | `tslab.lab.JobTimeLimitSignal` |
| `tslab.Json` | `tslab.lab.Json` |
| `tslab.LogLevel` | `tslab.lab.LogLevel` |
| `tslab.MachineSetting` | `tslab.lab.MachineSetting` |
| `tslab.Metrics` | `tslab.lab.Metrics` |
| `tslab.ModelAsEncoder` | `tslab.ts.stages.ModelAsEncoder` |
| `tslab.NotFingerprinted` | `tslab.lab.NotFingerprinted` |
| `tslab.Run` | `tslab.lab.Run` |
| `tslab.RunLockedError` | `tslab.lab.RunLockedError` |
| `tslab.RunOptions` | `tslab.lab.RunOptions` |
| `tslab.RunState` | `tslab.lab.RunState` |
| `tslab.RunStatus` | `tslab.lab.RunStatus` |
| `tslab.Stage` | `tslab.lab.Stage` |
| `tslab.Tensors` | `tslab.ml.Tensors` |
| `tslab.Tracker` | `tslab.lab.Tracker` |
| `tslab.UnavailableArtifact` | `tslab.lab.UnavailableArtifact` |
| `tslab.UnavailableArtifactError` | `tslab.lab.UnavailableArtifactError` |
| `tslab.UnavailableReason` | `tslab.lab.UnavailableReason` |
| `tslab.WandbMode` | `tslab.ml.WandbMode` |
| `tslab.is_available` | `tslab.lab.is_available` |
| `tslab.load_config` | `tslab.lab.load_config` |
| `tslab.parse_config` | `tslab.lab.parse_config` |
| `tslab.save_config` | `tslab.lab.save_config` |
| `tslab.stage` | `tslab.lab.stage` |
| `tslab.stages` (module) | `tslab.ml.stages`, `tslab.ts.stages` |
| `tslab.study.DEFAULT_RUN_SETTINGS` | `tslab.lab.study.DEFAULT_RUN_SETTINGS` |
| `tslab.study.After` | `tslab.lab.study.After` |
| `tslab.study.Grid` | `tslab.lab.study.Grid` |
| `tslab.study.Override` | `tslab.lab.study.Override` |
| `tslab.study.SettingValue` | `tslab.lab.study.SettingValue` |
| `tslab.study.Settings` | `tslab.lab.study.Settings` |
| `tslab.study.Study` | `tslab.lab.study.Study` |
| `tslab.study.StudyRun` | `tslab.lab.study.StudyRun` |
| `tslab.study.check_paths` | `tslab.lab.study.check_paths` |
| `tslab.study.from_presets` | `tslab.lab.study.from_presets` |
| `tslab.study.setting_axis` | `tslab.lab.study.setting_axis` |
| `tslab.study.settings_to_args` | `tslab.lab.study.settings_to_args` |
| `tslab.study.slurm_output` | `tslab.lab.study.slurm_output` |
| `tslab.interfaces.SPLITS` | `tslab.ml.SPLITS` |
| `tslab.interfaces.Batch` | `tslab.ml.Batch` |
| `tslab.interfaces.DataSource` | `tslab.ml.DataSource` |
| `tslab.interfaces.DownstreamModel` | `tslab.ts.DownstreamModel` |
| `tslab.interfaces.Encoder` | `tslab.ts.Encoder` |
| `tslab.interfaces.LabeledSplits` | `tslab.ml.LabeledSplits` |
| `tslab.interfaces.Metric` | `tslab.ml.Metric` |
| `tslab.interfaces.Probe` | `tslab.ts.Probe` |
| `tslab.interfaces.SequenceEncoder` | `tslab.ts.SequenceEncoder` |
| `tslab.interfaces.Trainer` | `tslab.ml.Trainer` |
| `tslab.stages.ModelAsEncoder` | `tslab.ts.stages.ModelAsEncoder` |
| `tslab.stages.check_representations` | `tslab.ts.stages.check_representations` |
| `tslab.stages.encode` | `tslab.ts.stages.encode` |
| `tslab.stages.evaluate` | `tslab.ts.stages.evaluate` |
| `tslab.stages.train` | `tslab.ml.stages.train` |
| `tslab.enums.AfterCondition` | `tslab.lab.enums.AfterCondition` |
| `tslab.enums.AfterVerdict` | `tslab.lab.enums.AfterVerdict` |
| `tslab.enums.ChainedRetry` | `tslab.lab.enums.ChainedRetry` |
| `tslab.enums.CheckStatus` | `tslab.lab.enums.CheckStatus` |
| `tslab.enums.InstanceState` | `tslab.lab.enums.InstanceState` |
| `tslab.enums.JobFlag` | `tslab.lab.enums.JobFlag` |
| `tslab.enums.JobPhase` | `tslab.lab.enums.JobPhase` |
| `tslab.enums.LauncherKind` | `tslab.lab.enums.LauncherKind` |
| `tslab.enums.LocalTaskState` | `tslab.lab.enums.LocalTaskState` |
| `tslab.enums.LogLevel` | `tslab.lab.enums.LogLevel` |
| `tslab.enums.OutputFormat` | `tslab.lab.enums.OutputFormat` |
| `tslab.enums.Retry` | `tslab.lab.enums.Retry` |
| `tslab.enums.RunState` | `tslab.lab.enums.RunState` |
| `tslab.enums.Satisfy` | `tslab.lab.enums.Satisfy` |
| `tslab.enums.ShownRunState` | `tslab.lab.enums.ShownRunState` |
| `tslab.enums.Split` | `tslab.ml.Split` |
| `tslab.enums.StudyState` | `tslab.lab.enums.StudyState` |
| `tslab.enums.WandbMode` | `tslab.ml.WandbMode` |

## What else changes

- Old run folders, caches and saved configs do not load after this change. Class paths are part of fingerprints, saved configs and artifact metadata, so every cache misses. Produce them again.
- Fingerprints changed. Caches, run results and checkpoints written by an earlier version are not reused after upgrading: every stage runs again and the cache says `fingerprint changed` for each instance.
- `tslab new` takes `--template lab|ts`. The default is `ts` when torch is installed, so a full install gives the same project as before, and `lab` on a bare install. `--template lab` gives a project with no torch and no time series code.
- Torch seeding and the torch, Lightning, CUDA and GPU metadata happen only when `tslab.ml` is imported. The lab seeds Python `random` and NumPy. A stage that uses torch must import `tslab.ml`. See [Seeding](lab/run-options.md#seeding).
- `tslab wandb-sync` and the W&B, MLflow and TensorBoard trackers are in `tslab.ml`.
- The module paths below the packages changed too (for example `tslab.core` became `tslab.lab.core`, `tslab.study` became `tslab.lab.study`, `tslab.adapters.chronocratic` became `tslab.ts.adapters.chronocratic`).

## Internal package imports

The doors of internal subpackages are empty, and a few doors are trimmed. Import from `tslab.lab`, `tslab.ml` or `tslab.ts`, or from the module that defines the name.

| Old import | New import |
|---|---|
| `tslab.lab.cli.main` | `tslab.lab.cli.entrypoint.main` |
| `tslab.lab.cli.common.CONSOLE`, `make_console`, `watch`, ... | `tslab.lab.cli.common.console` |
| `tslab.lab.cli.common.UsageError` | `tslab.lab.cli.common.errors.UsageError` |
| `tslab.lab.cli.common.print_json`, `error_document`, ... | `tslab.lab.cli.common.output` |
| `tslab.lab.cli.common.Request`, `resolve_request` | `tslab.lab.cli.common.project` |
| `tslab.lab.cli.study.STUDY_COMMANDS` | `tslab.lab.cli.study.registry.STUDY_COMMANDS` |
| `tslab.lab.cli.study.Submit`, `Status`, `Cancel`, ... | `tslab.lab.cli.study.submit`, `status`, `cancel`, ... |
| `tslab.lab.core.meta.hostname`, `collect_*`, `register_*` | `tslab.lab.core.meta.run` |
| `tslab.lab.core.meta.can_resume`, `stale_reason`, ... | `tslab.lab.core.meta.instance` |
| `tslab.lab.study.commands.submit`, `status`, `cancel`, ... | `tslab.lab.study.commands.submit.submit`, `status.status`, `cancel.cancel`, ... |
| `tslab.lab.study.definition.Study`, `Grid`, `expand_runs`, ... | `tslab.lab.study.definition.definition` |
| `tslab.lab.study.definition.After`, `chained_grids`, ... | `tslab.lab.study.definition.after` |
| `tslab.lab.study.definition.Settings`, `from_presets`, ... | `tslab.lab.study.definition.settings` |
| `tslab.lab.study.definition.load_study` | `tslab.lab.study.definition.loading.load_study` |
| `tslab.lab.study.machine.MachineProfile`, `FollowupSettings` | `tslab.lab.study.machine.profile` |
| `tslab.lab.study.machine.load_machine` | `tslab.lab.study.machine.load.load_machine` |
| `tslab.lab.study.machine.SlurmSettings`, `LocalSettings` | still exported there; also `tslab.lab.study.machine.slurm`, `local` |
| `tslab.lab.study.launchers.local.LocalBackend` | `tslab.lab.study.launchers.local.backend.LocalBackend` |
| `tslab.lab.study.launchers.slurm.SlurmBackend` | `tslab.lab.study.launchers.slurm.backend.SlurmBackend` |
| `tslab.ml.train` | `tslab.ml.stages.train` |
| `tslab.ts.encode`, `evaluate`, `check_representations`, `ModelAsEncoder` | `tslab.ts.stages.encode`, `evaluate`, `check_representations`, `ModelAsEncoder` |
| `tslab.ml.interfaces.Trainer`, `DataSource`, ... | `tslab.ml.Trainer`, `tslab.ml.DataSource`, ... |
| `tslab.ts.interfaces.Encoder`, `Probe`, ... | `tslab.ts.Encoder`, `tslab.ts.Probe`, ... |
| `tslab.lab.analysis.stars`, `format_number`, `is_dark_cell`, `complete_row_count`, `DEFAULT_*`, `FULL_WIDTH`, `PLOTS_FOLDER` | `tslab.lab.analysis.stats`, `numbers`, `style`, `output` |
| `tslab.lab.jobs.slurm.select_ids`, `command_lines`, `IdScope` | `tslab.lab.jobs.slurm.catalogue` |
| `tslab.lab.jobs.slurm.QOS_LIMIT_MARKERS` | `tslab.lab.jobs.slurm.commands.QOS_LIMIT_MARKERS` |
| `tslab.lab.jobs.slurm.STUCK_PENDING_REASONS` | `tslab.lab.jobs.slurm.states.STUCK_PENDING_REASONS` |
| `tslab.ml.lightning.AccumulatingTimerCallback` | `tslab.ml.lightning.callbacks.AccumulatingTimerCallback` |
| `tslab.ts.tasks.TSDataSource` | `tslab.ts.tasks.data.TSDataSource` |
| `tslab.ts.tasks.encode_series`, `encode_windows`, `horizon_samples` | `tslab.ts.tasks.forecasting` |

## Install extras

torch, Lightning, safetensors and scikit-learn are no longer base dependencies; `uv add tslab` installs only the lab. They moved to the extras `ml` (torch, Lightning, safetensors) and `ts` (`ml` and scikit-learn). `uv add "tslab[full]"` restores the old install and adds the chronocratic adapters and analysis packages. The extra `wandb-sync` installs only W&B, enough for `tslab wandb-sync` on a login node without torch; `tslab study submit` with `followup.wandb_sync` needs `tslab[wandb]` or `tslab[wandb-sync]` in the submitting environment.
