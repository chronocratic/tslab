Chained `failed` dependencies now wait through retryable upstream interruptions,
`started` delays use the upstream SLURM start time, and dependency failures
propagate through chains of any depth. `{after:<grid>}` now expands to an
absolute run folder. The project template demonstrates separate SVM and Ridge
probes that reuse one upstream checkpoint.
