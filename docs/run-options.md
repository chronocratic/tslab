# Run options

Add `run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)` to your
config and these flags exist. They never change a fingerprint.

| Flag | Effect |
|---|---|
| `--run.force-all` | run every instance again |
| `--run.rerun-from STAGE` | run this stage and every stage called after it again |
| `--run.rerun ID ...` | run these instances again, for example `--run.rerun 'transform[forecasting/h24/smooth]'` |
| `--run.fail-fast` | stop at the first failing instance |
| `--run.dry-run` | print what would be loaded or run; write nothing |
| `--run.log-level LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR`; in Python, a `tslab.lab.LogLevel` member |

A `--run.rerun` id or `--run.rerun-from` stage that matches nothing gives a
warning at the end of the run, so a typo does not go unnoticed.
