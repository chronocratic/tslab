# Resume and failures

## Running again

Running the same command again is always safe:

- finished instances with matching fingerprints are **loaded**;
- `train` resumes from its last checkpoint, mid-epoch when the data loader is
  stateful (a choice of the dataset, for example the `StatefulDataLoader` of
  `torchdata`); a resumed training reaches the same
  weights as an uninterrupted one;
- instances that failed or were skipped **run again**.

## When an instance fails

An exception inside a stage does not stop the run:

- the instance is recorded as `failed`, with the first line of the error in
  `status.json` and the full traceback in the `error.txt` of the folder;
- every instance that receives its result is `skipped`;
- the other instances run normally;
- at the end, the run prints a summary of failed and skipped instances, and
  the trackers receive a `failed` status.

Fix the cause and run again: only the failed and skipped instances run.
`--run.fail-fast` re-raises the first error instead, which is handy while
debugging.

## Checking from a script

A failed instance does not raise out of the run, so a script that needs to know
reads `status.json`:

```python
from tslab.lab.core.paths import STATUS_FILE
from tslab.lab.core.status import read_status
from tslab.lab.enums import InstanceState

snapshot = read_status(path=run_folder / STATUS_FILE)
failed = [
    instance_id
    for instance_id, record in snapshot.instances.items()
    if record.state is InstanceState.FAILED
]
```

`read_status` returns `None` when the run has not started. The `seconds` of each
record is the wall-clock time of that instance.

## Killed runs

A run killed from outside (a SLURM time limit, a lost node, `kill -9`) cannot
record its end. `status.json` still says `running`, but the lock file stops
receiving its heartbeat, so after ten minutes:

- `tslab status` shows the run as `interrupted`;
- the next run of the same command takes the lock over, on any machine, and
  resumes.

Add `--watch SECONDS` to `tslab status runs/` to refresh until Ctrl-C. Its hints
explain resuming interrupted runs and point to error files for failed instances.

## Many runs without a study

A [study](studies.md) is the supported way to run many runs: one saved
definition, one SLURM array per grid, fixed run names, status and resubmit.

For one-off batches without a study, give each run its own folder
(`--run-path ts2vec/Coffee/seed_0`) and submit one job per run. Check the
folders afterwards:

```bash
uv run tslab status runs/            # one line per run: state and instance counts
uv run tslab status runs/ --stage transform
```

Add `--watch SECONDS` to refresh run status until Ctrl-C. Run-level hints point
to failed-instance error files and explain interrupted runs; folder hints
suggest opening one run.
