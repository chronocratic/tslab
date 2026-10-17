# Resume and failures

## Running again

Running the same command again is always safe:

- finished instances with matching fingerprints are **loaded**;
- `train` resumes from its last checkpoint, mid-epoch when the data loader is
  stateful (the `StatefulDataLoader` of `torchdata`); a resumed training reaches the
  same weights as an uninterrupted one;
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

## Killed runs

A run killed from outside (a SLURM time limit, a lost node, `kill -9`) cannot
record its end. `status.json` still says `running`, but the lock file stops
receiving its heartbeat, so after ten minutes:

- `tslab status` shows the run as `interrupted`;
- the next run of the same command takes the lock over, on any machine, and
  resumes.
