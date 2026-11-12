---
name: tslab
description: Run, follow, debug and collect TSLab studies of this project from the command line.
---

# TSLab study workflow

Any coding agent with shell access can use this workflow to run or inspect a study. The
commands return JSON so you can make decisions from fields, not tables.

The study argument is the short name `example`; `tslab.toml` at the project
root expands it to `<package>.studies.example:study`. Each package defines its
study in `study.py` and exposes `study`; shared definitions live in
`studies/common/`. Its `[machine_by_host]` table can pick the machine by host,
so `--machine` can be dropped from every command.

## Seeding

Import `tslab.ml` in any stage that uses torch. The lab seeds Python `random` and
NumPy; `tslab.ml` adds torch seeding and the torch, CUDA and GPU metadata.

## Before submitting

For a new machine, run its readiness checks before the first submit:

```bash
uv run tslab study list --json
uv run tslab doctor --machine local example --json
```

Stop on failed checks. Resolve warnings with the researcher when they affect the
requested work. For an untested cluster profile, run a small canary before the
full study; `doctor` checks setup, not whether a real cluster job succeeds.

Learn valid grid and axis names, then inspect the plan before launching:

```bash
uv run tslab study show example --json
uv run tslab study submit example --machine local --dry-run --json
```

Check the `schema` of each document before using its fields. Use `grids[].name` for
`--grid` and `grids[].axes` for `--only axis=value`. Review the dry-run jobs and
resources, then submit only the selection the researcher requested:

```bash
uv run tslab study submit example --machine local --detach
```

## Wait and respond to the result

Run wait in a background shell so it can finish without blocking other work.
Keep its process exit code and JSON output; do not poll by launching repeated
wait commands.

```bash
uv run tslab study wait example --machine local --json
```

Always pass `--json` to supported commands and check the `schema` field. The
status commands include `exit_code` in JSON even when `--exit-code` is omitted.
Use `--exit-code` on `tslab status` or `tslab study status` when the shell needs
the state as its process exit code. `study wait` always returns the state code.

| Code | Meaning | Next action |
|---:|---|---|
| 0 | All selected runs done | Collect results. |
| 1 | Command failed | Read standard error; fix the command or cause. |
| 2 | Usage error | Correct the flag, study, or machine name. |
| 3 | Runs queued or running | Keep waiting. |
| 4 | At least one run failed | Inspect `study failures`, fix cause, then resubmit failed runs. |
| 5 | No live runs; some interrupted or missing | Resubmit with default states. |
| 6 | SLURM job states unknown | Report the problem. Do not resubmit. |
| 124 | Wait timed out | Wait again or report; if SLURM is unknown, do not resubmit. |

For code 4, inspect the error details and log tails before changing code or
settings. The `hint` field of each failed run states what the data shows — a
value to raise, or a resubmit — and nothing else. After fixing the cause,
retry only failed runs:

```bash
uv run tslab study failures example --machine local --json
uv run tslab study resubmit example --machine local --states failed --json
```

To read the log of one run while it runs, select exactly one run. Never pass
`-f`: it waits for new lines until Ctrl-C. Read the last lines again instead:

```bash
uv run tslab study logs example --machine local --only <axis>=<value> -n 100
```

To reproduce one run in the terminal, select exactly one run. `study run`
runs the command of the run, prints to the terminal, and records nothing; it
refuses a queued or running run and does not call SLURM:

```bash
uv run tslab study run example --machine local --only <axis>=<value>
```

`--include-done` belongs to `study submit`, not `study resubmit`. It submits
every non-live run in the selected slice, including failed, interrupted, and
missing runs. If the researcher asks to rerun done runs too, check that broader
selection against the request; ask before submitting any unwanted states.

For code 5, the default resubmit selection is interrupted and missing runs:

```bash
uv run tslab study resubmit example --machine local --json
```

When a group needs more time or memory, size the change before editing the
machine profile. `study usage` shows what each run used, and the `groups`
field suggests a `time` and a `mem_per_cpu` per resource group. It is SLURM
only and exits 1 on a local machine.

```bash
uv run tslab study usage example --machine local --json
```

To test the timeout path without waiting for the time limit, `study cancel
--signal` sends `USR2` to the running jobs of the selection: the runs save a
checkpoint and end on their own as `interrupted`, and queued jobs are skipped.
It never cancels.

```bash
uv run tslab study cancel example --machine local --signal --json
```

After either resubmit, start another background wait and handle its exit code
the same way. For code 6, do not retry until the researcher resolves unknown
SLURM status and authorizes the next action.

For code 0, collect the results as JSON on standard output:

```bash
uv run tslab study results example --machine local --out - --json
```

## Safety

- Never pass `--include-done` or `--cancel-live` unless the researcher asks.
- Never pass `--include-done` to `study resubmit`; that command selects states
  with `--states` instead.
- Never run `tslab study cancel` unless the researcher asks. With `--signal`,
  it sends `USR2` to the running jobs instead of canceling them: the runs
  save and end on their own.
- Inspect failures before retrying them; change code or settings before using
  `--states failed`.
- Unknown SLURM states are not evidence that jobs stopped. Report them; do not
  resubmit or cancel based on unknown state.
- When a submit with a code snapshot stops (exit 2) listing untracked files
  under `src/`, ask the researcher: add them with `git add`, pass
  `--code-snapshot-include-untracked`, or drop the snapshot with
  `--no-code-snapshot`. Never pick a flag on your own.
