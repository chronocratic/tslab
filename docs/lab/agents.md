# Driving studies with an agent

Use `tslab` JSON output and exit codes when a coding agent submits, follows,
debugs, or collects a study. With `--json`, commands write one document to
standard output, except `study results`, which writes to `--out`. Set `--out -`
for standard output. Logs and task output go to standard error. Read the
`schema` field first; breaking document changes need a new schema version.

New projects include the general skill at
`.agents/skills/tslab/SKILL.md`, and `AGENTS.md` points to it. Compatible
agents can discover the shared skill path directly; any agent reading
`AGENTS.md` can follow its link.

Paths in documents are absolute POSIX paths. `tslab status` and
`tslab study status` include an `exit_code` field in JSON but keep returning
process code 0 by default. Add `--exit-code` when a shell needs the state as its
process code. `tslab study wait` always returns the state code.

Study commands take the short study name of [tslab.toml](project-settings.md);
with its `[machine_by_host]` table, `--machine` can be dropped:

```bash
uv run tslab study submit example --json
```

## JSON documents

| Schema | Fields |
|---|---|
| `tslab.error/1` | `exit_code`, `message`; used for command and usage errors. |
| `tslab.machine-list/1` | `machines`: `name`, `source`, `profile`, `note`. |
| `tslab.study-list/1` | `studies`: `short_name`, `name`. |
| `tslab.run-status/1` | `exit_code`, `runs`: `directory`, `state`, `instances` (`instance`, `stage`, `state`, `fingerprint`, `seconds`, `error`). |
| `tslab.study-status/1` | `study`, `machine`, `study_directory`, `job_states_known`, `exit_code`, `counts`, `runs`. Counts include every state and `total` per grid and under `all`. Each run has `run_path`, `grid`, `values`, `state`, `reason`, `job_id`, `job_state`, `attempt`, `directory`, and, when the scheduler reports a start for a queued job, `expected_start` (null when unknown), `pending_reason` (null when unknown) and `pending_stuck` (bool, false when no reason). |
| `tslab.study-show/1` | `study`, `settings` (dotted path to value), `grids`, `total_runs`. Each grid has `name`, `short_name`, `job_name` (the template), `command`, `axes` (axis to value names), `excludes`, `overrides`, and `runs`. |
| `tslab.study-launch/1` | `dry_run`, `jobs`, `cancelled`, `exit_code`. Each job has `job_name`, `job_id` (null on dry run), `runs`, `indices`, `resources`, `first_command`. Used by submit and resubmit. |
| `tslab.study-wait/1` | The `tslab.study-status/1` fields without `expected_start`, `pending_reason` and `pending_stuck` on runs, plus `waited_seconds` and `timed_out`. |
| `tslab.study-failures/1` | `runs`: `run_path`, `state`, `reason`, `job_id`, `job_state`, `attempt`, `directory`, `errors` (`instance`, `error_file`, `error`), `log_file`, `log_tail`, `run_log`, `run_log_tail`, `hint` (what the data suggests, or null), `exit_code` (the code SLURM recorded for the job, or null). |
| `tslab.study-cancel/1` | `cancelled` (job ids), `dry_run`, `signal` (`USR2` when sent with `--signal`, else null), `exit_code`. |
| `tslab.study-usage/1` | `study`, `machine`, `runs`, `groups`. Each run has `run_path`, `job_id`, `state`, `elapsed`, `time_limit`, `max_rss` and `requested_memory` (bytes), `total_cpu_seconds`, `alloc_cpus`, `gpus`, `paused`. Each group has `runs`, `slurm` (resource line), `suggested_time`, `suggested_mem_per_cpu`. |
| `tslab.study-results/1` | `rows`, the same rows as the CSV. |
| `tslab.doctor/1` | `machine`, `project_settings` (`path`, `values`: each key with `value` and `source` as `flag`, `file`, or `default`), `checks` (`name`, `status` as `ok`, `warn`, or `fail`, `message`), `exit_code`. |

## Exit codes

Use these codes for `tslab status --exit-code`,
`tslab study status --exit-code`, and `tslab study wait`:

| Code | Meaning | Next action |
|---:|---|---|
| 0 | Every selected run is done. | Collect results. |
| 1 | The command failed, such as a submission error or bad plan. | Read standard error and fix the cause. |
| 2 | Usage error, such as a bad flag, study, or machine. | Fix the command. |
| 3 | Runs are queued or running; none failed. | Keep waiting. |
| 4 | At least one run failed. | Inspect `study failures`; fix the cause before retrying failed runs. |
| 5 | No live runs; some are interrupted or missing. | Resubmit with default states. |
| 6 | SLURM job states are unknown. | Report the issue; do not resubmit. |
| 124 | `study wait` reached `--timeout`. | Wait again or report. If SLURM state is unknown, do not resubmit. |

Priority is 6, then 4, then 3, then 5, then 0. An empty selection is a usage
error (2), not success. `study wait` treats unknown SLURM state (6) as live and
keeps checking; it warns once. A timeout returns 124.

`study usage` returns 0 on success, 1 when the accounting query fails. `study run` returns the exit code of the command it
runs, 1 when the run is queued or running or the plan has problems, and 2
when the selection does not name exactly one run.

## Worked study session

Set references from the researcher request. Run doctor before the first submit
on a new machine. For a cluster profile that has never launched a job, use a
small canary after setup checks; doctor does not prove that a real cluster job
will succeed.

```bash
STUDY=example
MACHINE=local

uv run tslab doctor --machine "$MACHINE" "$STUDY" --json
uv run tslab study show "$STUDY" --json
uv run tslab study submit "$STUDY" --machine "$MACHINE" --dry-run --json
```

Check each `schema`, readiness result, grid and axis name, and planned command
and resources. Then submit the requested selection:

```bash
uv run tslab study submit "$STUDY" --machine "$MACHINE" --json
```

On a local machine, submit with `--detach` and never run `submit` in the
foreground: a foreground submit blocks until every task ends. `--detach` starts
the launch in the background and returns once it runs; its output goes to
`.study/local_logs/<pid>.log`. Then use `wait`, `status`, `failures`, `usage` and
`cancel` as below.

```bash
uv run tslab study submit "$STUDY" --machine "$MACHINE" --detach
```

Run wait in a background shell and retain its process exit code and JSON output:

```bash
uv run tslab study wait "$STUDY" --machine "$MACHINE" --json > wait.json &
wait_pid=$!
# Continue other work while wait runs. Later, collect its exit code:
wait "$wait_pid"
```

When wait returns 4, inspect errors and log tails before changing code or
settings. The `hint` field of each failed run states what the data shows — a
value to raise, or a resubmit — and nothing else. After fixing the cause,
retry only failed runs:

```bash
uv run tslab study failures "$STUDY" --machine "$MACHINE" --json
uv run tslab study resubmit "$STUDY" --machine "$MACHINE" --states failed --json
```

To read the log of one run while it runs, select exactly one run. Do not pass
`-f` from an agent: it waits for new lines until Ctrl-C. Read the last lines
again instead:

```bash
uv run tslab study logs "$STUDY" --machine "$MACHINE" --only <axis>=<value> -n 100
```

To reproduce one run in the terminal, select exactly one run. `study run`
runs the command of the run with the environment and setup of the machine,
prints to the terminal, and records nothing; it refuses a queued or running
run and does not call SLURM:

```bash
uv run tslab study run "$STUDY" --machine "$MACHINE" --only <axis>=<value>
```

When it returns 5, resubmit the default interrupted and missing states:

```bash
uv run tslab study resubmit "$STUDY" --machine "$MACHINE" --json
```

When a group needs more time or memory, size the change before editing the
profile. `study usage` shows what each run used, and the `groups` field
suggests a `time` and a `mem_per_cpu` per resource group — the largest usage
of the group plus a 20 percent margin:

```bash
uv run tslab study usage "$STUDY" --machine "$MACHINE" --json
```

On a local machine, `study usage` reads the usage file of each task, and
`study cancel` stops the live local launch with `SIGTERM`; the runs read as
`interrupted`, and `--signal` is for SLURM only.

To test the timeout path without waiting for the time limit, send the signal
to the running jobs: the runs save a checkpoint and end on their own as
`interrupted`, and queued jobs are skipped:

```bash
uv run tslab study cancel "$STUDY" --machine "$MACHINE" --signal --json
```

Start another background wait after resubmitting. Repeat failure review and
retry only when the exit code and researcher request call for it. For exit 6,
report the unknown SLURM state and do not resubmit. On exit 0, collect results:

```bash
uv run tslab study results "$STUDY" --machine "$MACHINE" --out - --json
```

## Safety

- Import `tslab.ml` in any stage that uses torch. The lab seeds Python `random` and
  NumPy only; `tslab.ml` adds torch seeding and the torch, CUDA and GPU metadata.
  A stage that uses torch without it is not seeded for torch. See
  [Seeding](run-options.md#seeding).
- Never pass `--include-done` or `--cancel-live` unless the researcher asks.
- `--include-done` belongs to `study submit`, not `study resubmit`. It selects
  every non-live run in the chosen slice, including failed, interrupted, and
  missing runs. Check that scope before submitting; ask if it exceeds the request.
- Never run `tslab study cancel` unless the researcher asks. On a local machine
  it stops the whole launch, not a slice. Use `--dry-run` to
  inspect selected job ids first. With `--signal`, the command sends `USR2` to
  the running jobs instead of cancelling them: the runs save and end on their
  own, so the command never stops a job hard.
- Inspect failures before retrying them. Change code or settings before using
  `--states failed`.
- Unknown SLURM state does not mean the job stopped. Report it; do not resubmit
  or cancel based on an unknown state.
- When a submit with a code snapshot stops (exit 2) listing untracked files
  under `src/`, ask the researcher how to proceed: `git add` the files, pass
  `--code-snapshot-include-untracked`, or drop the snapshot with
  `--no-code-snapshot`. Never pick a flag on your own.
