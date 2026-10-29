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

## JSON documents

| Schema | Fields |
|---|---|
| `tslab.error/1` | `exit_code`, `message`; used for command and usage errors. |
| `tslab.machine-list/1` | `machines`: `name`, `source`, `profile`, `note`. |
| `tslab.run-status/1` | `exit_code`, `runs`: `directory`, `state`, `instances` (`instance`, `stage`, `state`, `fingerprint`, `seconds`, `error`). |
| `tslab.study-status/1` | `study`, `machine`, `study_directory`, `job_states_known`, `exit_code`, `counts`, `runs`. Counts include every state and `total` per grid and under `all`. Each run has `run_path`, `grid`, `values`, `state`, `reason`, `job_id`, `job_state`, `attempt`, `directory`. |
| `tslab.study-show/1` | `study`, `grids`, `total_runs`. Each grid has `name`, `short_name`, `command`, `axes` (axis to value names), `excludes`, `overrides`, and `runs`. |
| `tslab.study-launch/1` | `dry_run`, `jobs`, `cancelled`, `exit_code`. Each job has `job_name`, `job_id` (null on dry run), `runs`, `indices`, `resources`, `first_command`. Used by submit and resubmit. |
| `tslab.study-wait/1` | All `tslab.study-status/1` fields, plus `waited_seconds` and `timed_out`. |
| `tslab.study-failures/1` | `runs`: `run_path`, `state`, `reason`, `job_id`, `job_state`, `attempt`, `directory`, `errors` (`instance`, `error_file`, `error`), `log_file`, `log_tail`, `run_log`, `run_log_tail`. |
| `tslab.study-cancel/1` | `cancelled` (job ids), `dry_run`, `exit_code`. |
| `tslab.study-results/1` | `rows`, the same rows as the CSV. |
| `tslab.doctor/1` | `machine`, `checks` (`name`, `status` as `ok`, `warn`, or `fail`, `message`), `exit_code`. |

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

## Worked study session

Set references from the researcher request. Run doctor before the first submit
on a new machine. For a cluster profile that has never launched a job, use a
small canary after setup checks; doctor does not prove that a real cluster job
will succeed.

```bash
STUDY=my_paper.studies:example
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

Run wait in a background shell and retain its process exit code and JSON output:

```bash
uv run tslab study wait "$STUDY" --machine "$MACHINE" --json > wait.json &
wait_pid=$!
# Continue other work while wait runs. Later, collect its exit code:
wait "$wait_pid"
```

When wait returns 4, inspect errors and log tails before changing code or
settings. After fixing the cause, retry only failed runs:

```bash
uv run tslab study failures "$STUDY" --machine "$MACHINE" --json
uv run tslab study resubmit "$STUDY" --machine "$MACHINE" --states failed --json
```

When it returns 5, resubmit the default interrupted and missing states:

```bash
uv run tslab study resubmit "$STUDY" --machine "$MACHINE" --json
```

Start another background wait after resubmitting. Repeat failure review and
retry only when the exit code and researcher request call for it. For exit 6,
report the unknown SLURM state and do not resubmit. On exit 0, collect results:

```bash
uv run tslab study results "$STUDY" --machine "$MACHINE" --out - --json
```

## Safety

- Never pass `--include-done` or `--cancel-live` unless the researcher asks.
- `--include-done` belongs to `study submit`, not `study resubmit`. It selects
  every non-live run in the chosen slice, including failed, interrupted, and
  missing runs. Check that scope before submitting; ask if it exceeds the request.
- Never run `tslab study cancel` unless the researcher asks. Use `--dry-run` to
  inspect selected job ids first.
- Inspect failures before retrying them. Change code or settings before using
  `--states failed`.
- Unknown SLURM state does not mean the job stopped. Report it; do not resubmit
  or cancel based on an unknown state.
