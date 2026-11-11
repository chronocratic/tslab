"""The ``slurm.sh`` file of a study: the SLURM command catalogue as bash functions."""

from datetime import datetime, timedelta
from pathlib import Path
import shlex

from tslab.lab.core.files import atomic_write_text
from tslab.lab.jobs.slurm import (
    CommandValues,
    QUEUED_JOB_STATES,
    RUNNING_JOB_STATES,
    SLURM_COMMANDS,
    SlurmCommand,
)
from tslab.lab.study.definition.study import (
    followup_job_name,
    RETRY_JOB,
    Study,
    study_job_names,
    WANDB_SYNC_JOB,
)
from tslab.lab.study.launchers.slurm.records import SlurmJobRecord
from tslab.lab.study.records import read_jobs

__all__ = ['SLURM_FILE', 'command_values', 'slurm_file_text', 'write_slurm_file']

SLURM_FILE = 'slurm.sh'

_MIN_WIDTH = 8
_UNKNOWN_JOB = 'Invalid job id specified'

_HELPERS = """\
# _tslab_pick <all|live|running>: sets _tslab_picked (comma-separated ids) and
# _tslab_fell_back (1 when sacct failed and every recorded id is used).
_tslab_pick() {
  _tslab_fell_back=0
  _tslab_picked=$_tslab_job_ids
  [ "$1" = all ] && return 0
  local rows
  rows=$(command sacct --jobs "$_tslab_job_ids" --noheader --parsable2 --allocations \\
    --starttime 1970-01-01 --format=JobID,State 2>/dev/null) || {
    _tslab_fell_back=1
    echo 'sacct did not answer; asking every recorded job id' >&2
    return 0
  }
  _tslab_picked=$(printf '%%s\\n' "$rows" | awk -F'|' -v scope="$1" '
    BEGIN {
      n = split("%(live)s", l, " "); for (i = 1; i <= n; i++) live[l[i]] = 1
      n = split("%(running)s", r, " "); for (i = 1; i <= n; i++) running[r[i]] = 1
    }
    {
      split($2, w, " "); state = w[1]
      if (scope == "live" && state in live) { split($1, p, "_"); id = p[1] }
      else if (scope == "running" && state in running && index($1, "[") == 0) id = $1
      else next
      if (!(id in seen)) { seen[id] = 1; out = out (out == "" ? "" : ",") id }
    }
    END { print out }')
}

# _tslab_run <tool> <args>...: runs the real tool; after the sacct fallback, ids that
# are purged from the controller are not an error.
_tslab_run() {
  local err code
  err=$(mktemp)
  command "$@" 2>"$err"
  code=$?
  cat "$err" >&2
  if [ "$code" -ne 0 ] && [ "$_tslab_fell_back" = 1 ] \\
    && grep -q '%(unknown)s' "$err"; then
    code=0
  fi
  rm -f "$err"
  return "$code"
}
"""


def command_values(*, study: Study, study_folder: Path) -> CommandValues:
    """The values that fill the commands: recorded job ids, name width, sacct start date.

    Args:
        study: The study the jobs belong to.
        study_folder: The ``.study`` folder holding ``submissions.jsonl``.
    """
    # why: a local record holds a host-pid id that no SLURM command accepts
    records = [
        record
        for record in read_jobs(path=study_folder / 'submissions.jsonl')
        if record.job_id.isdecimal()
    ]
    names = {name for group in study_job_names(study=study).values() for name in group}
    # why: a job from a submit with an older template can have a longer name
    names |= {
        record.job_name
        for record in records
        if isinstance(record, SlurmJobRecord) and record.job_name is not None
    }
    names |= {
        followup_job_name(study=study.name, kind=kind) for kind in (RETRY_JOB, WANDB_SYNC_JOB)
    }
    earliest = min((datetime.fromisoformat(record.time) for record in records), default=None)
    start = ((earliest or datetime.now().astimezone()) - timedelta(days=1)).date().isoformat()
    return CommandValues(
        job_ids=tuple(dict.fromkeys(record.job_id for record in records)),
        width=max(_MIN_WIDTH, *(len(name) for name in names)),
        start=start,
    )


def _argument(*, text: str, values: CommandValues) -> str:
    """One argument as a bash word; ``{ids}`` and ``{id}`` stay shell variables."""
    if text == '{ids}':
        return '"$_tslab_picked"'
    if text == '{id}':
        return '"$id"'
    filled = text.replace('{width}', str(values.width)).replace('{start}', values.start)
    return shlex.quote(filled)


def _function(*, command: SlurmCommand, values: CommandValues) -> str:
    words = ' '.join(_argument(text=text, values=values) for text in command.arguments)
    call = f'_tslab_run {command.tool} {words}'
    body = (
        f'  local code=0\n'
        f'  for id in $(printf %s "$_tslab_picked" | tr , " "); do {call} || code=$?; done\n'
        f'  return "$code"'
        if command.per_job
        else f'  {call}'
    )
    return (
        f'# {command.name}: {command.description}\n'
        f'{command.name}() {{\n'
        f'  local id\n'
        f'  _tslab_pick {command.scope.value}\n'
        f'  [ -n "$_tslab_picked" ] || return 0\n'
        f'{body}\n'
        f'}}\n'
    )


def slurm_file_text(*, study: Study, values: CommandValues) -> str:
    """The bash text of ``slurm.sh``: one function per catalogue command.

    Args:
        study: The study the file is for.
        values: The values that fill the commands.
    """
    helpers = _HELPERS % {
        'live': ' '.join(sorted(QUEUED_JOB_STATES | RUNNING_JOB_STATES)),
        'running': ' '.join(sorted(RUNNING_JOB_STATES)),
        'unknown': _UNKNOWN_JOB,
    }
    header = (
        f'# SLURM commands for study {study.name}, written by tslab; '
        f'rebuilt on every submit and resubmit.\n'
        f'# Use: source <path>/{SLURM_FILE}, then call a command by name. '
        f'Needs only SLURM on PATH.\n'
        f"_tslab_job_ids='{','.join(values.job_ids)}'\n\n"
    )
    functions = '\n'.join(_function(command=command, values=values) for command in SLURM_COMMANDS)
    return header + helpers + '\n' + functions


def write_slurm_file(*, study: Study, study_folder: Path) -> Path:
    """Write ``slurm.sh`` into the study folder atomically; return its path."""
    values = command_values(study=study, study_folder=study_folder)
    path = study_folder / SLURM_FILE
    atomic_write_text(path=path, text=slurm_file_text(study=study, values=values))
    return path
