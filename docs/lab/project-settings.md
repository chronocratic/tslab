# Project settings

A `tslab.toml` at the project root holds the settings every study command of
the project shares: the studies package, the machine per host, and the default
runs folder. It is committed with the project.

```toml
# tslab.toml
studies = "my_paper.studies"
runs = "runs"

[machine_by_host]
"login*.hpc.example" = "hpc"
"*" = "local"
```

## Where the file is found

The command looks for `tslab.toml` in the working folder and in each folder
above it; the nearest file wins, so a project nested in another uses its own
file. Without a file, the built-in defaults hold and nothing errors.

## Keys

| Key | Default | Meaning |
|---|---|---|
| `studies` | none | The package that holds the studies. A short name expands to its `study` attribute: `ablation` means `my_paper.studies.ablation:study`. |
| `machine_by_host` | none | Host pattern to machine name. Used when `--machine` is not passed. |
| `runs` | none | The default folder of `tslab status` without a `FOLDER` argument. |
| `code_snapshot` | `false` | Take a [code snapshot](studies.md#code-snapshots) of `src/` when a study is submitted, so the jobs run the code of submit time. Resubmits keep the snapshot recorded on their runs and ignore this key. |
| `code_snapshot_check_untracked` | `true` | With a code snapshot, stop with a list of the untracked files under `src/` instead of leaving them out silently. |

Precedence: a `--machine` flag wins over the table, a full `module:attribute`
reference wins over a short name, and the file wins over the built-in
defaults. A short name is expanded before anything is loaded or recorded, so
contexts and records hold only the full reference and the machine name. Run
`tslab study list` to list the public study packages under `studies`.

## The machine per host

Patterns use shell-style matching (`*`, `?`, and character classes) and are
tried in the order of the file; the first match wins. Put a catch-all like
`"*"` last: it also matches hosts you forgot, so a typo in a host pattern
falls back to it silently. The picked machine and the file it came from are
printed on standard error.

The value is a machine name or a profile path, resolved like the value of
`--machine`: see [Machine profiles](studies.md#machine-profiles).

## The default runs folder

`runs` only sets the folder that `tslab status` reads without a `FOLDER`
argument. It does not move where runs are written: that is the `output_root`
of the machine profile.

## View the settings

`tslab doctor` shows the file in use, each key, and where its value comes
from — `flag`, `file`, or `default` — in the text output and in the
`project_settings` field of the JSON document.
