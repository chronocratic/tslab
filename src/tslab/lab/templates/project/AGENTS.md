# AGENTS.md: working in {{project_name}}

Instructions for coding agents (and humans) who run and extend the experiments
in this project. The project was created by `tslab new --template lab` and
belongs to you: change any file here. TSLab itself is a library installed from
PyPI; never edit it inside `.venv/`. The full guide is at
<https://tslab.readthedocs.io>; `README.md` has the short version.

## 1. What TSLab gives you

- **Run:** `with tslab.lab.Run(config=..., root=..., relative_path=...) as run:`
  owns one folder, locks it, seeds Python `random` and NumPy, writes
  `config.yaml`, `meta.json` and `status.json`, and starts the trackers.
- **Stage:** a cached, resumable step; a function decorated with
  `@tslab.lab.stage`. A script calls stages like ordinary functions, in any
  order, in loops and conditions.
- **Instance:** one call of a stage in a run. A stage called several times in
  one run gets a `name` per call, for example `measure[a]`.
- **Fingerprint:** a hash of the stage name and `version`, the seed, the
  settings passed in and the fingerprints of the input artifacts. A matching
  fingerprint loads the saved result instead of computing it. Anything else
  recomputes the instance and everything after it. Bump `version` after
  changing the code of a stage.
- **Artifact:** what a stage returns, saved atomically: `tslab.lab.Json` or
  `tslab.lab.Metrics` (also sent to the trackers).

Running the same command again is always safe: finished work is loaded, failed
and skipped instances run again.

## 2. How the project is split

```
pyproject.toml                  dependencies and tool settings
README.md                       quick start for humans
machines/
  local.toml                    machine profile: run study tasks on this machine
  slurm.example.toml            machine profile template for a SLURM cluster (copy, fill CHANGE_ME)
src/{{package_name}}/
  stages.py                     your own stages (example: measure)
  experiment.py                 script: config and one run that calls the stages
  studies/example/study.py      example study over the script
tests/
  test_smoke.py                 runs the script once; dry-runs the study
runs/                           run folders (git-ignored)
```

## 3. Commands

```bash
uv run pytest
uv run python -m {{package_name}}.experiment --help
uv run tslab study submit example --machine local --dry-run
uv run tslab study submit example --machine local
uv run tslab study status example --machine local  # add --watch 60 to refresh
uv run tslab study wait example --machine local
uv run tslab study failures example --machine local
uv run tslab study show example
```

Check the plan with `--dry-run` before a real submit. Never edit files under
`runs/` by hand.

## 4. Rules

- Put experiment settings on the config dataclass or the study, machine
  settings only in `machines/*.toml`.
- A stage takes keyword-only arguments, including `run`, and returns an artifact.
- A stage that uses deep learning code must import `tslab.ml`. The lab seeds Python `random` and NumPy only; importing `tslab.ml` adds seeding and hardware metadata for that code. Without the import, the stage is not seeded for it.
- Keep tests fast; add a smoke test for every new script.
