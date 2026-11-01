# AGENTS.md: contributing to TSLab

Instructions for coding agents (and humans) who change TSLab itself. If you
are building experiments *with* TSLab, read the `AGENTS.md` of the project that
`tslab new` created instead.

TSLab is a small library for resumable, cached research pipelines for time
series representation learning. The core is deliberately small and readable:
users debug it inside `site-packages`, so every line you add is a line they
have to read.

[`CONTRIBUTING.md`](CONTRIBUTING.md) is the source for setup, the checks CI
runs, test markers, branches, changelog fragments, protected files and docs.
Read it first; this file does not repeat it. It adds the checks to run, how
the code is split, the project template, public API promises and git.

[`CODING_RULES.md`](CODING_RULES.md) holds every coding convention: how to
work on a change, control flow (no nested `if`), functional style, design
patterns, types, names, errors, docstrings, prose and tests. Read it before
you write code; reviewers hold every change to it.

## 1. Before you say a change is done

Run every check listed in `CONTRIBUTING.md` and report failures as they are;
never claim a check passed without running it. Slow tests are skipped by
default, and CI runs them. Never run the whole slow suite (`uv run pytest -m ""`)
locally. When you add or edit a slow test, or change the code it covers, run only
that test by node id: `uv run pytest -m slow tests/path/test_file.py::test_name`.
`uv run ruff check --fix . && uv run ruff format .`
fixes most lint and format problems.

## 2. How the repository is split

```
src/tslab/
  __init__.py          the public API (re-exports); __all__ is the contract
  core/                the engine: Run, stage, artifacts, fingerprints, status, config, paths
  interfaces/          Protocols and ABCs: Trainer, DataSource, Encoder, DownstreamModel, Metric, Tracker
  stages/              the ready stages: train, encode, evaluate
  tasks/               classification and forecasting data shapes and helpers
  jobs/                general job helpers (slurm/, local/), no TSLab imports (planned own package)
  lightning/           LightningTrainer, LightningDataSource, resume, SLURM signals
  probes/              SVMProbe, RidgeProbe (scikit-learn)
  tracking/            W&B, MLflow, TensorBoard trackers and TrackingOptions
    wandb/             the W&B tracker, the transaction-log reader and the resync logic
  adapters/chronocratic/
    models/            chronocratic-models as TSLab models and encoders
    datasets/          chronocratic-datasets as TSLab data sources, study helpers
  study/               studies: grids, machine profiles, SLURM and local launch, results
     launchers/         the launch backends: one per launcher, SLURM and local apart
  cli/                 `tslab new | status | study | wandb-sync`
  templates/project/   the files `tslab new` copies (see section 3)
tests/
  unit/<package>/      unit tests, one folder per source package
  integration/         whole pipelines on toy data
  adapters/            chronocratic adapter tests (marker `chronocratic`)
  import_rules/        dependency direction and dataclass rules
  typing/              files with deliberate type errors, checked one by one
  template/            generates a project and runs its own checks
  examples/            runs examples/
docs/                  Sphinx + MyST, published on Read the Docs
examples/              minimal/ and advanced/ scripts
changelog.d/           towncrier fragments, one per pull request
```

### Dependency direction (enforced by `tests/import_rules/`)

- `core`, `interfaces` and `stages` never import Lightning,
  scikit-learn, any tracker SDK, chronocratic, or `tslab.adapters`,
  `tslab.lightning`, `tslab.tasks`, `tslab.probes`, `tslab.study`, `tslab.cli`.
  Tracker SDKs are imported lazily, inside the tracker that needs them.
- `core` may import `tslab.jobs.slurm` and `tslab.jobs.local`.
- `tslab.lightning` builds on Lightning; `tslab.tracking` (trackers wrap
  Lightning loggers) and `tslab.adapters.chronocratic` import it in their
  submodules, but importing `tslab.tracking` itself loads no Lightning and
  no tracker SDK.
- Only `tslab.probes` imports scikit-learn.
- Only `tslab.adapters.chronocratic` imports chronocratic; the models adapter
  never imports the datasets adapter and the other way round.
- `tslab.tasks` imports none of the optional libraries.
- `tslab.jobs` imports only the standard library. TSLab code imports from
  `tslab.jobs.slurm` or `tslab.jobs.local`, only through each domain root.

If a feature seems to need a forbidden import, the design is wrong: add an
interface in `interfaces/`, or put the code in the package that may import
the library. Never weaken a rule in `test_import_rules.py`.

### The `tslab.jobs` package

These rules keep SLURM and local job helpers movable to their own distribution.
Static checks live in `tests/import_rules/test_jobs_package.py`; API and runtime
rules are enforced by review.

1. **Standard library only.** Files in `src/tslab/jobs/` import only modules in
   `sys.stdlib_module_names` and `tslab.jobs.*`. Ask a maintainer before adding a
   third-party dependency. The static AST check
   `test_jobs_package_imports_only_stdlib_and_itself` enforces this rule.
2. **No TSLab imports.** Job code imports no `tslab.*` module except
   `tslab.jobs.*`. Keep local copies of small helpers such as `require_path` or
   `atomic_write_text` if needed. The same static AST check enforces this rule.
3. **No TSLab words.** After removing `tslab.jobs`, no package file contains
   `tslab`, in any case, except the required package docstring. Do not use TSLab
   vocabulary such as a TSLab run, a stage, a study, an attempt, a machine
   profile, a fingerprint or W&B. Pass TSLab values such as a `/tmp/tslab-...`
   prefix, banners and extra exports as parameters. The static
   `test_jobs_package_only_mentions_tslab_in_required_docstring` check enforces
   the text rule; review enforces the vocabulary rule.
4. **Domain doors in.** Code outside the package imports APIs only from
   `tslab.jobs.slurm` or `tslab.jobs.local`, never from `tslab.jobs` or an
   internal module. `test_code_outside_imports_only_jobs_domain_roots` checks
   `src/` and `tests/`, except the package and `tests/unit/jobs/`.
5. **SLURM vocabulary in the API.** Use sbatch long option names such as
   `cpus_per_task`, `mem_per_cpu`, `gres` and `array`, with
   `Mapping[str, object]`. `True` writes a flag; `False` omits it. Add no
   settings dataclasses with TSLab defaults. Review enforces this rule.
6. **No hidden state.** Read no environment variables, files or clock values
   at import time. Pass environment and time as keyword parameters, such as
   `environment: Mapping[str, str]` and `now: float`. `sbatch` may read
   `os.environ` only when called, then must filter inherited job variables.
   Review enforces this rule.
7. **One subprocess door.** Route every scheduler command through private
   `_run` in `tslab/jobs/slurm/commands.py`. Missing commands raise `SlurmError`
   with the command name. Review enforces this rule.
8. **Own tests, movable.** Keep job tests in `tests/unit/jobs/`. They may import
   the standard library, `pytest`, `tslab.jobs.slurm`, `tslab.jobs.local` and
   `tests.unit.jobs.*`; tests may import internal job modules to test private
   helpers. `test_jobs_tests_import_only_the_jobs_domains` enforces this rule.
9. **TSLab code rules still apply.** Every rule in `CODING_RULES.md` holds:
   keyword-only parameters, `__all__`, Google-style docstrings, assigned
   error messages before raising, `_LOGGER` with `%s` arguments, Python 3.12
   syntax, frozen keyword-only dataclasses. Ruff and ty enforce most of them.
10. **Minimal and general.** Generalize only what TSLab uses today. Add no
    speculative options or unused features. The package docstring is
    `General helpers to run jobs on SLURM and on local machines, with no TSLab
    dependency; planned to move to its own distribution.` The static check
    verifies the wording; review enforces minimality.

### Where new code goes

- A new engine concept: `core/`, exported from `tslab/__init__.py` only if
  users need it.
- A new extension point: a Protocol or ABC in `interfaces/`. Add one only when
  there is a published second implementation or users must plug in their own.
- An integration with a third-party library: its own package (like
  `lightning/` or `probes/`) or `adapters/<library>/`, so the core stays free of it.
- A command: `cli/<command>.py`, routed from `cli/__init__.py`.
- Tests mirror the source (see `CODING_RULES.md` Part III, rule 37).

## 3. The project template

`src/tslab/templates/project/` holds plain files that `tslab new` copies into a
folder the user owns. `{{project_name}}` (for example `my-paper`) and
`{{package_name}}` (`my_paper`) are replaced in file names and contents.

- The template is excluded from ruff and ty in this repository because the
  placeholders do not parse. `tests/template/test_template.py` (slow) creates a
  project and runs ruff, ty and pytest inside it, so template code must pass
  the same rules as the library.
- Template code is user-facing example code: keep it short, commented for
  someone new to TSLab, and runnable on the toy data in under a minute.
- When the public API changes, update the template, `examples/`, `docs/` and
  the template `AGENTS.md` and `README.md` in the same change.
- Use only the public API there (`tslab`, `tslab.stages`, `tslab.lightning`,
  `tslab.tracking`, `tslab.adapters...`), never private `_` names.

## 4. Public API and compatibility

- `tslab/__init__.py` and the `__all__` of each subpackage are a promise to
  users. Adding a name is cheap; renaming or removing one is a breaking change
  and needs a `removed` or `changed` changelog fragment.
- Fingerprints and stage versions: see `CODING_RULES.md` Part III, section 11.

## 5. Git and pull requests

Branches, commit flow, changelog fragments and protected files: see
`CONTRIBUTING.md`. In addition:

- Commit subjects: a conventional prefix (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `ci:`, `chore:`) and an imperative summary under 72 characters.
  One logical change per commit.
- No `Co-Authored-By:` or other authorship trailers in commit messages.
- Never push, tag, release or upload unless a maintainer asks you to.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
