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
Read it first; this file does not repeat it. It adds how the code is split and
the code rules reviewers hold every change to.

## 1. Before you say a change is done

Run every check listed in `CONTRIBUTING.md` and report failures as they are;
never claim a check passed without running it. Slow tests are skipped by
default: run `uv run pytest -m ""` when you touch training, resume, the
template or the study launcher. `uv run ruff check --fix . && uv run ruff format .`
fixes most lint and format problems.

## 2. How the repository is split

```
src/tslab/
  __init__.py          the public API (re-exports); __all__ is the contract
  core/                the engine: Run, stage, artifacts, fingerprints, status, config, paths
  helpers/             small helpers for the layers above the engine (text tables)
  interfaces/          Protocols and ABCs: Trainer, DataSource, Encoder, DownstreamModel, Metric, Tracker
  stages/              the ready stages: train, encode, evaluate
  tasks/               classification and forecasting data shapes and helpers
  lightning/           LightningTrainer, LightningDataSource, resume, SLURM signals
  probes/              SVMProbe, RidgeProbe (scikit-learn)
  tracking/            W&B, MLflow, TensorBoard trackers and TrackingOptions
  adapters/chronocratic/
    models/            chronocratic-models as TSLab models and encoders
    datasets/          chronocratic-datasets as TSLab data sources, study helpers
  study/               studies: grids, machine profiles, SLURM and local launch, results
  cli/                 `tslab new | status | study | wandb-sync`
  templates/project/   the files `tslab new` copies (see section 6)
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

- `core`, `interfaces`, `stages` and `tracking` never import Lightning,
  scikit-learn, any tracker SDK, chronocratic, or `tslab.adapters`,
  `tslab.lightning`, `tslab.tasks`, `tslab.probes`, `tslab.study`, `tslab.cli`.
  Tracker SDKs are imported lazily, inside the tracker that needs them.
- Only `tslab.lightning` imports Lightning.
- Only `tslab.probes` imports scikit-learn.
- Only `tslab.adapters.chronocratic` imports chronocratic; the models adapter
  never imports the datasets adapter and the other way round.
- `tslab.tasks` imports none of the optional libraries.

If a feature seems to need a forbidden import, the design is wrong: add an
interface in `interfaces/`, or put the code in the package that may import
the library. Never weaken a rule in `test_import_rules.py`.

### Where new code goes

- A new engine concept: `core/`, exported from `tslab/__init__.py` only if
  users need it.
- A new extension point: a Protocol or ABC in `interfaces/`. Add one only when
  there is a published second implementation or users must plug in their own.
- An integration with a third-party library: its own package (like
  `lightning/` or `probes/`) or `adapters/<library>/`, so the core stays free of it.
- A command: `cli/<command>.py`, routed from `cli/__init__.py`.
- Tests mirror the source: code in `src/tslab/study/plan.py` is tested in
  `tests/unit/study/test_plan.py`.

## 3. Clean code rules

These are enforced by ruff (every rule on), ty (every rule an error) and the
tests in `tests/import_rules/`. Where a tool does not enforce one, reviewers do.

### Simplicity

1. Write the minimum code that solves the problem. No speculative features,
   no options nobody asked for, no "for later" scaffolding.
2. No abstraction with one implementation unless it is a published extension
   point. No factory for one product, no config for a value that never changes.
3. Prefer the standard library and the dependencies already in
   `pyproject.toml`. A new dependency needs a strong reason and version bounds.
4. Deletion over addition. Boring over clever.
5. **Surgical changes:** touch only what the task needs. Note unrelated
   problems in your report; do not fix them in the same change.

### Python and types

6. Python 3.12 syntax: `class Trainer[M](ABC)`, `type InstanceName = str | tuple[str, ...]`,
   `X | None` (never `Optional`), `Self`. No `from __future__ import annotations`
   unless a circular import forces it.
7. Every method that overrides a base method has `@override` (`from typing import override`).
8. No `# ty: ignore` in `src/`. Fix types by narrowing (`isinstance`), typed
   helpers, or a `cast(...)` with a comment that says why it is safe. In
   tests, `# ty: ignore[code]` only on a line that misuses an API on purpose.
9. Prefer Protocols for things users already own (data sources, encoders,
    metrics), so their objects fit without inheriting from TSLab.

### Functions, calls and data

10. **Keyword-only parameters** with a leading `*`: `def fit(*, run: Run, model: M) -> Checkpoint:`.
11. Keyword arguments in calls (see `CONTRIBUTING.md`) also apply to your own
    functions; the only exception is a third-party API with positional-only parameters.
12. **Dataclasses are `@dataclass(frozen=True, kw_only=True)`**, written on one
    line. `tests/import_rules/test_dataclass_rules.py` checks every `.py` and
    `.md` file. Mappings in frozen dataclasses are read-only (`MappingProxyType`
    or a tuple of pairs), so frozen really means frozen.
13. **Paths are `pathlib.Path`.** Every public parameter that is a location is
    typed `Path`, never `str` or `str | Path`, and is checked with
    `require_path(value=..., parameter=...)`, which raises `TypeError` for `str`.
    Convert to `str` only at a third-party boundary. Saved files store paths
    relative to the run folder with `.as_posix()`.
14. **Atomic writes:** write a temporary file, `fsync`, then rename. Use the
    helpers in `tslab.core.paths` (`atomic_write_text`, `atomic_write_json`).
    A crash must never leave a half-written file that looks valid.
15. **Names say what things are:** a scope prefix plus what it is
    (`TSForecastingDataInfo`, `LightningTrainer`, `DownstreamModel`), never a
    bare `Info`, `Model` or `Source`. Private module helpers start with `_`.
16. Each module that is imported by users lists its public names in `__all__`.

### Errors and logging

17. **Fail loud, contain the blast radius.** Never swallow an error silently;
    one failing stage instance must not stop unrelated work. A tracker that
    fails is logged and skipped, never fails the run.
18. Assign the message, then raise: `message = f'...'; raise ValueError(message)`.
    Messages say what was wrong and what to do (`pass --name`).
19. Catch the narrowest exception. A broad `except` needs
    `# noqa: BLE001 # why: <reason>`.
20. Logging: each module defines `_LOGGER = logging.getLogger(__name__)` and
    uses `%s` arguments, not f-strings. No `print()` in library code; only
    `cli/` and `examples/` may print.

### Suppressions

21. Never add a rule to a global or per-file ignore list. Fix the code, or
    suppress one line with `# noqa: CODE # why: <reason>`; a blanket `# noqa`
    is never allowed.

### Formatting

22. Single quotes, line length 100, imports sorted by ruff. Run
    `uv run ruff format .`; do not hand-format against it.

## 4. Docstrings, comments and prose

23. Docstrings (Google style, see `CONTRIBUTING.md`) carry `Args:`, `Returns:`
    and `Raises:` sections, and `Attributes:` on dataclasses. They become the
    API page of the docs.
24. Comments explain **why**, not what. Remove comments that repeat the code.
25. Comments and docs explain themselves: never refer to internal design
    notes, decision ids, issue-tracker shorthand or files that are not in the
    repository. Write the reason itself ("keeps `import tslab.core.run` free
    of Lightning").
26. No academic citations in code, docstrings or docs: no author names, no
    years, no "as in the ... paper". Model names such as `TS2Vec` and `CoST` stay.
27. Write "time series" as two words, never "time-series".
28. No possessive `'s` on things: "the run folder", not "the run's folder";
    "the settings of the recipe", not "the recipe's settings". People are fine.
29. Plain, short sentences. No marketing words.
30. Nothing personal or site-specific lands in the repository: no cluster
    partition or QOS names, no personal paths, no W&B entities, no personal
    names. Examples use placeholders such as `CHANGE_ME`. Authorship is
    "The Chronocratic Developers".

## 5. Tests

31. Test first: write the test, see it fail for the expected reason, write the
    code, see it pass, then run the full checks.
32. Every folder a test writes to comes from `tmp_path`. Tests never write
    into the repository.
33. Tests that use randomness call `tslab.core.seeding.seed_everything(seed=0)`
    or pass an explicit seed.
34. Test names say what they check (`test_require_path_rejects_str`),
    so they need no docstring.
35. Use the `slow` and `chronocratic` markers from `CONTRIBUTING.md`.
36. A bug fix comes with a test that fails without the fix.

## 6. The project template

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

## 7. Public API and compatibility

- `tslab/__init__.py` and the `__all__` of each subpackage are a promise to
  users. Adding a name is cheap; renaming or removing one is a breaking change
  and needs a `removed` or `changed` changelog fragment.
- Settings that change a result change the fingerprint. Settings that cannot
  change a result (device, workers, logging intervals) are
  `Annotated[..., tslab.NotFingerprinted]`.
- A change to what a ready stage computes bumps the `version` of that stage,
  so cached results of the old code are not reused.
- Run options (`tslab.RunOptions`) never change a fingerprint.

## 8. Git and pull requests

Branches, commit flow, changelog fragments and protected files: see
`CONTRIBUTING.md`. In addition:

- Commit subjects: a conventional prefix (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `ci:`, `chore:`) and an imperative summary under 72 characters.
  One logical change per commit.
- No `Co-Authored-By:` or other authorship trailers in commit messages.
- Never push, tag, release or upload unless a maintainer asks you to.
