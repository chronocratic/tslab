# CODING_RULES.md: how to write TSLab code

Every coding convention of TSLab. [`AGENTS.md`](AGENTS.md) holds the rest:
the checks to run, how the repository is split and which package may import
what, the project template, public API promises and git.
[`CONTRIBUTING.md`](CONTRIBUTING.md) holds setup, CI, markers and changelog
fragments.

Part I and Part II are general principles and standards. Part III holds the
rules specific to TSLab; where a general rule and a TSLab rule differ, the
TSLab rule wins. Ruff (every rule on), ty (every rule an error) and the tests
in `tests/import_rules/` enforce most of Part III. Where a tool does not
enforce a rule, reviewers do.

# Part I: Principles of Effective Coding and Collaboration

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

# Part II: Coding Standards and Best Practices

TSLab is a research library for time series representation learning, built
on Python, PyTorch and Lightning. It trains, encodes and evaluates models in
large experiments, on laptops and on SLURM clusters. It emphasizes modular
design, docstrings, type hints, and functional coding patterns for clarity
and maintainability, and uses Lightning to streamline deep learning tasks.
Reproducibility and thorough testing are key, ensuring robust performance in
both HPC and local setups. The focus is well-structured, flexible
experimentation code rather than production-grade deployments.

## Code Style and Structure
- Write concise, modular code with a preference for **functional programming** patterns.
- Leverage **type hints** (in Python) and docstrings to increase clarity and maintainability. Type hints should be used for all functions, including return types; it is not optional to omit them.
- Use Google style for the docstrings.
- Use descriptive names for variables and functions (e.g., `encode_sentence`, `calculate_accuracy`).
- Keep lines concise; avoid deeply nested loops and rely on built-in library functions where possible. No nested `if` either (see Part III, Control flow).
- Always write robust code that is readable and expandable. Follow the principles introduced in the "Robust Python" book by Patrick Viafore, which emphasizes writing clear and maintainable code, using type hints and docstrings effectively, and organizing code into modular components for better readability and maintainability.
- Do not always write `from __future__ import annotations` at the top of every file. Use it when it is absolutely necessary to avoid crashes due to circular imports or when it significantly improves readability by allowing for forward references in type hints.
- **Prioritize Lightning** for all training-related code, inside `tslab.lightning` only (the engine never imports Lightning; see `AGENTS.md`):
  - Use `LightningModule` to define models and training/validation/test steps.
  - Use `LightningDataModule` or well-structured data-loading classes for consistent, scalable datasets.
  - Keep logging and checkpointing within the Lightning ecosystem (loggers, callbacks) for minimal boilerplate. Configuration stays in TSLab configs.
- For logic that doesn’t depend on deep learning, write pure functions that clearly define inputs and outputs.
- Organize code into separate packages and modules, each in its home in the layout of `AGENTS.md`:
  - Data handling and transformations: `tasks/`, `adapters/chronocratic/datasets/`.
  - Model definitions and training: `adapters/chronocratic/models/`, `lightning/`, the `train` stage in `stages/`.
  - Logging, metrics and evaluations: `tracking/`, `probes/`, the `encode` and `evaluate` stages, `Metric` in `interfaces/`.
  - Helpers live in the module of their concept (`core/text.py`, `core/paths.py`), never in a `utils.py` grab bag.
- All function calls should be using keyword arguments (kwargs) for better readability and flexibility.
- When needed, use design patterns to structure the codebase, including but not limited to the ones below. Use one when it removes a branch that repeats or when two or more implementations exist; never for single-use code.
  - **Creational Patterns:** Singleton, Factory Method, Abstract Factory, Builder, Prototype, Object Pool
    - e.g. Factory Method (a table) to pick the launcher of a machine profile, Prototype with `dataclasses.replace` for changed settings, Object Pool for the GPU leases in `tslab.jobs.local`. No Singleton: it is hidden state; pass shared objects as parameters.
  - **Structural Patterns:** Adapter, Bridge, Composite, Decorator, Facade, Flyweight, Proxy
    - e.g. Adapter for integrating third-party libraries or APIs (`tslab.adapters.chronocratic`, `LightningTrainer`), Facade for simplifying complex subsystems (`tslab/__init__.py`), Decorator for extending functionality dynamically (`@tslab.stage`).
  - **Behavioral Patterns:** Chain of Responsibility, Command, Interpreter, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor, Null Object
    - e.g. Strategy for swappable algorithms or processing pipelines (`Tracker`, `Trainer`, `DownstreamModel`, the study `Launcher`), Observer for event-driven logic (the `on_start`, `on_line`, `on_end` callbacks of `run_commands`, Lightning callbacks), Command for encapsulating actions or operations (`PoolCommand`), Template Method for fixed steps with hooks (`Stage`), Null Object to remove `None` checks (`UnavailableArtifact`), Chain of Responsibility for "the first rule that matches wins", Memento for resumable state (checkpoints, `status.json`), State for fixed state sets (`RunState`, `StudyState`), Iterator for lazy items (generators).
  - **Architectural Patterns:** MVC, MVP, MVVM, Repository, Unit of Work, CQRS, Event Sourcing, Layered Architecture, Hexagonal (Ports & Adapters), Clean Architecture, Microservices, Service Locator, Dependency Injection
    - e.g. Repository for data access abstraction (`read_jobs` and `append_job` in `study/records.py`), Dependency Injection for decoupling components (`sleep`, `environment`, `now` and `trackers` keyword parameters), Layered Architecture and Ports & Adapters for the engine against its integrations (`core/` and `interfaces/` against `lightning/`, `probes/`, `adapters/`), Unit of Work for changes kept or dropped together (`Run` as a context manager, a stage work folder published by rename). TSLab is a library, not a service or a GUI: no MVC, MVP, MVVM, CQRS, Event Sourcing, Microservices or Service Locator.
  - **Concurrency Patterns:** Active Object, Monitor Object, Half-Sync/Half-Async, Thread Pool, Reactor, Proactor, Scheduler, Read-Write Lock, Producer-Consumer
    - e.g. Thread Pool for managing parallel workloads (`run_commands` in `tslab.jobs.local`), Producer-Consumer for async data pipelines (the output lines of `run_process`), Read-Write Lock for safe concurrent access (shared and exclusive GPU locks with `tasks_per_gpu`), Monitor Object for shared state behind one lock (the task output of a local launch).

## Naming Conventions
- For **Python**, use `snake_case` for functions/variables and `PascalCase` for classes.
- TSLab has no Rust code. If Rust is added, follow official community guidelines: `snake_case` for module names and functions, `CamelCase` for structs.
- Keep test files prefixed with `test_` (e.g., `test_data_loader.py`) in Python, or placed in a `tests` module in Rust.

## Experimentation and Logging
- Use the logging features of Lightning and the local tracker of TSLab for experiment tracking. Only use [Weights & Biases](https://docs.wandb.ai/), MLflow or TensorBoard when asked to; library code never starts a tracker by itself.
- Maintain `.yaml` or `.json` configs for hyperparameters (typed config dataclasses saved as `config.yaml`) and the TOML machine profile for environment variables (`[env]`). No hyperparameter is hard-coded in a script.
- Log every run with relevant metadata (date, hyperparameters, commit hash, seeds). `Run` already writes `config.yaml` and `meta.json` (start time, seed, config hash, library versions, host, git commit); a new fact needed to reproduce a run goes there too.

## Testing and Validation
- Write **unit tests** for data preprocessing and model components:
  - In Python, use `pytest`; maintain test coverage (`pytest-cov` is installed; a change does not lower the coverage of the code it touches).
  - In Rust, use the built-in test framework (`cargo test`).
- Include integration tests for training loops (small dataset or subset), in `tests/integration/` on small synthetic data.
- Set seeds for reproducible experiments with `tslab.core.seeding.seed_everything(seed=...)`, which seeds `random`, NumPy and PyTorch; `Run` calls it for every run.

## Research Code (No Deployment)
- Assume code is local or HPC-based; skip production frameworks and containerization unless you specify otherwise.
- Provide clear instructions on environment setup: `uv` with `pyproject.toml` and `uv.lock`, described in `CONTRIBUTING.md` and the README.

## Performance and Optimization
- Prefer **vectorized operations** with PyTorch Tensors for CPU/GPU speedups.
- Profile code if needed using PyTorch profiler or `cProfile` (or `cargo flamegraph` for Rust), and say in a comment what the measurement was.
- For GPU code, minimize host-device transfers.

## Documentation and Referencing
- Do not reference papers in docstrings or comments: no author names, no years, no "as in the ... paper". Model names such as `TS2Vec` and `CoST` stay. Explain the method in your own words, so the text is complete without the paper.
- Include a **README.md** or `docs/` directory explaining how to reproduce experiments, run training scripts, and test the code. A change that alters one of these steps updates the docs in the same change.

## Security and Ethics
- Handle sensitive data responsibly and follow relevant data privacy regulations (e.g., GDPR).
- In open-source projects, include a permissive or relevant license; ensure you have rights to share datasets or weights. TSLab is BSD-3-Clause; anything copied into the repository needs a compatible license.

## Collaboration Workflow
- Use branches for new features or experiments.
- Keep commits concise and descriptive.
- Regularly run linters (`ruff`, `ty`) and test suites in CI pipelines. TSLab uses no `black` or `flake8`. Branches, commits and CI are described in `AGENTS.md` and `CONTRIBUTING.md`.

## General Notes
- Take your time to write clean, maintainable code. Avoid shortcuts that may lead to technical debt, especially in research code that may evolve over time.
- Always prioritize clarity and reproducibility over cleverness or optimization, especially in the early stages of research.
- When in doubt, refer to the official documentation of the libraries you are using (e.g., PyTorch Lightning) for best practices and recommended patterns.
- When asked something you have no clue about and want to search the whole codebase, try `/memory search` first.

## Project-Specific Guidelines
- Use Python 3.12 for all code in this project.
- Use `uv` for environment management and task execution, as specified in the project README.
- Use `ruff` for linting and code formatting, following the configurations specified in the project.
- Use the latest conventions and best practices for Python 3.12 and the used libraries as of 2026, including any new features or syntax improvements that enhance code clarity and maintainability. It is not acceptable to suggest code that is outdated or does not leverage the latest language features effectively.

---

# Part III: TSLab rules

Users debug TSLab inside `site-packages`, so every line you add is a line
they have to read.

## 1. Simplicity

1. Write the minimum code that solves the problem. No speculative features,
   no options nobody asked for, no "for later" scaffolding.
2. No abstraction with one implementation unless it is a published extension
   point. No factory for one product, no config for a value that never changes.
3. Prefer the standard library and the dependencies already in
   `pyproject.toml`. A new dependency needs a strong reason and version bounds.
4. Deletion over addition. Boring over clever.
5. **Surgical changes:** touch only what the task needs. Note unrelated
   problems in your report; do not fix them in the same change.

## 2. Control flow: no nested `if`

6. **No `if` inside the body or the `else` of another `if`.** A flat
   `if`/`elif`/`else` chain is not nesting, but a long chain that switches on
   one value is still a smell: use a table. Older code is being moved to this
   rule. Reach for these, in order:

   1. **Guard clauses.** Return, raise or `continue` early, so the main path
      stays at one indentation level.

      ```python
      # Not this
      def remaining(*, job: JobRecord) -> int:
          if job.launcher == 'slurm':
              if job.tasks:
                  return len(job.tasks)
          return 0


      # This
      def remaining(*, job: JobRecord) -> int:
          if job.launcher != 'slurm' or not job.tasks:
              return 0
          return len(job.tasks)
      ```

   2. **A helper function per branch.** When a branch needs its own decision,
      give the decision a name and a docstring.
   3. **Tables instead of `if`/`elif` on one value.** A `Mapping` from the
      value to a result or a function. For "first rule that matches wins", a
      tuple of `(predicate, result)` pairs and `next(...)`:

      ```python
      _HINTS: tuple[tuple[Callable[[RunReport], bool], str], ...] = (
          (lambda report: report.job_state == 'TIMEOUT', 'raise time'),
          (lambda report: report.job_state in {'PREEMPTED', 'NODE_FAIL'}, 'resubmit'),
      )


      def failure_hint(*, report: RunReport) -> str | None:
          return next((hint for matches, hint in _HINTS if matches(report)), None)
      ```

   4. **Strategy objects instead of the same `if` in many places.** When code
      checks the same kind (`launcher == 'slurm'`) in more than one module,
      each kind becomes an object that implements one Protocol, picked once by
      a table. The callers call methods and never branch on the kind.
   5. **Null Objects instead of `None` checks that repeat.**
      `UnavailableArtifact` stands in for an artifact that could not be made,
      so callers do not test for `None` at every step.
   6. **`match` for the shape of data** (class patterns), when each case is
      flat. A switch over kinds, names, states or reasons is a variant
      switch (rule 50), never a `match`. A `case` body follows the same rule.

7. More than two levels of blocks (loops, `with`, `try`, `if`) in one
   function means a helper is missing.

## 3. Functional style

8. **Decisions are pure.** Code that decides (plans, states, hints, argument
   lists, scripts) does not read the environment, the clock or files: the
   caller passes them as keyword parameters, such as
   `environment: Mapping[str, str]` and `now: float`.
9. **Side effects at the edges.** Reading files, running commands, writing
   logs and printing live in thin functions at the top (the CLI, the
   launchers, `Run`), so the tests of the decisions below need no temporary
   folders or fake commands.
10. **No mutation of arguments.** Return a new value: `dataclasses.replace`, a
    new tuple, a new `dict`.

## 4. Python and types

11. Python 3.12 syntax: `class Trainer[M](ABC)`, `type InstanceName = str | tuple[str, ...]`,
    `X | None` (never `Optional`), `Self`.
12. Every method that overrides a base method has `@override` (`from typing import override`).
13. No `# ty: ignore` in `src/`. Fix types by narrowing (`isinstance`), typed
    helpers, or a `cast(...)` with a comment that says why it is safe. In
    tests, `# ty: ignore[code]` only on a line that misuses an API on purpose.
14. Prefer Protocols for things users already own (data sources, encoders,
    metrics), so their objects fit without inheriting from TSLab.

## 5. Functions, calls and data

15. **Keyword-only parameters** with a leading `*`: `def fit(*, run: Run, model: M) -> Checkpoint:`.
16. Keyword arguments in calls also apply to your own functions; the only
    exception is a third-party API with positional-only parameters.
17. **Dataclasses are `@dataclass(frozen=True, kw_only=True)`**, written on one
    line. `tests/import_rules/test_dataclass_rules.py` checks every `.py` and
    `.md` file. Mappings in frozen dataclasses are read-only (`MappingProxyType`
    or a tuple of pairs), so frozen really means frozen.
18. **Paths are always `pathlib.Path`, never `str`.** Every value that is a
    location is a `Path`: parameters (public and private), return values,
    dataclass fields, attributes and local variables. Never `str` or
    `str | Path`. Public parameters are checked with
    `require_path(value=..., parameter=...)`, which raises `TypeError` for
    `str`. Build paths with `/`, `.with_suffix()` and `.parent`, never with
    `os.path` or string joins. Turn text into a `Path` once, where it enters
    (CLI argument, TOML, YAML, environment variable). Convert to `str` only at
    a third-party or subprocess boundary. Saved files store paths relative to
    the run folder with `.as_posix()`. A path relative to a study or run that code never
    opens directly is a `pathlib.PurePosixPath`; join it to a root `Path` before
    touching the disk, and save it with `.as_posix()`.
19. **Atomic writes:** write a temporary file, `fsync`, then rename. Use the
    helpers in `tslab.core.paths` (`atomic_write_text`, `atomic_write_json`).
    A crash must never leave a half-written file that looks valid.
20. **Names say what things are:** a scope prefix plus what it is
    (`TSForecastingDataInfo`, `LightningTrainer`, `DownstreamModel`), never a
    bare `Info`, `Model` or `Source`. Private module helpers start with `_`.
21. Each module that is imported by users lists its public names in `__all__`.

## 6. Errors and logging

22. **Fail loud, contain the blast radius.** Never swallow an error silently;
    one failing stage instance must not stop unrelated work. A tracker that
    fails is logged and skipped, never fails the run.
23. Assign the message, then raise: `message = f'...'; raise ValueError(message)`.
    Messages say what was wrong and what to do (`pass --name`).
24. Catch the narrowest exception. A broad `except` needs
    `# noqa: BLE001 # why: <reason>`.
25. Logging: each module defines `_LOGGER = logging.getLogger(__name__)` and
    uses `%s` arguments, not f-strings. No `print()` in library code; only
    `cli/` and `examples/` may print.

## 7. Suppressions

26. Never add a rule to a global or per-file ignore list. Fix the code, or
    suppress one line with `# noqa: CODE # why: <reason>`; a blanket `# noqa`
    is never allowed.

## 8. Formatting

27. Single quotes, line length 100, imports sorted by ruff. Run
    `uv run ruff format .`; do not hand-format against it.

## 9. Docstrings, comments and prose

28. Docstrings (Google style) carry `Args:`, `Returns:` and `Raises:`
    sections, and `Attributes:` on dataclasses. They become the API page of
    the docs.
29. Comments explain **why**, not what. Remove comments that repeat the code.
30. Comments and docs explain themselves: never refer to internal design
    notes, decision ids, issue-tracker shorthand or files that are not in the
    repository. Write the reason itself ("keeps `import tslab.core.run` free
    of Lightning").
31. No academic citations in code, docstrings or docs: no author names, no
    years, no "as in the ... paper". Model names such as `TS2Vec` and `CoST` stay.
32. Write "time series" as two words, never "time-series".
33. No possessive `'s` on things: "the run folder", not "the run's folder";
    "the settings of the recipe", not "the recipe's settings". People are fine.
34. Plain, short sentences. No marketing words.
35. Nothing personal or site-specific lands in the repository: no cluster
    partition or QOS names, no personal paths, no W&B entities, no personal
    names. Examples use placeholders such as `CHANGE_ME`. Authorship is
    "The Chronocratic Developers".

## 10. Tests

36. Test first: write the test, see it fail for the expected reason, write the
    code, see it pass, then run the full checks.
37. Tests mirror the source: code in `src/tslab/study/plan.py` is tested in
    `tests/unit/study/test_plan.py`. Tests of a package that may move to its
    own distribution (`tslab.jobs`) import no other test folder, and no other
    test folder imports them.
38. Every folder a test writes to comes from `tmp_path`. Tests never write
    into the repository.
39. Tests that use randomness call `tslab.core.seeding.seed_everything(seed=0)`
    or pass an explicit seed.
40. Test names say what they check (`test_require_path_rejects_str`),
    so they need no docstring.
41. Use the `slow` and `chronocratic` markers from `CONTRIBUTING.md`.
42. A bug fix comes with a test that fails without the fix.
43. No real datasets in the repository. Tests use small synthetic data with
    the same format.

## 11. Fingerprints and stage versions

44. Settings that change a result change the fingerprint. Settings that
    cannot change a result (device, workers, logging intervals) are
    `Annotated[..., tslab.NotFingerprinted]`.
45. A change to what a ready stage computes bumps the `version` of that
    stage, so cached results of the old code are not reused.
46. Run options (`tslab.RunOptions`) never change a fingerprint.

## 12. Modules

47. A module holds one concept. Above about 400 lines, check whether it
    holds more than one, and split it by concept if so. Line count alone is
    no reason to split: a long module with one concept stays whole.
48. A module that is not an `__init__.py` lists in `__all__` only names it
    defines. Re-exports belong in package doors.
49. Import a name from the module that defines it or from a package door,
    never through another module that imported it.

## 13. Variants

50. Never branch on which variant you handle: no `if`/`elif` chain or `match`
   over kinds, names, states or reasons. Give each variant its own object or
   function, in its own module when it is a backend, and pick it once from a
   `Mapping` (Strategy), or call a method on the object you were given.
   `if kind == 'slurm': ... elif kind == 'local': ...` is never allowed.

## 14. Choices and typed values

51. **A fixed set of choices is a `StrEnum`, never a `str` or a `Literal`.**
    Modes, kinds, states, reasons and backends are enum members:
    `class RunStatus(StrEnum)` in `core/tracker.py`, `JobPhase` in
    `study/launchers/`. A parameter or field that takes a choice is typed with
    the enum (`status: RunStatus`), never `status: str` or
    `status: Literal['ok', 'fail']`. Members use `auto()` or the lowercase text
    saved to disk, so saved files stay readable. When you touch code with a
    `Literal` set of choices, change it to a `StrEnum`. Enums are saved, read,
    shown on the CLI and fingerprinted by their value, never by member name.
52. **Parse at the boundary, use types inside.** Text from the CLI, TOML,
    YAML, JSON or the environment becomes an enum member, a `Path` or a
    dataclass where it enters, with `RunStatus(value)` or a typed parser that
    raises a clear error naming the field and the valid choices. Code past
    the boundary never compares raw strings (`if mode == 'train'`) and never
    passes them on. Turn a value back into text only when you write it out.
53. **No `bool` flag that picks a mode.** A parameter that switches between
    two behaviors is an enum (`direction: Direction`), not `minimize: bool`,
    so a third mode never needs a second flag. A `bool` stays a `bool` when it
    is a yes or no answer (`overwrite`, `dry_run`).
54. **No magic values.** A string, number or key used in more than one place,
    or that has a meaning, is an enum member or a named module constant
    (`_DISK_WARNING_BYTES`), never a bare literal repeated across modules.
