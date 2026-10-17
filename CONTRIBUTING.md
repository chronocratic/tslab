# Contributing

Thank you for helping with TSLab. This page covers the development setup, the
checks every change must pass, and how changes reach a release.

## Setup

TSLab uses [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
git clone https://github.com/skafai/tslab
cd tslab
git switch dev
uv sync --all-extras --group docs
```

## Checks

Every pull request runs these in CI; run them before you push:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going docs docs/_build/html
```

Test markers:

- `slow`: long-running tests (subprocess kills, template creation). They run in CI.
- `chronocratic`: tests that need the `chronocratic-models` and
  `chronocratic-datasets` extras. `uv run pytest -m "not chronocratic"` skips them.

## Code style

- Type hints on every function, including return types. `ty` runs in strict mode.
- Google-style docstrings on public modules, classes and functions.
- Keyword arguments in calls.
- `ruff` with every rule enabled; an ignore needs a `# why:` comment.
- No real datasets in the repository. Tests use small synthetic data that has
  the same format.

## Branches

- `main` holds released code. It only moves by fast-forward from `dev`.
- `dev` is the integration branch. Branch from `dev` and open your pull
  request against `dev`.
- History is linear: rebase your branch on `dev` instead of merging `dev` into it.
- A maintainer moves `main` forward by commenting `/fast-forward` on the
  `dev → main` pull request; `/check-fast-forward` only reports whether it can.

## Changelog fragments

Every pull request into `dev` adds one fragment in `changelog.d/`. The
`Changelog fragment` workflow writes `changelog.d/<pr>.<type>.md` from the
pull request title for you:

- the type comes from a `changelog:<type>` label, or from the Conventional
  Commits prefix of the title (`feat` gives `added`, `fix` gives `fixed`,
  anything else gives `changed`);
- a `## Summary` section in the pull request description is appended to the entry;
- the `skip-changelog` label opts out (for changes users never see).

Edit the fragment by hand when the title is not enough. See
[`changelog.d/README.md`](https://github.com/skafai/tslab/blob/main/changelog.d/README.md) for the file names and types.

## Protected files

Changes to `pyproject.toml`, `ruff.toml` or `uv.lock` affect every user and
contributor. The `Protected Files Check` workflow adds a comment to pull
requests that touch them so that a maintainer checks that:

- new dependencies are needed and have version bounds;
- new lint ignores do not hide real problems;
- the change does not break CI or existing projects.

## Documentation

The docs live in `docs/` (Sphinx with MyST Markdown) and are published on
[Read the Docs](https://tslab.readthedocs.io). Each pull request gets a
preview build. Public functions are documented from their docstrings on the
API page.
