# Changelog fragments

Each change that users should hear about gets one small Markdown file here.
At release time, `towncrier build` moves them into `CHANGELOG.md`.

## File names

- `<pr>.<type>.md` for a pull request, for example `42.added.md`. The
  `auto-changelog-fragment` workflow writes this file from the pull request
  title; add or edit it by hand to say more.
- `+<name>.<type>.md` for a change without a pull request number, for example
  `+run-heartbeat.added.md`.

## Types

| Type | Use for |
|------|---------|
| `added` | new features |
| `changed` | changes to existing behavior |
| `deprecated` | features that will be removed |
| `removed` | removed features |
| `fixed` | bug fixes |
| `security` | security fixes |

## Commands

```bash
uv run towncrier build --draft --version 0.1.0a1   # preview the notes of the next release
uv run towncrier check --compare-with origin/dev   # does this branch add a fragment?
```
