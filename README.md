# TSLab

Resumable, cached research pipelines for time-series representation learning.

> Work in progress: v0.1 is under development. Nothing here is stable yet.

## Development

```bash
uv sync                      # create .venv with the dev tools
uv run pytest                # tests
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

TSLab is released under the BSD-3-Clause license (see `LICENSE`).
