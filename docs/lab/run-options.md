# Run options

Add `run: tslab.lab.RunOptions = field(default_factory=tslab.lab.RunOptions)` to your
config and these flags exist. They never change a fingerprint.

| Flag | Effect |
|---|---|
| `--run.force-all` | run every instance again |
| `--run.rerun-from STAGE` | run this stage and every stage called after it again |
| `--run.rerun ID ...` | run these instances again, for example `--run.rerun 'measure[small/a]'` |
| `--run.fail-fast` | stop at the first failing instance |
| `--run.dry-run` | print what would be loaded or run; write nothing |
| `--run.log-level LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR`; in Python, a `tslab.lab.LogLevel` member |

A `--run.rerun` id or `--run.rerun-from` stage that matches nothing gives a
warning at the end of the run, so a typo does not go unnoticed.

## Seeding

At the start of a run, the lab seeds Python `random` and NumPy from the seed of
the run and sets `PYTHONHASHSEED`.

Torch is not seeded by the lab. Importing `tslab.ml` (directly, or through any
`tslab.ml` or `tslab.ts` module) plugs into the lab: from then on every run also
seeds torch on all devices and records the torch, Lightning, CUDA and GPU
metadata. A stage that uses torch without importing `tslab.ml` is not seeded
for torch and records no GPU block. Import `tslab.ml` in any stage that uses
torch. See [Seeding in `tslab.ml`](../ml/index.md#seeding).
