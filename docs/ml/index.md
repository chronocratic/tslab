# ml: torch and Lightning

`tslab.ml` adds torch and Lightning on top of the lab: the `train` stage, the
trainer and data source interfaces, the `Tensors` and `Checkpoint` artifacts, and
the experiment trackers. It never depends on `tslab.ts`.

## Seeding

The lab seeds Python `random` and NumPy and sets `PYTHONHASHSEED`. Importing
`tslab.ml` adds more, whether you import it directly or through any `tslab.ml` or
`tslab.ts` module:

- torch is seeded on all devices at the start of each run;
- `meta.json` records the torch, Lightning, CUDA and GPU metadata.

A stage that uses plain torch without importing `tslab.ml` is not seeded for
torch and records no GPU block. Import `tslab.ml` in any stage that uses torch.
The Lightning trainer and the ts stages import it for you. Importing `tslab.ml`
more than once seeds only once. See also [Seeding in the lab](../lab/run-options.md#seeding).

## Pages

```{toctree}
:maxdepth: 1

tracking
extending
```
