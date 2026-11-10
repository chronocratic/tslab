# ts: time series

`tslab.ts` holds everything specific to time series: the encoder and
downstream model interfaces, the encode and evaluate stages, forecasting and
classification tasks, probes, the chronocratic adapters and the `ts` project
template. It builds on [ml](../ml/index.md) and the [lab](../lab/guide.md).

The ts stages import `tslab.ml`, so runs that use them are seeded for torch and
record the torch metadata. See [Seeding](../ml/index.md#seeding).

```{toctree}
:maxdepth: 1

getting-started
chronocratic
extending
```
