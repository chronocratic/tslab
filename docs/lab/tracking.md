# Tracking

A local tracker is always on: metrics land in `trackers/csv/version_<n>/` of
the run folder, one `version` per execution, with the config as YAML. It is the
record when other trackers fail. Pass more trackers to `Run(trackers=...)`; each
subclasses `tslab.lab.Tracker` (see [Extending the lab](extending.md#a-tracker)).

Each tracker receives the config (flattened with `/`) and every `Metrics`
result. The final state of the run (`finished` or `failed`) is sent at the
end. A tracker that fails never fails the run: it is logged and skipped.

The ready trackers for Weights & Biases, MLflow and TensorBoard are in
`tslab.ml`; see [Tracking in ml](../ml/tracking.md).
