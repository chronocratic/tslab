# TSLab

TSLab turns a research script into a pipeline of **cached, resumable stages**.
You keep writing a normal Python `main()`; TSLab gives each step a
fingerprint, stores its result, and on the next run loads finished results,
resumes unfinished training, and reruns only what changed or failed.

It ships ready stages for time series representation learning (train, encode,
evaluate with linear probes), Lightning training with safe resume and SLURM
time limits, experiment trackers, and adapters for the chronocratic model and
dataset libraries.

```bash
uv add "tslab[chronocratic-models,chronocratic-datasets]"
uvx tslab new my-paper
```

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
concepts
resume-and-failures
run-options
configs-and-presets
studies
tracking
hpc
extending
chronocratic
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
changelog
contributing
```
