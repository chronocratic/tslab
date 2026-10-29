# TSLab

**A batteries-included lab for time series research.**
TSLab turns a research script into a pipeline of **cached, resumable stages**.
Keep writing a normal Python `main()`. TSLab saves each stage result, so a
second run reuses finished work and reruns only steps affected by a change.
Interrupted Lightning training resumes from its last checkpoint.

## Start here

- {doc}`getting-started` — install TSLab, create a project and run it on toy data.
- {doc}`guide` — learn about caching, configure runs and manage experiments.
- {doc}`api` — look up public Python interfaces and commands.

## How a run works

Your script chooses the order of stages. TSLab ships stages to **train** a
model, **encode** time series and **evaluate** representations with probes.
You can add your own stages anywhere in that flow.

Each stage call saves an artifact and a fingerprint of its settings and inputs.
Run the script again to load matching results. Change a setting and TSLab
recomputes the affected stages and their downstream work, not the whole run.
If training stops early or a stage fails, inspect the run and continue from
the work already saved. See {doc}`concepts` and {doc}`resume-and-failures`.

## Find your workflow

- **Choose models and datasets:** {doc}`configs-and-presets` and
  {doc}`chronocratic`.
- **Run many experiments:** {doc}`studies` for grids on a laptop or cluster;
  {doc}`hpc` for SLURM time limits and resubmission.
- **Follow results:** {doc}`tracking` for CSV and optional experiment trackers.
- **Use your own components:** {doc}`extending` for stages, trainers, encoders
  and data sources.

## Guides

- {doc}`agents` — drive study runs with JSON output and exit codes.

```{toctree}
:maxdepth: 2
:hidden:

getting-started
guide
agents
reference
```
