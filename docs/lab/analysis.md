# Analyse study results

`tslab.lab.analysis` turns the results of a study into tests, a critical-difference diagram, a
numbers file for a paper and an organised folder of tables and figures. It needs the
`analysis` extra:

```bash
pip install "tslab[analysis]"
```

Only this package imports pandas, SciPy and matplotlib. Library functions never print and
never save: the caller passes tables and figures to an output tree.

## Load the results

```python
from pathlib import Path
import os

from tslab.lab.analysis import load_results

results = load_results(
    study=example,
    machine=Path('machines/local.toml'),
    project_root=Path('..'),
    environment=os.environ,
)
```

The frame has one row per run and instance: the axes of the study as columns, `run_path`,
the states, and one `metric/<name>` column per metric. A notebook usually starts in
`notebooks/`, but the machine profile and its relative `output_root` are read against the
working folder of the command line. `load_results` therefore reads inside `project_root`
and restores the working folder after. The working folder is process wide, so do not call
it from two threads at once.

`missing_runs(results=results)` lists every run that has no finished instance. A metric is
described by `AnalysisMetric(name=..., label=..., direction=...)`, where the direction says
whether a larger value is better.

## Compare methods

One run has several metric rows: the template evaluates once on clean data and once per
noise level, under the same metric names. Pick the `instance` first, then build the matrix:

```python
from tslab.lab.analysis import cd_diagram, method_matrix, pairwise_wilcoxon

scores = results[results['instance'] == 'evaluate[default]']
matrix = method_matrix(
    frame=scores, unit='dataset', method='recipe', value='metric/accuracy', average_over=['seed']
)
diagram = cd_diagram(matrix=matrix, direction=metric.direction)
tests = pairwise_wilcoxon(matrix=matrix)
```

`method_matrix` raises when two rows share the same dataset, method and seed, so clean and
noisy scores are never averaged silently. `cd_diagram` and the Friedman test use only the
datasets that every method finished, and the caption says how many that is. `diagram.figure`
is `None`, with the reason in `diagram.caption`, when there are too few complete datasets or
the methods do not differ. Pass `reverse=True` to put rank 1 on the right, and a `CdStyle` as `style=` to change the room for method names (`text_space`, in inches), tick lengths, spacing and line widths. `average_ranks`, `friedman`, `bootstrap_median`,
`rank_agreement` and `correlations` cover the rest of the usual comparison, and
`against_baseline` compares each condition with a baseline condition.

## Save into the output tree

```python
from tslab.lab.analysis import AnalysisOutput, FigureFormat, OutputFolder

output = AnalysisOutput(root=Path('analysis'), formats=(FigureFormat.PDF, FigureFormat.PNG))
output.save_figure(figure=diagram.figure, category='critical_difference', name='cd_accuracy')
output.save_table(frame=tests, folder=OutputFolder.STATS, name='pairwise_wilcoxon_accuracy')
```

The tree is the same every time:

```
analysis/
  results_numbers.tex, results_numbers.csv   the only files at the top
  stats/<name>.csv                           test results
  tables/<name>.csv                          metrics and summaries
  coverage/<name>.csv                        missing and failed runs
  plots/<category>/<format>/<name>.<format>  one folder per chosen format
  <child>/                                   a nested tree, from output.child(name=...)
```

Names are lowercase slugs (`[a-z0-9][a-z0-9_.-]*`), general to specific: `<what>_<slice>`.
A bad name raises, so no file lands outside the tree. Files are written atomically, and a
figure that fails to render leaves no file. Pass `formats=` to `save_figure` to override the
file types for one figure; use `paper_style()` around the plotting code for one font size.

## Numbers for a paper

```python
from datetime import datetime, UTC

from tslab.lab.analysis import NumberKind, number_key, numbers_frame, write_numbers

numbers = numbers_frame(
    records=[(number_key(parts=['friedman', 'accuracy', 'p']), 1.2e-5, NumberKind.P)]
)
write_numbers(numbers=numbers, output=output, source='analysis.ipynb', now=datetime.now(tz=UTC))
```

In LaTeX, `\input{results_numbers}` and then `\res{friedman-accuracy-p}`. Text values (`NumberKind.TEXT`) are escaped for LaTeX, and `number_key` raises on a part that TeX cannot take in a macro name (`% \ { } # & ~ ^`). A rerun updates
the paper and no number is copied by hand.

## The template notebook

A project made by `tslab new` has `notebooks/analysis.ipynb`. It loads the results, lists the
missing runs and then compares methods. The example study has too little data for a
comparison, so those sections run on made-up data. Every table, figure and caption made from
it says `SYNTHETIC DATA`, its files start with `synthetic_` and go to `analysis/synthetic/`,
and the notebook shows the one line that swaps in the real results.
