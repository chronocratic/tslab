# chronocratic adapters

`tslab[chronocratic-models]` and `tslab[chronocratic-datasets]` connect the
[chronocratic](https://github.com/chronocratic) libraries to TSLab. The two
adapters are independent: either can be used without the other.

## Models

```python
from tslab.adapters.chronocratic_models import (
    ChronocraticEncoder,
    build_chronocratic_model,
    chronocratic_model_choice,
    model_settings,
)

Models = chronocratic_model_choice(augmentations=(JitterThenScaleRecipe,))
```

- Every model chronocratic exports is offered as a subcommand: `model:ts2vec`,
  `model:tstcc`, `model:cost`, ... Its settings are generated from its
  parameter class, so `--model.<field>` flags and `--help` follow chronocratic.
- `input_dim` always comes from the data. `sequence_length`, where a model has
  it, is the flag's value, otherwise the data's length; forecasting data has no
  single length, so CoST, TimeVAE and TST then need `--model.sequence-length`.
- `build_chronocratic_model(config=..., data_info=data.info)` builds the model;
  chronocratic's own checks run there.
- `model_settings(choice=Models, name='ts2vec', max_train_length=128)` builds
  settings in code, for presets.

### Augmentation recipes

A recipe is a frozen dataclass whose `__call__` builds a chronocratic
augmentation producer; its return annotation names the view type:

```python
@dataclass(frozen=True)
class JitterThenScaleRecipe:
    sigma: float = 0.8
    ratio: float = 1.1

    def __call__(self) -> AugmentationProducer[ViewPair]:
        return RolePairProducer(
            first=Jitter(params=JitterParameters(sigma=self.sigma)),
            second=Scaling(params=ScalingParameters(sigma=self.ratio)),
        )
```

A recipe appears only on models with that view type (`ViewPair`: TSTCC, MHCCL,
SimCLR, CoST; `AlignedPair`: TS2Vec; `SingleView`: AutoTCL):
`model:tstcc model.augmentation:jitter-then-scale --model.augmentation.sigma 0.5`.
`model.augmentation:default` keeps the model's built-in augmentation. A
recipe given to a model of another view type is rejected when the command
line is parsed. The recipe's settings are part of the training fingerprint.

### Encoding

`ChronocraticEncoder` (the `encoder` argument of the encode stages) calls
chronocratic's encoding. `--encoder.batch-size` and
`--encoder.sliding-padding` are fingerprinted; `--encoder.device` (default
`auto`) and `--encoder.num-workers` are not. TS2Vec, AutoTCL and CoST also
encode whole series step by step, which the forecasting stages use to encode
each series once for all horizons.

## Datasets

```python
from tslab.adapters.chronocratic_datasets import (
    ChronocraticDataChoice,
    build_chronocratic_data_source,
    data_settings,
)
```

- Every datamodule chronocratic exports is a subcommand: `data:ucr`,
  `data:uea`, `data:ett`, `data:electricity`, `data:weather`, with its
  constructor arguments as flags.
- `build_chronocratic_data_source(config=...)` returns a
  `ChronocraticClassificationData` or a `ChronocraticForecastingData`; check
  which with `isinstance` before passing it to a task's stages.
- In forecasting data, the time features chronocratic adds come first and are
  covariates (`data.info.covariate_channels`); the rest are targets.
- Dataset files are fingerprinted by **path, not content**: replacing a file in
  place keeps old results. Use `--run.force-all` after changing data.

## Evaluation recipes

The defaults are chronocratic's. These commands give other common protocols;
add the usual `--data.*` path flags.

### Classification

```bash
# the archive's own train/test split (the default)
python -m my_paper.classify no-preset model:ts2vec data:ucr

# train and test merged and re-split 50/50, stratified, batch size 16
python -m my_paper.classify no-preset model:ts2vec data:ucr \
    --data.splitting-strategy MANUAL --data.batch-size 16
#   CoST: add --model.max-train-length 1000000 (its architecture uses this length)
```

### Forecasting

```bash
# the whole training series (the default)
python -m my_paper.forecast no-preset model:ts2vec data:ett --data.variant ETTh1

# random 128-step training crops, batch size 1
python -m my_paper.forecast no-preset model:ts2vec data:ett --data.variant ETTh1 \
    --data.batch-size 1 --model.max-train-length 128
#   ts2vec, autotcl, mhccl, simclr:   --model.max-train-length 128
#   cost, timevae, tst:               --model.max-train-length 128 --model.sequence-length 128
#   mcl, tstcc, recurrentautoencoder, series2vec, timenet: no crop setting
```

For long classification series, `--model.max-train-length 3000` follows the
TS2Vec paper.

## Known limitations

- chronocratic-models 0.1.0a20 draws part of the training randomness of
  TS2Vec, AutoTCL and CoST (training crops, encoder masks, TS2Vec's
  crop-and-shift views) from NumPy generators that no seed controls. Two
  trainings of these models with the same seed can therefore differ.
- A model's built-in augmentation (`default`) is not part of the fingerprint
  beyond the model's settings; its constants change only with the
  chronocratic-models version.
