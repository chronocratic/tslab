# expect-error: invalid-argument-type
import torch

from tslab.ts.stages.encoders import ModelAsEncoder

ModelAsEncoder()(torch.nn.Linear(in_features=2, out_features=2))
