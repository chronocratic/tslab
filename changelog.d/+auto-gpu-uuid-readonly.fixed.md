A local profile with `gpus = "auto"` no longer breaks `study status`, `results` and `cancel` when `CUDA_VISIBLE_DEVICES` holds a UUID; `study submit` still reports the bad id.
