`skip_busy_gpus` also skips GPUs with more than `busy_gpu_memory_fraction`
(default 0.1) of their memory in use, which catches processes nvidia-smi does not list.
